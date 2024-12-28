import numpy as np
from dataclasses import dataclass
from typing import Protocol
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, Pipeline
from sklearn.metrics.pairwise import cosine_similarity


class StoryScorer(Protocol):
    """Protocol defining the interface for story scoring components."""

    def score(self, story: str) -> float:
        """Return a score between 0 and 1."""
        ...


@dataclass
class CombinedScore:
    coherence: float
    fluency: float
    genre_alignment: float
    total: float


class CoherenceScorer(StoryScorer):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        max_pairs: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_pairs = max_pairs

    def score(self, story: str) -> float:
        """Calculate coherence score based on sentences cosine similarity."""

        sentences = [s.strip() for s in story.split(".") if s.strip()]

        embeddings = []

        # Generate embeddings for each sentence
        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model.bert(**inputs).last_hidden_state[:, 0, :]
                embeddings.append(emb.cpu().numpy())

        # Calculate cosine similarity between adjacent embeddings
        coherence_scores = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])[0][0]
            coherence_scores.append(sim)

        # Average coherence score
        avg_coherence = (
            sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        )
        return avg_coherence


class FluencyScorer(StoryScorer):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def score(self, story: str) -> float:
        # Mask each token in the story and calculate the probability of the original token
        # Fluency is measured by the average probability of each token in the story
        inputs = self.tokenizer(story, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        mask_token_id = self.tokenizer.mask_token_id

        if mask_token_id is None:
            self.tokenizer.mask_token = "[MASK]"
            mask_token_id = self.tokenizer.encode(self.tokenizer.mask_token)[0]

        fluency_scores = []
        for i in range(1, input_ids.size(1) - 1):
            masked_input_ids = input_ids.clone()
            masked_input_ids[0, i] = mask_token_id

            with torch.no_grad():
                outputs = self.model(input_ids=masked_input_ids)
                logits = outputs.logits

            original_token_id = input_ids[0, i]
            token_probability = logits[0, i].softmax(dim=-1)[original_token_id].item()
            fluency_scores.append(token_probability)

        avg_fluency = (
            sum(fluency_scores) / len(fluency_scores) if fluency_scores else 0.0
        )
        return avg_fluency


class GenreAlignmentScorer(StoryScorer):
    def __init__(self, pipeline: Pipeline, genre: str):
        self.pipeline = pipeline
        self.genre = genre

    def score(self, story: str) -> float:
        if not self.genre:
            return 0.5

        # Evaluate by sentence to check whether the genre is maintained throughout
        sentences = [s.strip() for s in story.split(".") if s.strip()]
        results = []
        for sentence in sentences:
            result = self.pipeline(
                sentence, candidate_labels=[self.genre], multi_label=True
            )
            results.append(result["scores"][0])

        avg_core = sum(results) / len(results) if results else 0.0
        return avg_core


class StoryEvaluator:
    def __init__(
        self,
        coherence_scorer: CoherenceScorer,
        fluency_scorer: FluencyScorer,
        genre_scorer: GenreAlignmentScorer,
        weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    ):
        self.coherence_scorer = coherence_scorer
        self.fluency_scorer = fluency_scorer
        self.genre_scorer = genre_scorer
        self.weights = weights

    def evaluate(self, story: str, max_scores: list[float]) -> CombinedScore:
        coherence = self.coherence_scorer.score(story)
        fluency = self.fluency_scorer.score(story)
        genre_alignment = self.genre_scorer.score(story)

        max_scores[0] = np.max([max_scores[0], coherence])
        max_scores[1] = np.max([max_scores[1], fluency])
        max_scores[2] = np.max([max_scores[2], genre_alignment])

        return CombinedScore(
            coherence=coherence,
            fluency=fluency,
            genre_alignment=genre_alignment,
            total=0,
        )

    def evaluate_multiple(self, stories: list[str]) -> list[tuple[str, CombinedScore]]:
        """Evaluate multiple stories and return them sorted by total score."""

        # Scores are normalized by the max scores on every evaluation
        # This is to ensure that the scores are comparable between each other, as they are originally on different scales
        
        # Reset max scores
        max_scores = [0.0, 0.0, 0.0]

        scored_stories = [
            (story, self.evaluate(story, max_scores)) for story in stories
        ]

        # Normalize scores
        for _, scores in scored_stories:
            scores.coherence, scores.fluency, scores.genre_alignment = np.divide(
                [scores.coherence, scores.fluency, scores.genre_alignment],
                max_scores,
            )
            scores.total = np.dot(
                [scores.coherence, scores.fluency, scores.genre_alignment], self.weights
            )

        return sorted(scored_stories, key=lambda x: x[1].total, reverse=True)
