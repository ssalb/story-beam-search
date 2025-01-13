import numpy as np
from dataclasses import dataclass
from typing import Protocol
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, Pipeline
from sklearn.metrics.pairwise import cosine_similarity


class StoryScorer(Protocol):
    """Protocol defining the interface for story scoring components."""

    def score(self, stories: list[str]) -> float:
        """Return a score between 0 and 1."""
        ...


@dataclass
class CombinedScore:
    coherence: float = 0.0
    fluency: float = 0.0
    genre_alignment: float = 0.0
    total: float = 0.0


class CoherenceScorer(StoryScorer):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        batch_size: int = 32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def score(self, stories: list[str]) -> list[float]:
        """Calculate coherence score based on sentences cosine similarity."""
        all_scores = []

        # Split stories into sentences for coherence scoring
        sentences_list = [
            [s.strip() for s in story.split(".") if s.strip()] for story in stories
        ]

        # Collect all sentence pairs that need embedding
        all_sentence_pairs = []
        story_boundaries = []  # Track where each story's sentences end
        current_position = 0

        for sentences in sentences_list:
            pairs_count = len(sentences) - 1
            all_sentence_pairs.extend(zip(sentences[:-1], sentences[1:]))
            story_boundaries.append(current_position + pairs_count)
            current_position += pairs_count

        # Process sentence pairs in batches
        all_embeddings = []
        for i in range(0, len(all_sentence_pairs), self.batch_size):
            batch_pairs = all_sentence_pairs[i : i + self.batch_size]
            # Flatten pairs for batch processing
            batch_sentences = [sent for pair in batch_pairs for sent in pair]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_sentences, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                batch_embeddings = outputs.hidden_states[-1][
                    :, 0, :
                ]
                all_embeddings.extend(batch_embeddings.cpu().numpy())

        # Calculate coherence scores for each story
        current_idx = 0
        for boundary in story_boundaries:
            story_pairs_count = boundary - current_idx
            story_scores = []

            for i in range(story_pairs_count):
                idx = current_idx + i
                first_emb = all_embeddings[idx * 2].reshape(1, -1)
                second_emb = all_embeddings[idx * 2 + 1].reshape(1, -1)
                sim = cosine_similarity(first_emb, second_emb)[0][0]
                story_scores.append(sim)

            avg_coherence = (
                sum(story_scores) / len(story_scores) if story_scores else 0.0
            )
            all_scores.append(avg_coherence)
            current_idx = boundary

        return all_scores


class FluencyScorer(StoryScorer):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        batch_size: int = 32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

        # Set up padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Add padding token to tokenizer only
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Set up mask token if it doesn't exist
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    def score(self, stories: list[str]) -> list[float]:
        all_scores = []
        mask_token_id = self.tokenizer.mask_token_id

        # Process stories in batches
        for i in range(0, len(stories), self.batch_size):
            batch_stories = stories[i : i + self.batch_size]
            batch_inputs = self.tokenizer(
                batch_stories, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            batch_scores = []

            # For each story in the batch
            for j in range(len(batch_stories)):
                story_scores = []
                input_ids = batch_inputs.input_ids[j : j + 1]
                attention_mask = batch_inputs.attention_mask[j : j + 1]

                # Only process tokens that aren't padding
                valid_tokens = attention_mask[0].sum().item()

                # For each token in the story (excluding padding)
                for k in range(1, valid_tokens - 1):
                    masked_input_ids = input_ids.clone()
                    masked_input_ids[0, k] = mask_token_id

                    # Ensure token is within vocab range
                    masked_input_ids = masked_input_ids.clamp(
                        0, self.tokenizer.vocab_size - 1
                    )

                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=masked_input_ids, attention_mask=attention_mask
                        )
                        logits = outputs.logits

                    original_token_id = input_ids[0, k]
                    token_probability = (
                        logits[0, k].softmax(dim=-1)[original_token_id].item()
                    )
                    story_scores.append(token_probability)

                avg_fluency = (
                    sum(story_scores) / len(story_scores) if story_scores else 0.0
                )
                batch_scores.append(avg_fluency)

            all_scores.extend(batch_scores)

        return all_scores


class GenreAlignmentScorer(StoryScorer):
    def __init__(self, pipeline: Pipeline, genre: str, batch_size: int = 32):
        self.pipeline = pipeline
        self.genre = genre
        self.batch_size = batch_size

    def score(self, stories: list[str]) -> list[float]:
        if not self.genre:
            return [0.5] * len(stories)

        all_scores = []
        # Split all stories into sentences
        all_sentences = []
        story_boundaries = []
        current_position = 0

        for story in stories:
            sentences = [s.strip() for s in story.split(".") if s.strip()]
            all_sentences.extend(sentences)
            story_boundaries.append(current_position + len(sentences))
            current_position += len(sentences)

        # Process sentences in batches
        all_sentence_scores = []
        for i in range(0, len(all_sentences), self.batch_size):
            batch_sentences = all_sentences[i : i + self.batch_size]
            results = self.pipeline(
                batch_sentences,
                candidate_labels=[self.genre],
                multi_label=True,
                batch_size=self.batch_size,
            )
            all_sentence_scores.extend([r["scores"][0] for r in results])

        # Calculate average score for each story
        current_idx = 0
        for boundary in story_boundaries:
            story_scores = all_sentence_scores[current_idx:boundary]
            avg_score = sum(story_scores) / len(story_scores) if story_scores else 0.0
            all_scores.append(avg_score)
            current_idx = boundary

        return all_scores


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

    def evaluate_multiple(self, stories: list[str]) -> list[tuple[str, CombinedScore]]:
        """Evaluate multiple stories in batches and return them sorted by total score."""

        # Get all scores in parallel using batch processing
        coherence_scores = self.coherence_scorer.score(stories)
        fluency_scores = self.fluency_scorer.score(stories)
        genre_scores = self.genre_scorer.score(stories)

        # Find max scores for normalization
        max_scores = [max(coherence_scores), max(fluency_scores), max(genre_scores)]

        # Create scored stories
        scored_stories = []
        for i, story in enumerate(stories):
            scores = CombinedScore(
                coherence=(
                    coherence_scores[i] / max_scores[0] if max_scores[0] != 0 else 0
                ),
                fluency=fluency_scores[i] / max_scores[1] if max_scores[1] != 0 else 0,
                genre_alignment=(
                    genre_scores[i] / max_scores[2] if max_scores[2] != 0 else 0
                ),
            )
            scores.total = np.dot(
                [scores.coherence, scores.fluency, scores.genre_alignment], self.weights
            )
            scored_stories.append((story, scores))

        return sorted(scored_stories, key=lambda x: x[1].total, reverse=True)
