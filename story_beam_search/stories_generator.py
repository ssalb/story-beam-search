from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForMaskedLM,
    pipeline,
    Pipeline
)
import torch

@dataclass
class ModelConfig:
    text_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    bert_name: str = "bert-base-uncased" # "answerdotai/ModernBERT-base"
    zero_shot_name: str = "facebook/bart-large-mnli"
    device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Models:
    """Container for all loaded models and tokenizers."""
    device: torch.device
    text_model: AutoModelForCausalLM
    text_tokenizer: AutoTokenizer
    bert_model: AutoModelForMaskedLM
    bert_tokenizer: AutoTokenizer
    zero_shot_pipeline: Pipeline

class ModelLoader:
    """Handles loading and initialization of all required models."""
    
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        self.device = torch.device(config.device)

    def load_models(self) -> Models:
        """Load all required models and return them in a Models container."""
        
        # Load Text model for writting stories
        print(f"Loading Text model ({self.config.text_model_name})...")
        text_tokenizer = AutoTokenizer.from_pretrained(self.config.text_model_name)
        text_model = AutoModelForCausalLM.from_pretrained(self.config.text_model_name).to(self.device)
        text_model.eval()

        # Load BERT model for coherence and fluency scoring
        print(f"Loading BERT model ({self.config.bert_name})...")
        bert_tokenizer = AutoTokenizer.from_pretrained(self.config.bert_name)
        bert_model = AutoModelForMaskedLM.from_pretrained(self.config.bert_name).to(self.device)
        bert_model.eval()

        # Load Zero-Shot classification pipeline for genre alignment scoring
        print("Loading Zero-Shot Classification pipeline...")
        zero_shot_pipeline = pipeline(
            "zero-shot-classification",
            model=self.config.zero_shot_name,
            device=self.device
        )

        return Models(
            device=self.device,
            text_model=text_model,
            text_tokenizer=text_tokenizer,
            bert_model=bert_model,
            bert_tokenizer=bert_tokenizer,
            zero_shot_pipeline=zero_shot_pipeline
        )

class StoryGenerationSystem:
    """
    High-level class that coordinates model loading and initialization of all components.
    Acts as a facade for the entire story generation system.
    """
    
    def __init__(self, model_config: ModelConfig = ModelConfig()):
        self.model_loader = ModelLoader(model_config)
        self.models = None
        self.beam_search = None
        self.evaluator = None

    def initialize(self):
        """Initialize all components of the story generation system."""
        from story_beam_search.beam_search import BeamSearchGenerator, BeamSearchConfig
        from story_beam_search.scoring import (
            CoherenceScorer,
            FluencyScorer,
            GenreAlignmentScorer,
            StoryEvaluator
        )

        # Load all models
        self.models = self.model_loader.load_models()

        # Initialize beam search
        self.beam_search = BeamSearchGenerator(
            model=self.models.text_model,
            tokenizer=self.models.text_tokenizer,
            device=self.models.device,
            config=BeamSearchConfig()
        )

        # Initialize scorers
        coherence_scorer = CoherenceScorer(
            model=self.models.bert_model,
            tokenizer=self.models.bert_tokenizer,
            device=self.models.device
        )

        fluency_scorer = FluencyScorer(
            model=self.models.text_model,
            tokenizer=self.models.text_tokenizer,
            device=self.models.device
        )

        # Note: genre_scorer will be created per request as it depends on the user's genre choice
        self.create_evaluator = lambda genre: StoryEvaluator(
            coherence_scorer=coherence_scorer,
            fluency_scorer=fluency_scorer,
            genre_scorer=GenreAlignmentScorer(
                pipeline=self.models.zero_shot_pipeline,
                genre=genre
            )
        )

    def generate_and_evaluate(self, prompt: str, genre: str, num_stories: int = 3):
        """Generate stories and evaluate them."""
        if not self.models:
            raise RuntimeError("System not initialized. Call initialize() first.")

        # Create evaluator with specified genre
        evaluator = self.create_evaluator(genre)

        # Generate stories
        # This is not strict beam search, inspired by https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute
        # to generate more diverse stories
        all_stories = []
        for _ in range(num_stories):
            stories = self.beam_search.generate_iterations(prompt, genre, evaluator)
            print("in loop")
            ranked_stories = evaluator.evaluate_multiple(stories)
            # keep the top story of this beam search iteration
            all_stories.append(ranked_stories[0][0])

        # Evaluate stories once more
        print("final evaluation")
        ranked_stories = evaluator.evaluate_multiple(all_stories)
        # Return top k stories with their scores
        return ranked_stories[:num_stories]