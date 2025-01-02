from pydantic.dataclasses import dataclass
from typing import Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from story_beam_search.scoring import StoryScorer


@dataclass
class BeamSearchConfig:
    num_beams: int = 3
    num_return_sequences: int = 3
    max_length: int = 100
    no_repeat_ngram_size: int = 2
    temperature: float = 0.8
    top_k: int = 8
    top_p: float = 0.95
    num_iterations: int = 3
    continuation_length: int = 10


class BeamSearchGenerator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        config: Optional[BeamSearchConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or BeamSearchConfig()

    def generate_iterations(
        self, prompt: str, genre: str, evaluator: StoryScorer
    ) -> list[str]:
        """
        Generate story continuations using multiple iterations of beam search.
        """

        # Adding some instructions to the prompt. These are removed in the end
        instructions = (
            f"Continue the following story in the {genre} genre, "
            "ensuring coherence with the tone, characters, and narrative established so far:\n"
        )
        instructions_len = len(instructions)

        stories = self._generate_single_iteration(instructions + prompt)
        ranked_stories = evaluator.evaluate_multiple(
            [story[instructions_len:] for story in stories]
        )

        stories = [story for story, _ in ranked_stories[:self.config.num_beams]]

        if stories:
            for _ in range(self.config.num_iterations):
                all_stories = []
                for story in stories:
                    continuations = self._generate_single_iteration(
                        instructions + story
                    )
                    all_stories.extend(continuations)
                ranked_stories = evaluator.evaluate_multiple(
                    [story[instructions_len:] for story in all_stories]
                )
                stories = [story for story, _ in ranked_stories[:self.config.num_beams]]

        return stories

    def _generate_single_iteration(self, prompt: str) -> list[str]:
        """
        Generate multiple continuations for a single iteration using beam search.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        self.config.continuation_length = (
            len(input_ids[0]) + self.config.max_length // self.config.num_iterations
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.continuation_length,
                num_beams=self.config.num_beams,
                num_return_sequences=self.config.num_return_sequences,
                early_stopping=True,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=True,
            )

        stories = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            stories.append(text)

        return stories
