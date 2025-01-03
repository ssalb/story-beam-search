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
        Generate story continuations using parallel beam search iterations.
        """
        instructions = (
            f"Continue the following story in the {genre} genre, "
            "ensuring coherence with the tone, characters, and narrative established so far:\n"
        )
        instructions_len = len(instructions)

        stories = self._generate_batch([instructions + prompt])
        ranked_stories = evaluator.evaluate_multiple(
            [story[instructions_len:] for story in stories]
        )
        stories = [story for story, _ in ranked_stories[: self.config.num_beams]]

        if stories:
            for _ in range(self.config.num_iterations):
                # Prepare all prompts for batch processing
                all_prompts = [instructions + story for story in stories]
                # Generate all continuations in one batch
                all_stories = self._generate_batch(all_prompts)

                ranked_stories = evaluator.evaluate_multiple(
                    [story[instructions_len:] for story in all_stories]
                )
                stories = [
                    story for story, _ in ranked_stories[: self.config.num_beams]
                ]

        return stories

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        """
        Generate multiple continuations for multiple prompts in a single batch.
        """
        # Tokenize all prompts
        tokenized = [self.tokenizer(prompt, return_tensors="pt") for prompt in prompts]

        # Pad input_ids and attention_masks to same length
        max_length = max(inputs["input_ids"].size(1) for inputs in tokenized)
        padded_input_ids = []
        padded_attention_masks = []

        for inputs in tokenized:
            input_ids = inputs["input_ids"][0]
            attention_mask = inputs["attention_mask"][0]

            # Pad to max_length
            padding_length = max_length - input_ids.size(0)
            if padding_length > 0:
                input_ids = torch.cat(
                    [input_ids, torch.zeros(padding_length, dtype=torch.long)]
                )
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros(padding_length, dtype=torch.long)]
                )

            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)

        # Stack into batches
        input_ids_batch = torch.stack(padded_input_ids).to(self.device)
        attention_mask_batch = torch.stack(padded_attention_masks).to(self.device)

        # Calculate continuation length
        continuation_length = (
            max_length + self.config.max_length // self.config.num_iterations
        )

        # Generate all continuations in one pass
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                max_length=continuation_length,
                num_beams=self.config.num_beams,
                num_return_sequences=self.config.num_return_sequences,
                early_stopping=True,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=True,
            ).to(self.device)

        stories = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            stories.append(text)

        return stories
