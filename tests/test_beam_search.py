import pytest
from unittest.mock import MagicMock
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from story_beam_search.beam_search import BeamSearchGenerator, BeamSearchConfig
from story_beam_search.scoring import StoryEvaluator, CombinedScore

@pytest.fixture
def mock_model():
    return MagicMock(spec=PreTrainedModel)

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    # tokenizer.return_tensors = "pt"
    tokenizer.side_effect = lambda text, return_tensors: {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]])
    }
    return tokenizer

@pytest.fixture
def mock_device():
    return torch.device("cpu")

@pytest.fixture
def mock_evaluator():
    return MagicMock(spec=StoryEvaluator)

@pytest.fixture
def beam_search_generator(mock_model, mock_tokenizer, mock_device):
    config = BeamSearchConfig()
    return BeamSearchGenerator(mock_model, mock_tokenizer, mock_device, config)

def test_generate_iterations(beam_search_generator, mock_evaluator):
    prompt = "Once upon a time"
    genre = "fantasy"

    # Mock the _generate_batch method
    beam_search_generator._generate_batch = MagicMock(return_value=["story1", "story2", "story3", "story4", "story5"])

    # Mock the evaluate_multiple method
    mock_evaluator.evaluate_multiple = MagicMock(return_value=[
        ("story2", CombinedScore(0.9, 0.9, 0.9, 0.9)),
        ("story1", CombinedScore(0.8, 0.8, 0.8, 0.8)),
        ("story4", CombinedScore(0.7, 0.7, 0.7, 0.7)),
        ("story3", CombinedScore(0.6, 0.6, 0.6, 0.6)),
        ("story5", CombinedScore(0.5, 0.5, 0.5, 0.5)),
    ])

    stories = beam_search_generator.generate_iterations(prompt, genre, mock_evaluator)

    assert len(stories) == beam_search_generator.config.num_beams
    assert stories == ["story2", "story1", "story4", "story3"]
    beam_search_generator._generate_batch.assert_called()
    mock_evaluator.evaluate_multiple.assert_called()
