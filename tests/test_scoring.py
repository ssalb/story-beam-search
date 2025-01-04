# tests/test_scoring.py
import pytest
import torch
from unittest.mock import Mock, MagicMock
from transformers import PreTrainedModel, PreTrainedTokenizer, Pipeline
from story_beam_search.scoring import (
    CoherenceScorer,
    FluencyScorer,
    GenreAlignmentScorer,
    StoryEvaluator,
    CombinedScore,
)

# Fixtures


@pytest.fixture
def mock_model():

    def mock_output(**inputs):
        output = MagicMock()
        output.logits = torch.ones(1, 10, 20)
        output.hidden_states = [
            torch.tensor(
                [
                    [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
                    [[0, 1.2, 2], [0, 1.2, 2], [0, 1.2, 2]],
                    [[0, 1, 2.2], [0, 1, 2.2], [0, 1, 2.2]],
                    [[0, 1.2, 2.2], [0, 1.2, 2.2], [0, 1.2, 2.2]],
                    [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
                    [[0, 1.2, 2], [0, 1.2, 2], [0, 1.2, 2]],
                ]
            )
        ]
        return output

    model = Mock(spec=PreTrainedModel, side_effect=mock_output)

    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock(spec=PreTrainedTokenizer)

    tokenizer.mask_token_id = 0
    tokenizer.pad_token = None
    tokenizer.mask_token = None
    tokenizer.eos_token = None
    tokenizer.vocab_size = 10

    tokenizer_output = MagicMock()
    tokenizer_output.input_ids = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    tokenizer_output.attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    tokenizer.__call__ = MagicMock(return_value=tokenizer_output)

    tokenizer.return_value.to = MagicMock(return_value=tokenizer_output)

    return tokenizer


@pytest.fixture
def mock_zero_shot_pipeline():
    pipeline = Mock(spec=Pipeline)
    pipeline.return_value = [{"labels": ["test_genre"], "scores": [0.8]}]
    return pipeline


# CoherenceScorer Tests
class TestCoherenceScorer:
    def test_initialization(self, mock_model, mock_tokenizer):
        scorer = CoherenceScorer(
            model=mock_model, tokenizer=mock_tokenizer, device=torch.device("cpu")
        )
        assert scorer.model == mock_model
        assert scorer.tokenizer == mock_tokenizer
        assert isinstance(scorer.device, torch.device)

    @pytest.mark.parametrize(
        "story, expected_score",
        [
            (
                ["First sentence. Second sentence."],
                0.99705446,
            ),  # result for mock hidden states
            (["Single sentence."], 0.0),
            ([""], 0.0),
        ],
    )
    def test_score_calculation(self, mock_model, mock_tokenizer, story, expected_score):
        scorer = CoherenceScorer(
            model=mock_model, tokenizer=mock_tokenizer, device=torch.device("cpu")
        )
        score = scorer.score(story)[0]
        assert abs(score - expected_score) < 1e-6


# FluencyScorer Tests
class TestFluencyScorer:
    def test_initialization(self, mock_model, mock_tokenizer):
        scorer = FluencyScorer(
            model=mock_model, tokenizer=mock_tokenizer, device=torch.device("cpu")
        )
        assert scorer.model == mock_model
        assert scorer.tokenizer == mock_tokenizer

    def test_score_calculation(self, mock_model, mock_tokenizer):
        scorer = FluencyScorer(
            model=mock_model, tokenizer=mock_tokenizer, device=torch.device("cpu")
        )
        score = scorer.score(["Test story"])[0]
        expected_score = 0.05  # result for a torch.ones tensor
        assert abs(score - expected_score) < 1e-6


# GenreAlignmentScorer Tests
class TestGenreAlignmentScorer:
    def test_initialization(self, mock_zero_shot_pipeline):
        scorer = GenreAlignmentScorer(
            pipeline=mock_zero_shot_pipeline, genre="test_genre"
        )
        assert scorer.pipeline == mock_zero_shot_pipeline
        assert scorer.genre == "test_genre"

    @pytest.mark.parametrize(
        "genre,expected_score",
        [
            ("test_genre", 0.8),
            ("", 0.5),
        ],
    )
    def test_score_calculation(self, mock_zero_shot_pipeline, genre, expected_score):
        scorer = GenreAlignmentScorer(pipeline=mock_zero_shot_pipeline, genre=genre)
        score = scorer.score(["Test story"])[0]
        assert abs(score - expected_score) < 1e-6


# StoryEvaluator Tests
class TestStoryEvaluator:
    @pytest.fixture
    def mock_scorers(self, mock_model, mock_tokenizer, mock_zero_shot_pipeline):
        coherence_scorer = CoherenceScorer(
            mock_model, mock_tokenizer, torch.device("cpu")
        )
        fluency_scorer = FluencyScorer(mock_model, mock_tokenizer, torch.device("cpu"))
        genre_scorer = GenreAlignmentScorer(mock_zero_shot_pipeline, "test_genre")
        return coherence_scorer, fluency_scorer, genre_scorer

    def test_initialization(self, mock_scorers):
        coherence_scorer, fluency_scorer, genre_scorer = mock_scorers
        evaluator = StoryEvaluator(
            coherence_scorer=coherence_scorer,
            fluency_scorer=fluency_scorer,
            genre_scorer=genre_scorer,
        )
        assert evaluator.coherence_scorer == coherence_scorer
        assert evaluator.fluency_scorer == fluency_scorer
        assert evaluator.genre_scorer == genre_scorer

    def test_evaluate_single_story(self, mock_scorers):
        coherence_scorer, fluency_scorer, genre_scorer = mock_scorers
        evaluator = StoryEvaluator(
            coherence_scorer=coherence_scorer,
            fluency_scorer=fluency_scorer,
            genre_scorer=genre_scorer,
        )

        result = evaluator.evaluate_multiple(["Test story"])[0]
        assert result[0] == "Test story"
        assert isinstance(result[1], CombinedScore)
        assert 0 <= result[1].total <= 1
        assert 0 <= result[1].coherence <= 1
        assert 0 <= result[1].fluency <= 1
        assert 0 <= result[1].genre_alignment <= 1

    def test_evaluate_multiple_stories(self, mock_scorers):
        coherence_scorer, fluency_scorer, genre_scorer = mock_scorers
        evaluator = StoryEvaluator(
            coherence_scorer=coherence_scorer,
            fluency_scorer=fluency_scorer,
            genre_scorer=genre_scorer,
        )
        stories = ["Story 1. Story 1", "Story 2. Story 2", "Story 3. Story 3"]
        results = evaluator.evaluate_multiple(stories)

        assert len(results) == len(stories)
        assert all(isinstance(score, CombinedScore) for _, score in results)
        scores = [score.total for _, score in results]
        assert scores == sorted(scores, reverse=True)
