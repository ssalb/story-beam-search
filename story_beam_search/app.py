import gradio as gr
from story_beam_search.stories_generator import StoryGenerationSystem
from typing import Tuple, List


def create_story_generation_interface() -> gr.Interface:
    # Initialize the story generation system
    system = StoryGenerationSystem()
    system.initialize()

    def generate_stories(
        prompt: str, genre: str, num_stories: int, temperature: float, max_length: int
    ) -> Tuple[str, List[str]]:
        """
        Generate and evaluate stories based on user input.
        Returns a tuple of (detailed_scores, story_texts).
        """
        # Update beam search config with user parameters
        system.beam_search.config.temperature = temperature
        system.beam_search.config.max_length = max_length

        # Generate and evaluate stories
        ranked_stories = system.generate_and_evaluate(
            prompt, genre, num_stories=num_stories
        )

        # Format detailed scores
        detailed_scores = ""
        story_texts = []

        for i, (story, scores) in enumerate(ranked_stories, 1):
            detailed_scores += f"Story {i}:\n"
            detailed_scores += f"Total Score: {scores.total:.3f}\n"
            detailed_scores += f"Coherence: {scores.coherence:.3f}\n"
            detailed_scores += f"Fluency: {scores.fluency:.3f}\n"
            detailed_scores += f"Genre Alignment: {scores.genre_alignment:.3f}\n"
            detailed_scores += "-" * 50 + "\n"

            story_texts.append(f"Story {i}:\n{story}\n")

        return detailed_scores, "\n".join(story_texts)

    # Define interface components
    prompt_input = gr.Textbox(
        label="Story Prompt",
        placeholder="Enter the beginning of your story...",
        lines=3,
    )

    genre_input = gr.Dropdown(
        choices=[
            "children",
            "romance",
            "mystery",
            "adventure",
            "sci-fi",
            "fantasy",
            "comedy",
            "drama",
            "thriller",
            "horror",
        ],
        label="Genre",
        value="fantasy",
    )

    num_stories_input = gr.Slider(
        minimum=2, maximum=7, value=3, step=1, label="Number of Stories to Generate"
    )

    temperature_input = gr.Slider(
        minimum=0.1, maximum=3.0, value=0.8, step=0.1, label="Temperature (Creativity)"
    )

    max_length_input = gr.Slider(
        minimum=30, maximum=150, value=60, step=30, label="Maximum Length"
    )

    # Output components
    scores_output = gr.Textbox(label="Detailed Scores", lines=10, interactive=False)

    stories_output = gr.Textbox(label="Generated Stories", lines=15, interactive=False)

    # Create the interface
    interface = gr.Interface(
        fn=generate_stories,
        inputs=[
            prompt_input,
            genre_input,
            num_stories_input,
            temperature_input,
            max_length_input,
        ],
        outputs=[scores_output, stories_output],
        title="AI Story Generator",
        description="""
        Generate creative stories using AI! Enter a prompt and choose your preferences.
        The system will generate multiple stories and evaluate them based on coherence,
        fluency, and genre alignment.
        """,
        examples=[
            ["Once upon a time in a magical forest,", "fantasy", 3, 0.8, 150],
            [
                "The detective examined the crime scene carefully,",
                "mystery",
                3,
                0.7,
                200,
            ],
        ],
        theme=gr.themes.Soft(),
    )

    return interface


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_story_generation_interface()
    interface.launch()
