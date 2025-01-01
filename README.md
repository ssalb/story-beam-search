---
title: Story Generator
emoji: 📚
colorFrom: blue
python_version: 3.10.13
colorTo: pink
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
preload_from_hub:
# - meta-llama/Llama-3.2-1B-Instruct # Do not preload llama, as the token is not available at build time
- google-bert/bert-base-uncased
- facebook/bart-large-mnli
license: mit
---

## Project Overview

The Story Generator project leverages advanced natural language processing models to generate coherent and engaging stories. By utilizing models such as GPT-2, BERT, and BART, this project aims to provide users with a tool to create narratives based on given prompts. The application is built using Gradio for an interactive user interface, making it easy to input prompts and receive generated stories in real-time.

The main purpose of this project is to explore the idea of beam search for selecting stories with high coherence, fluency, and genre alignment scores. This ensures that the generated stories are not only creative but also maintain a logical flow and adhere to the specified genre.

Note that the final implementation is not strictly beam search and was modified to allow more diversity (creativity) inspired by the DVTS method in [this blog post](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute).
