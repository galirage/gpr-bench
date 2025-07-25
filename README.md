<div align="center">
  <img src="./assets/galirage_logo.png" width="100%" alt="galirage_logo" />
</div>
<hr>

[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-gpr--bench-blue)](https://huggingface.co/datasets/galirage/gpr-bench)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2505.02854v1-blue.svg)](https://arxiv.org/html/2505.02854v1)
# GPR-bench

GPR-bench is a package that provides a framework for regression testing of generative AI systems. This tool is designed to evaluate the performance of generative AI models and measure their correctness and conciseness.

## Features

- Comprehensive benchmarking for evaluating generative AI model responses
- Evaluation from two perspectives: correctness and conciseness
- Objective scoring using OpenAI API as an evaluator
- Excel export functionality for results
- Customizable evaluation metrics

## Requirements

- Python 3.10 or higher
- OpenAI API key

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/galirage/gpr-bench
   cd gpr-bench
   ```

2. Install dependencies using uv:

   ```bash
   uv sync
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Set up environment variables:

   Copy `.env.sample` to `.env` and set your OpenAI API key:

   ```bash
   cp .env.sample .env
   # Edit .env file to set your OpenAI API key
   ```

## Usage

1. Basic execution:

   ```bash
   uv run python 01_generate_answer_and_evaluate.py
   ```

   ```bash
   uv run python 02_compare_by_model.py
   ```

   ```bash
   uv run python 03_compare_by_prompt_type.py
   ```

   ```bash
   uv run python 04_statistical_test.py
   ```

2. Customization:

   Edit the `generate_answer` function in `01_generate_answer_and_evaluate.py` to configure the generative AI model you want to evaluate.

## Output

The program generates two Excel files:

1. `gpr-bench-loaddata-{timestamp}.xlsx`: Input dataset
2. `gpr-bench-result-{timestamp}.xlsx`: Evaluation results
   - Prompts
   - Expected answers
   - Generated answers
   - Skill information
   - Language information
   - Correctness scores and comments
   - Conciseness scores and comments

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Author

[Galirage Inc.](https://galirage.com)

## Acknowledgments

- Uses evaluation prompts from the [OpenEvals](https://github.com/langchain-ai/openevals) project
- Uses the [GPR-bench](https://huggingface.co/datasets/galirage/gpr-bench) dataset
