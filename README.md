# GPR-bench

GPR-bench is a package that provides a framework for regression testing of generative AI systems. This tool is designed to evaluate the performance of generative AI models and measure their correctness and conciseness.

## Features

- Comprehensive benchmarking for evaluating generative AI model responses
- Evaluation from two perspectives: correctness and conciseness
- Objective scoring using OpenAI API as an evaluator
- Excel export functionality for results
- Customizable evaluation metrics

## Requirements

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/galirage/gpr-bench
cd gpr-bench
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
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
python main.py
```

2. Customization:

Edit the `generate_answer` function in `main.py` to configure the generative AI model you want to evaluate.

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
