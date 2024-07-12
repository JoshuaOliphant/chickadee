# Chickadee - ChatGPT Conversation Analysis

This project analyzes ChatGPT conversations to extract and refine prompts, providing insights into common themes and patterns in user queries. It uses OpenAI's GPT-4o model to process the conversations and generate reusable prompts.

## Features

- Extracts questions from ChatGPT conversation JSON files
- Analyzes questions to identify common themes and patterns
- Generates reusable prompts based on the analysis
- Aggregates and refines prompts to create a concise set of high-quality prompts
- Uses asynchronous processing for improved performance
- Implements robust error handling and logging

## Requirements

- Python 3.7+
- Poetry for dependency management

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatgpt-conversation-analysis.git
   cd chatgpt-conversation-analysis
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set up your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Configuration

The main script (`chickadee.py`) uses the following configuration:

- OpenAI model: gpt-4o
- Max tokens per batch: 6000
- Input file: `test.json` (ChatGPT conversation export)
- Output files:
  - `refined_prompts.txt`: Contains the refined prompts
  - `refinement_analysis.txt`: Contains the analysis of the refinement process

You can modify these settings in the `main()` function of the script.

## Usage

1. Ensure your ChatGPT conversation export (in JSON format) is in the same directory as the script, named `test.json`.

2. Run the script using Poetry:
   ```bash
   poetry run python chickadee.py
   ```

3. The script will process the conversations, analyze the questions, and generate refined prompts. Progress and results will be logged to the console.

4. After completion, check the `refined_prompts.txt` and `refinement_analysis.txt` files for the results.

## Project Structure

- `chickadee.py`: Main script containing the conversation analysis logic
- `test.json`: Input file containing ChatGPT conversations (not included in the repository)
- `refined_prompts.txt`: Output file containing the refined prompts
- `refinement_analysis.txt`: Output file containing the analysis of the refinement process
- `pyproject.toml`: Poetry configuration file for managing dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for providing the GPT-4 model
- The Instructor library for structured outputs from language models
- Logfire for efficient logging