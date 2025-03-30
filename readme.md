# Text Classification Model Evaluator

This project evaluates the performance of text classification using two language models:
1. A locally running Ollama model (deepseek r1 7b)
2. OpenAI's GPT-4 via API calls

## Project Structure

```
├── main.py                # Entry point for the application
├── orchestrator.py        # Coordinates the evaluation process
├── prompt_runner.py       # Handles running prompts on both models
├── datahandler.py         # Reads and processes data from CSV
├── label_model.py         # Data class for storing text classification data
├── .env                   # Configuration file for environment variables
├── requirements.txt       # Project dependencies
└── logs/                  # Directory for log files
```

## Requirements

- Python 3.8+
- Local installation of Ollama with the deepseek-r1-7b model
- OpenAI API key
- CSV file with text classification data

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure the `.env` file with your settings

## Usage

1. Prepare your CSV file with columns: `text`, `label`, and `label_name`
2. Update the `.env` file with your configuration
3. Run the application:
   ```
   python main.py
   ```

## Features

- Dependency injection using the `dependency_injector` package
- Comprehensive exception handling
- Logging to both console and file
- Type hints throughout the codebase
- Support for three prompting methods:
  - Zero-shot prompting
  - One-shot prompting
  - Few-shot prompting

## Evaluation Process

The evaluation process:
1. Reads data from the specified CSV file
2. Randomly selects 300 rows
3. Evaluates both models with zero-shot, one-shot, and few-shot prompting
4. Calculates accuracy metrics for each model and prompting method
5. Logs results to console and file

## Extending the Project

To add support for additional models or prompting methods:
1. Extend the `PromptRunner` class with new methods
2. Update the `Orchestrator` class to use these new methods
3. Update the dependency injection container in `main.py` if necessary
