# Text Classification Model Evaluator

This project evaluates the performance of text classification using two language models:
1. A locally running Ollama LLM 
2. An OpenAI LLM via API calls

## Project Structure

```
├── main.py                         # Entry point for the application
├── orchestrator.py                 # Coordinates the evaluation process
├── services/prompt_runner.py       # Handles running prompts on both models
├── services/datahandler.py         # Reads and processes data from CSV
├── models/label_model.py           # Data class for storing text classification data
├── .env                            # Configuration file for environment variables
├── log.ini                         # Configuration file for the logger
├── requirements.txt                # Project dependencies
└── logs/                           # Directory for log files
```

## Requirements

- Python 3.8+
- Local installation of an Ollama model
- OpenAI API key
- CSV file with text and labels with the columns: text; labels

## Installation

* Clone the repository
* Install dependencies:
   ```
   pip install -r requirements.txt
   ```
* Configure the `.env` file in the root folder with the following variables:

        OLLAMA_MODEL=[name of the model e.g. deepseek-r1:7b]
        OPENAI_MODEL=[name of the OpenAI model e.g. gpt-4o]
        OPENAI_API_KEY=[your_openai_api_key_here]
        CSV_PATH=[Path to your CSV file]

## Usage

1. Prepare your CSV file with columns: `text`;`label` and `;` as the seperator
2. Update the `.env` file with your configuration
3. Run the application:
   ```
   python main.py
   ```

## Features

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