import os
import logging
import datetime
from pathlib import Path
from typing import Optional

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from models.label_model import LabelModel
from services.datahandler import DataHandler
from services.prompt_runner import PromptRunner
from orchestrator import Orchestrator


def setup_logging(log_filepath: str) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_filepath: Path to the log file.

    Returns:
        Configured logger instance.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_filepath)
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class Container(containers.DeclarativeContainer):
    """
    Dependency injection container.
    """
    config = providers.Configuration()

    # Logging
    logger = providers.Resource(
        setup_logging,
        log_filepath=config.log_filepath
    )

    # Services
    data_handler = providers.Factory(
        DataHandler,
        csv_path=config.csv_path,
        sample_size=300
    )

    prompt_runner = providers.Factory(
        PromptRunner,
        ollama_url=config.ollama_url,
        ollama_model=config.ollama_model,
        openai_model=config.openai_model,
        openai_api_key=config.openai_api_key
    )

    orchestrator = providers.Factory(
        Orchestrator,
        data_handler=data_handler,
        prompt_runner=prompt_runner,
        csv_path=config.csv_path
    )


@inject
def main(orchestrator: Orchestrator = Provide[Container.orchestrator]) -> None:
    """
    Main entry point for the application.

    Args:
        orchestrator: Orchestrator instance provided by dependency injection.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting text classification evaluation")

    try:
        # Run the orchestrator
        results = orchestrator.run()

        # Log summary of results
        logger.info("Evaluation summary:")
        for model in ["ollama", "openai"]:
            for prompt_type in ["zero_shot", "one_shot", "few_shot"]:
                accuracy = results[model][prompt_type]["accuracy"]
                logger.info(f"Model: {model}, Prompt Type: {prompt_type}, Accuracy: {accuracy:.4f}")

        logger.info("Text classification evaluation completed successfully")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    # Generate log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    log_filepath = f"logs/{timestamp}_log.txt"

    # Create and configure the container
    container = Container()

    # Load configuration from environment variables
    container.config.ollama_url.from_env("OLLAMA_URL", required=True)
    container.config.ollama_model.from_env("OLLAMA_MODEL", required=True)
    container.config.openai_model.from_env("OPENAI_MODEL", required=True)
    container.config.openai_api_key.from_env("OPENAI_API_KEY", required=True)
    container.config.csv_path.from_env("CSV_PATH", required=True)
    container.config.log_filepath.from_value(log_filepath)

    # Wire container
    container.wire(modules=[__name__])

    # Run the application
    main()

    # For debugging/testing with mocks
    # from unittest import mock
    # with container.prompt_runner.override(mock.Mock()):
    #     main()