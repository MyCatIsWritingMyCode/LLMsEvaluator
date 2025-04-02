import logging, logging.config, datetime
from dotenv import load_dotenv

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from services.datahandler import DataHandler
from services.prompt_runner import PromptRunner
from orchestrator import Orchestrator

class Container(containers.DeclarativeContainer):
    """
    Dependency injection container.
    """

    config = providers.Configuration()

    # Logging
    logger = providers.Resource(
        logging.config.fileConfig,
        fname="log.ini",
        defaults = {'logpath': datetime.datetime.now().strftime('%Y_%m_%d.log')}
    )

    # Services
    data_handler = providers.Factory(
        DataHandler,
        csv_path=config.csv_path,
        sample_size=60
    )

    prompt_runner = providers.Factory(
        PromptRunner,
        ollama_model=config.ollama_model,
        openai_model=config.openai_model,
        openai_api_key=config.openai_api_key
    )

    orchestrator = providers.Factory(
        Orchestrator,
        data_handler=data_handler,
        prompt_runner=prompt_runner,
        ollama_active=config.ollama_active,
        openai_active=config.openai_active,
        csv_path=config.csv_path
    )


@inject
def main(orchestrator: Orchestrator = Provide[Container.orchestrator]) -> None:
    """
    Main entry point for the application.

    Args:
        orchestrator: instance provided by dependency injection
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting text classification evaluation")

    try:
        # Run the orchestrator
        orchestrator.run()

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    # create and configure the container
    container = Container()

    load_dotenv()
    # Load configuration from environment variables
    container.config.ollama_model.from_env("OLLAMA_MODEL", required=True)
    container.config.openai_model.from_env("OPENAI_MODEL", required=True)
    container.config.openai_api_key.from_env("OPENAI_API_KEY", required=True)
    container.config.csv_path.from_env("CSV_PATH", required=True)
    container.config.ollama_active.from_env("OLLAMA_ACTIVE", required=True)
    container.config.openai_active.from_env("OPENAI_ACTIVE", required=True)

    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    # log_filepath = f"logs/{timestamp}_log.txt"
    # container.config.log_filepath.from_value(log_filepath)

    container.init_resources()

    # Wire container
    container.wire(modules=[__name__])

    # Run the application
    main()