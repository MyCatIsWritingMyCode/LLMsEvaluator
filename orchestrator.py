import logging, random
from collections import defaultdict

from models.label_model import LabelModel
from services.datahandler import DataHandler
from services.prompt_runner import PromptRunner

class Orchestrator:
    """
    Class to orchestrate the process of evaluating language models for text classification.
    """

    def __init__(self, 
                 data_handler: DataHandler, 
                 prompt_runner: PromptRunner,
                 csv_path: str):
        """
        Initialize the Orchestrator.
        
        Args:
            data_handler: Instance of DataHandler for reading data.
            prompt_runner: Instance of PromptRunner for running prompts.
            csv_path: Path to the CSV file containing data.
        """
        self.data_handler = data_handler
        self.prompt_runner = prompt_runner
        self.csv_path = csv_path
        self.logger = logging.getLogger(__name__)

        self._ERROR = "ERROR"
    
    def run(self):
        """
        Run the evaluation process.
        
        Returns:
            Dictionary containing evaluation results.
        """
        try:
            self.logger.info("Starting evaluation process")
            
            # Read data
            data = self.data_handler.read_csv()

            # Randomly select x entries
            random_testing_samples = random.sample(data, 100)

            # Remove the x samples that are including in the testing set
            for item in random_testing_samples:
                data.remove(item)

            # Get unique label names
            label_names = list(set(item.label_name for item in data))
            self.logger.info(f"Found {len(label_names)} unique labels: {label_names}")
            
            # Select examples for few-shot prompting
            few_shot_examples = self._select_few_shot_examples(data, label_names)

            # Run evaluations
            results: dict[str, dict[str, (str, list[LabelModel])]] = {
                "ollama": {
                    "zero_shot": self._evaluate_zero_shot(random_testing_samples, label_names, "ollama"),
                    "one_shot": self._evaluate_one_shot(random_testing_samples, label_names, "ollama", random.choice(few_shot_examples)),
                    "few_shot": self._evaluate_few_shot(random_testing_samples, label_names, "ollama", few_shot_examples)
                },
                "openai": {
                    "zero_shot": self._evaluate_zero_shot(random_testing_samples, label_names, "openai"),
                    "one_shot": self._evaluate_one_shot(random_testing_samples, label_names, "openai", random.choice(few_shot_examples)),
                    "few_shot": self._evaluate_few_shot(random_testing_samples, label_names, "openai", few_shot_examples)
                }
            }
            
            # Calculate and log summary metrics
            self._calculate_metrics(results)
            
            self.logger.info("Evaluation process completed successfully")

            # Log summary of results
            self.logger.info("Evaluation summary:")
            for model in ["ollama", "openai"]:
                for prompt_type in ["zero_shot", "one_shot", "few_shot"]:
                    accuracy = results[model][prompt_type]["accuracy"]
                    self.logger.info(f"Model: {model}, Prompt Type: {prompt_type}, Accuracy: {accuracy:.4f}")

            self.logger.info("Text classification evaluation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in evaluation process: {str(e)}")
            raise

    @staticmethod
    def _select_few_shot_examples(data: list[LabelModel], label_names: list[str]) -> list[LabelModel]:
        """
        Select examples for one-shot and few-shot prompting.
        
        Args:
            data: List of LabelModel objects.
            label_names: List of unique label names.
            
        Returns:
            Dictionary mapping label names to (text, label_name) examples.
        """
        examples:list[LabelModel] = []

        # Group data by label
        label_groups = defaultdict(list)
        for item in data:
            label_groups[item.label_name].append(item)
        
        # Select one example per label
        for label in label_names:
            if label in label_groups and label_groups[label]:
                example = random.choice(label_groups[label])
                examples.append(example)
        
        return examples
    
    def _evaluate_zero_shot(self, 
                           data: list[LabelModel],
                           label_names: list[str],
                           model_type: str) -> (str, list[LabelModel]):
        """
        Evaluate zero-shot prompting performance.
        
        Args:
            data: List of LabelModel objects.
            label_names: List of unique label names.
            model_type: Either "ollama" or "openai".
            
        Returns:
            Dictionary containing evaluation results.
        """
        self.logger.info(f"Evaluating {model_type} with zero-shot prompting")
        
        results = []
        
        for item in data:
            try:
                response = self.prompt_runner.run_zero_shot(
                    text=item.text,
                    label_names=label_names,
                    model_type=model_type
                )
                
                # Parse response to get predicted label
                item.predicted_label = self._parse_label_from_response(response, label_names)
                
                results.append(item)

            except Exception as e:
                self.logger.error(f"Error in zero-shot evaluation for {model_type} with text '{item.text[:30]}...': {str(e)}")
                item.predicted_label = self._ERROR

                results.append(item)
        
        # Calculate accuracy
        correct_count = sum(1 for r in results if r.predicted_label == r.predicted_label)
        accuracy = correct_count / len(results) if results else 0
        
        self.logger.info(f"{model_type} zero-shot accuracy: {accuracy:.4f}")
        
        return {
            "accuracy": accuracy,
            "results": results
        }
    
    def _evaluate_one_shot(self, 
                          data: list[LabelModel],
                          label_names: list[str],
                          model_type: str,
                          one_shot_examples: LabelModel) -> (str, list[LabelModel]):
        """
        Evaluate one-shot prompting performance.
        
        Args:
            data: List of LabelModel objects.
            label_names: List of unique label names.
            model_type: Either "ollama" or "openai".
            one_shot_examples: example (text, label_name).
            
        Returns:
            Dictionary containing evaluation results.
        """
        self.logger.info(f"Evaluating {model_type} with one-shot prompting")
        
        results:list[LabelModel] = []
        
        for item in data:
            try:
                response = self.prompt_runner.run_one_shot(
                    text=item.text,
                    example=one_shot_examples,
                    label_names=label_names,
                    model_type=model_type
                )
                
                # Parse response to get predicted label
                item.predicted_label = self._parse_label_from_response(response, label_names)

                results.append(item)

            except Exception as e:
                self.logger.error(f"Error in one-shot evaluation for {model_type} with text '{item.text[:30]}...': {str(e)}")
                item.predicted_label = self._ERROR
                results.append(item)
        
        # Calculate accuracy
        correct_count = sum(1 for r in results if r.label_name == r.predicted_label)
        accuracy = correct_count / len(results) if results else 0
        
        self.logger.info(f"{model_type} one-shot accuracy: {accuracy:.4f}")
        
        return {
            "accuracy": accuracy,
            "results": results
        }
    
    def _evaluate_few_shot(self, 
                          data: list[LabelModel],
                          label_names: list[str],
                          model_type: str,
                          few_shot_examples: list[LabelModel]) -> (str, list[LabelModel]):
        """
        Evaluate few-shot prompting performance.
        
        Args:
            data: List of LabelModel objects.
            label_names: List of unique label names.
            model_type: Either "ollama" or "openai".
            few_shot_examples: Dictionary mapping label names to (text, label_name) examples.
            
        Returns:
            Dictionary containing evaluation results.
        """
        self.logger.info(f"Evaluating {model_type} with few-shot prompting")
        
        results = []
        
        for item in data:
            try:
                response = self.prompt_runner.run_few_shot(
                    text=item.text,
                    examples=few_shot_examples,
                    label_names=label_names,
                    model_type=model_type
                )
                
                # Parse response to get predicted label
                item.predicted_label = self._parse_label_from_response(response, label_names)

                results.append(item)

            except Exception as e:
                self.logger.error(f"Error in few-shot evaluation for {model_type} with text '{item.text[:30]}...': {str(e)}")
                item.predicted_label = self._ERROR
                results.append(item)
        
        # Calculate accuracy
        correct_count = sum(1 for r in results if r.label_name == r.predicted_label)
        accuracy = correct_count / len(results) if results else 0
        
        self.logger.info(f"{model_type} few-shot accuracy: {accuracy:.4f}")
        
        return {
            "accuracy": accuracy,
            "results": results
        }
    
    def _parse_label_from_response(self, response: str, label_names: list[str]) -> str:
        """
        Parse model response to extract the predicted label.
        
        Args:
            response: Model's response.
            label_names: List of possible label names.
            
        Returns:
            Predicted label or "UNKNOWN" if parsing fails.
        """
        # First, check for exact matches
        for label in label_names:
            if label in response:
                return label
        
        # If no exact match, try case-insensitive matching
        response_lower = response.lower()
        for label in label_names:
            if label.lower() in response_lower:
                return label
        
        self.logger.warning(f"Could not parse label from response: {response[:50]}...")
        return "UNKNOWN"
    
    def _calculate_metrics(self, results: dict[str, dict[str, (str, list[LabelModel])]]) -> None:
        """
        Calculate and log summary metrics from the evaluation results.
        
        Args:
            results: Dictionary containing evaluation results.
        """
        self.logger.info("Summary Metrics:")
        
        for model in ["ollama", "openai"]:
            for prompt_type in ["zero_shot", "one_shot", "few_shot"]:
                accuracy = results[model][prompt_type]["accuracy"]
                self.logger.info(f"{model.upper()} {prompt_type}: Accuracy = {accuracy:.4f}")
                
        # Compare model performance
        ollama_avg = (results["ollama"]["zero_shot"]["accuracy"] + 
                     results["ollama"]["one_shot"]["accuracy"] + 
                     results["ollama"]["few_shot"]["accuracy"]) / 3
                     
        openai_avg = (results["openai"]["zero_shot"]["accuracy"] + 
                     results["openai"]["one_shot"]["accuracy"] + 
                     results["openai"]["few_shot"]["accuracy"]) / 3
                     
        self.logger.info(f"Average accuracy - OLLAMA: {ollama_avg:.4f}, OPENAI: {openai_avg:.4f}")
