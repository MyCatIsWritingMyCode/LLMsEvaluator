import logging
import random
from typing import List, Dict, Any, Tuple
import pandas as pd
from collections import defaultdict

from label_model import LabelModel
from datahandler import DataHandler
from prompt_runner import PromptRunner

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
    
    def run(self) -> Dict[str, Any]:
        """
        Run the evaluation process.
        
        Returns:
            Dictionary containing evaluation results.
            
        Raises:
            Exception: If any error occurs during evaluation.
        """
        try:
            self.logger.info("Starting evaluation process")
            
            # Read data
            data = self.data_handler.read_csv()
            
            # Get unique label names
            label_names = list(set(item.label_name for item in data))
            self.logger.info(f"Found {len(label_names)} unique labels: {label_names}")
            
            # Select examples for few-shot prompting
            few_shot_examples = self._select_few_shot_examples(data, label_names)
            
            # Run evaluations
            results = {
                "ollama": {
                    "zero_shot": self._evaluate_zero_shot(data, label_names, "ollama"),
                    "one_shot": self._evaluate_one_shot(data, label_names, "ollama", few_shot_examples),
                    "few_shot": self._evaluate_few_shot(data, label_names, "ollama", few_shot_examples)
                },
                "openai": {
                    "zero_shot": self._evaluate_zero_shot(data, label_names, "openai"),
                    "one_shot": self._evaluate_one_shot(data, label_names, "openai", few_shot_examples),
                    "few_shot": self._evaluate_few_shot(data, label_names, "openai", few_shot_examples)
                }
            }
            
            # Calculate and log summary metrics
            self._calculate_metrics(results)
            
            self.logger.info("Evaluation process completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in evaluation process: {str(e)}")
            raise
    
    def _select_few_shot_examples(self, data: List[LabelModel], label_names: List[str]) -> Dict[str, Tuple[str, str]]:
        """
        Select examples for one-shot and few-shot prompting.
        
        Args:
            data: List of LabelModel objects.
            label_names: List of unique label names.
            
        Returns:
            Dictionary mapping label names to (text, label_name) examples.
        """
        examples = {}
        
        # Group data by label
        label_groups = defaultdict(list)
        for item in data:
            label_groups[item.label_name].append(item)
        
        # Select one example per label
        for label in label_names:
            if label in label_groups and label_groups[label]:
                example = random.choice(label_groups[label])
                examples[label] = (example.text, example.label_name)
        
        return examples
    
    def _evaluate_zero_shot(self, 
                           data: List[LabelModel], 
                           label_names: List[str], 
                           model_type: str) -> Dict[str, Any]:
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
                predicted_label = self._parse_label_from_response(response, label_names)
                
                results.append({
                    "text": item.text,
                    "true_label": item.label_name,
                    "predicted_label": predicted_label,
                    "correct": predicted_label == item.label_name
                })
            except Exception as e:
                self.logger.error(f"Error in zero-shot evaluation for {model_type} with text '{item.text[:30]}...': {str(e)}")
                results.append({
                    "text": item.text,
                    "true_label": item.label_name,
                    "predicted_label": "ERROR",
                    "correct": False
                })
        
        # Calculate accuracy
        correct_count = sum(1 for r in results if r["correct"])
        accuracy = correct_count / len(results) if results else 0
        
        self.logger.info(f"{model_type} zero-shot accuracy: {accuracy:.4f}")
        
        return {
            "accuracy": accuracy,
            "results": results
        }
    
    def _evaluate_one_shot(self, 
                          data: List[LabelModel], 
                          label_names: List[str], 
                          model_type: str,
                          few_shot_examples: Dict[str, Tuple[str, str]]) -> Dict[str, Any]:
        """
        Evaluate one-shot prompting performance.
        
        Args:
            data: List of LabelModel objects.
            label_names: List of unique label names.
            model_type: Either "ollama" or "openai".
            few_shot_examples: Dictionary mapping label names to (text, label_name) examples.
            
        Returns:
            Dictionary containing evaluation results.
        """
        self.logger.info(f"Evaluating {model_type} with one-shot prompting")
        
        # Get a random example for one-shot prompting
        example_label = random.choice(list(few_shot_examples.keys()))
        example_text, _ = few_shot_examples[example_label]
        
        results = []
        
        for item in data:
            try:
                response = self.prompt_runner.run_one_shot(
                    text=item.text,
                    example_text=example_text,
                    example_label=example_label,
                    label_names=label_names,
                    model_type=model_type
                )
                
                # Parse response to get predicted label
                predicted_label = self._parse_label_from_response(response, label_names)
                
                results.append({
                    "text": item.text,
                    "true_label": item.label_name,
                    "predicted_label": predicted_label,
                    "correct": predicted_label == item.label_name
                })
            except Exception as e:
                self.logger.error(f"Error in one-shot evaluation for {model_type} with text '{item.text[:30]}...': {str(e)}")
                results.append({
                    "text": item.text,
                    "true_label": item.label_name,
                    "predicted_label": "ERROR",
                    "correct": False
                })
        
        # Calculate accuracy
        correct_count = sum(1 for r in results if r["correct"])
        accuracy = correct_count / len(results) if results else 0
        
        self.logger.info(f"{model_type} one-shot accuracy: {accuracy:.4f}")
        
        return {
            "accuracy": accuracy,
            "results": results
        }
    
    def _evaluate_few_shot(self, 
                          data: List[LabelModel], 
                          label_names: List[str], 
                          model_type: str,
                          few_shot_examples: Dict[str, Tuple[str, str]]) -> Dict[str, Any]:
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
        
        # Convert few-shot examples to the format expected by run_few_shot
        examples = [(text, label) for label, (text, _) in few_shot_examples.items()]
        
        results = []
        
        for item in data:
            try:
                response = self.prompt_runner.run_few_shot(
                    text=item.text,
                    examples=examples,
                    label_names=label_names,
                    model_type=model_type
                )
                
                # Parse response to get predicted label
                predicted_label = self._parse_label_from_response(response, label_names)
                
                results.append({
                    "text": item.text,
                    "true_label": item.label_name,
                    "predicted_label": predicted_label,
                    "correct": predicted_label == item.label_name
                })
            except Exception as e:
                self.logger.error(f"Error in few-shot evaluation for {model_type} with text '{item.text[:30]}...': {str(e)}")
                results.append({
                    "text": item.text,
                    "true_label": item.label_name,
                    "predicted_label": "ERROR",
                    "correct": False
                })
        
        # Calculate accuracy
        correct_count = sum(1 for r in results if r["correct"])
        accuracy = correct_count / len(results) if results else 0
        
        self.logger.info(f"{model_type} few-shot accuracy: {accuracy:.4f}")
        
        return {
            "accuracy": accuracy,
            "results": results
        }
    
    def _parse_label_from_response(self, response: str, label_names: List[str]) -> str:
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
        
        self.logger.warning(f"Could not parse label from response: {response[:100]}...")
        return "UNKNOWN"
    
    def _calculate_metrics(self, results: Dict[str, Any]) -> None:
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
