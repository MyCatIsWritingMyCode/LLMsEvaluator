import logging
import json
import requests
from typing import List, Dict, Any, Optional, Union, Tuple

import openai
from openai import OpenAI

class PromptRunner:
    """
    Class to handle running prompts on different language models.
    """
    
    def __init__(self, 
                 ollama_url: str, 
                 ollama_model: str,
                 openai_model: str, 
                 openai_api_key: str):
        """
        Initialize the PromptRunner.
        
        Args:
            ollama_url: URL for the local Ollama API.
            ollama_model: Model identifier for Ollama.
            openai_model: Model identifier for OpenAI.
            openai_api_key: API key for OpenAI.
        """
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.openai_model = openai_model
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.logger = logging.getLogger(__name__)
    
    def run_ollama_prompt(self, prompt: str) -> str:
        """
        Run a prompt on the local Ollama model.
        
        Args:
            prompt: The prompt to send to the model.
            
        Returns:
            The model's response as a string.
            
        Raises:
            requests.RequestException: If the API request fails.
            ValueError: If the response format is unexpected.
        """
        try:
            self.logger.info(f"Running prompt on Ollama model: {self.ollama_model}")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120  # 2 minute timeout
            )
            response.raise_for_status()
            result = response.json()
            if "response" not in result:
                raise ValueError(f"Unexpected response format from Ollama: {result}")
            return result.get("response", "")
        except requests.RequestException as e:
            self.logger.error(f"API error running Ollama prompt: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response from Ollama: {str(e)}")
            raise ValueError(f"Invalid JSON response from Ollama: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error running Ollama prompt: {str(e)}")
            raise
    
    def run_openai_prompt(self, prompt: str) -> str:
        """
        Run a prompt on the OpenAI model.
        
        Args:
            prompt: The prompt to send to the model.
            
        Returns:
            The model's response as a string.
            
        Raises:
            openai.OpenAIError: If the API request fails.
        """
        try:
            self.logger.info(f"Running prompt on OpenAI model: {self.openai_model}")
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in text classification."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error running OpenAI prompt: {str(e)}")
            raise
    
    def run_zero_shot(self, text: str, label_names: List[str], model_type: str) -> str:
        """
        Run a zero-shot prompting on the specified model.
        
        Args:
            text: The text to classify.
            label_names: List of possible label names.
            model_type: Either "ollama" or "openai".
            
        Returns:
            The model's classification response.
            
        Raises:
            ValueError: If the model type is unknown.
        """
        prompt = f"""
        Please classify the following text into one of these categories: {', '.join(label_names)}.
        
        Text: "{text}"
        
        Category: 
        """
        
        if model_type.lower() == "ollama":
            return self.run_ollama_prompt(prompt)
        elif model_type.lower() == "openai":
            return self.run_openai_prompt(prompt)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def run_one_shot(self, text: str, example_text: str, example_label: str, 
                    label_names: List[str], model_type: str) -> str:
        """
        Run a one-shot prompting on the specified model.
        
        Args:
            text: The text to classify.
            example_text: An example text.
            example_label: The label for the example text.
            label_names: List of possible label names.
            model_type: Either "ollama" or "openai".
            
        Returns:
            The model's classification response.
            
        Raises:
            ValueError: If the model type is unknown.
        """
        prompt = f"""
        Please classify the following text into one of these categories: {', '.join(label_names)}.
        
        Example:
        Text: "{example_text}"
        Category: {example_label}
        
        Now classify this:
        Text: "{text}"
        
        Category: 
        """
        
        if model_type.lower() == "ollama":
            return self.run_ollama_prompt(prompt)
        elif model_type.lower() == "openai":
            return self.run_openai_prompt(prompt)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def run_few_shot(self, text: str, examples: List[Tuple[str, str]], 
                    label_names: List[str], model_type: str) -> str:
        """
        Run a few-shot prompting on the specified model.
        
        Args:
            text: The text to classify.
            examples: List of (example_text, example_label) tuples.
            label_names: List of possible label names.
            model_type: Either "ollama" or "openai".
            
        Returns:
            The model's classification response.
            
        Raises:
            ValueError: If the model type is unknown.
        """
        examples_str = "\n\n".join([
            f"Text: \"{ex_text}\"\nCategory: {ex_label}" 
            for ex_text, ex_label in examples
        ])
        
        prompt = f"""
        Please classify the following text into one of these categories: {', '.join(label_names)}.
        
        Examples:
        {examples_str}
        
        Now classify this:
        Text: "{text}"
        
        Category: 
        """
        
        if model_type.lower() == "ollama":
            return self.run_ollama_prompt(prompt)
        elif model_type.lower() == "openai":
            return self.run_openai_prompt(prompt)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
