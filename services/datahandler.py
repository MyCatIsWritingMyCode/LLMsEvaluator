import logging
import pandas as pd

from models.label_model import LabelModel

class DataHandler:
    """
    Class to handle data operations for text classification.
    """
    
    def __init__(self, csv_path: str, sample_size: int):
        """
        Initialize the DataHandler.
        
        Args:
            csv_path: Path to the CSV file containing the data.
            sample_size: Number of rows to randomly sample from the CSV.
        """
        self.csv_path = csv_path
        self.sample_size = sample_size
        self.logger = logging.getLogger(__name__)
    
    def read_csv(self) -> list[LabelModel]:
        """
        Read data from CSV file and randomly select a sample.
        
        Returns:
            A list of LabelModel objects containing text, label, and label_name.
        
        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If required columns are missing.
        """
        try:
            self.logger.info(f"Reading data from {self.csv_path}")
            df = pd.read_csv(self.csv_path, delimiter=';')
            
            # Check if required columns exist
            required_columns = ['text', 'label']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in CSV")
            
            # Sample rows
            if len(df) > self.sample_size:
                df = df.sample(n=self.sample_size, random_state=42)
            
            # Convert to list of LabelModel objects
            data = []
            for _, row in df.iterrows():
                data.append(LabelModel(
                    text=row['text'],
                    label_name=row['label'],
                    predicted_label=None
                ))
            
            self.logger.info(f"Successfully read {len(data)} rows from CSV")
            return data
            
        except FileNotFoundError as e:
            self.logger.error(f"CSV file not found: {self.csv_path}")
            raise
        except ValueError as e:
            self.logger.error(f"Invalid CSV format: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading CSV file: {str(e)}")
            raise
