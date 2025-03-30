from dataclasses import dataclass
from typing import Union


@dataclass
class LabelModel:
    """
    Data class to store text classification data.

    Attributes:
        text: The input text to be classified
        label_name: Name of the label
        predicted_label: Label predicted by the model

    """
    text: str
    label_name: str
    predicted_label: Union[str, None]