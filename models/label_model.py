from dataclasses import dataclass

@dataclass
class LabelModel:
    """
    Data class to store text classification data.

    Attributes:
        text: The input text to be classified
        label: Integer label for the text
        label_name: String representation of the label
    """
    text: str
    label: int
    label_name: str