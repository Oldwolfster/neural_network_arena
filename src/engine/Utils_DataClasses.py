from dataclasses import dataclass


@dataclass
class IterationData:
    model_id: str
    epoch: int
    step: int
    inputs: str  # Serialized as JSON
    target: float
    prediction: float
    loss: float