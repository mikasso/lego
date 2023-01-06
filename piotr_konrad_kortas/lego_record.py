from dataclasses import dataclass


@dataclass
class LegoRecord:
    path: str
    label: float
    features: list[int]

    def __str__(self) -> str:
        features_string = [str(feature) for feature in self.features]
        features_str = "|".join(features_string)
        return f"{self.path},{self.label},{features_str}"
