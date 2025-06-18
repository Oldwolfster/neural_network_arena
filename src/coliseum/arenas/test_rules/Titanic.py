import seaborn as sns
import pandas as pd

from src.engine.BaseArena import BaseArena


def load_titanic_dataframe():
    df = sns.load_dataset('titanic')

    # Select features and target
    df = df[["pclass", "sex", "age", "fare", "survived"]]

    # Drop rows with missing values
    df = df.dropna()

    # Encode 'sex' as 0/1
    df["sex"] = df["sex"].map({"male": 0, "female": 1})

    return df


class   Arena_TitanicSurvivors_Real(BaseArena):
    def __init__(self, max_rows=None):
        self.max_rows = max_rows

    def generate_training_data(self):
        df = load_titanic_dataframe()

        if self.max_rows:
            df = df.head(self.max_rows)

        samples = df.to_numpy().tolist()
        training_data = [tuple(row) for row in samples]

        labels = ["Pclass", "Sex", "Age", "Fare", "Outcome"]
        return training_data, labels, ["Died", "Survived"]
