from typing import List, Tuple
import pandas as pd
import seaborn as sns
from src.engine.BaseArena import BaseArena


class TitanicSurvivalArena(BaseArena):
    """
    Titanic Survivors Dataset (Binary Classification)

    Predicts whether a passenger survived (0 = No, 1 = Yes) based on features:
      - Pclass: Ticket class (1st, 2nd, 3rd)
      - Sex: Encoded as binary (Sex_Female, Sex_Male)
      - Age: Imputed with median if missing
      - SibSp: Siblings/spouses aboard
      - Parch: Parents/children aboard
      - Fare: Ticket price
      - Embarked: One-hot encoded (C, Q, S)

    Target is: Survived (1 = Yes, 0 = No)
    """
    def __init__(self):
        self.df = sns.load_dataset("titanic").copy()
        self.df = self.df.drop(columns=["embark_town", "alive", "who", "deck", "class", "adult_male", "alone", "name", "ticket", "cabin"])

    def generate_training_data(self) -> Tuple[List[Tuple[float, ...]], List[str]]:
        df = self.df

        # Drop rows with missing target
        df = df[df["survived"].notna()]

        # Impute missing age with median
        df["age"] = df["age"].fillna(df["age"].median())

        # Drop any remaining rows with missing values
        df = df.dropna()

        # One-hot encode sex and embarked
        df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=False)

        # Select relevant columns
        columns = [
            "pclass", "age", "sibsp", "parch", "fare",
            "sex_female", "sex_male",
            "embarked_C", "embarked_Q", "embarked_S"
        ]
        df = df[columns + ["survived"]]

        samples = [tuple(row) for row in df.to_numpy()]
        labels = [
            "Pclass", "Age", "SibSp", "Parch", "Fare",
            "Sex_Female", "Sex_Male",
            "Embarked_C", "Embarked_Q", "Embarked_S",
            "Survived"
        ]

        return samples, labels
