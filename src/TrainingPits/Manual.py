from src.TrainingPit import TrainingPit
import random
from typing import List, Tuple
class Manual(TrainingPit):
    """
    This class allows you to send literal training data
    for example if you need repeatable results
    """
    def __init__(self,num_samples: int):
        self.num_samples = num_samples
    def generate_training_data(self) -> List[Tuple[float, float]]:
        #return [(48.87730117867727, 0.5242352072001506), (93.32423633044634, 0.9510600352027276), (98.66735264119876, 0.9298603136331564)]
        return [(54.2006068654841, 0.54742558974329), (38.1047634309179, 0.384478364244135),
         (55.4749634278304, 0.590577020714811), (54.7616870566915, 0.527049456764278),
         (59.1060772448001, 0.638683330155766), (78.093170606691, 0.694090136176182),
         (53.2086112530682, 0.652711027057854), (32.8197905474707, 0.207073841047952),
         (3.47509829601228, 0.151372444325172), (24.4746841209769, 0.0728439293642262), ]
