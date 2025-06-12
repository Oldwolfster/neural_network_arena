from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
#from src.engine.Reporting import generate_reports, create_weight_tables
from src.ArenaSettings import HyperParameters
from src.Legos.LegoSelector import LegoSelector
from src.engine.Config import Config
from src.engine.RamDB import RamDB
from src.engine.TrainingData import TrainingData
from src.engine.Utils_DataClasses import RecordLevel


class TrainingRunInfo:
    __slots__ = ("hyper", "db", "training_data", "config", "setup",
                 "run_id", "gladiator", "time_start","time_end",
                 "seed", "record_level", "last_epoch", "converge_cond",
                 "mae", "lowest_mae", "lowest_mae_epoch",
                 "bd_target_alpha","bd_target_beta","bd_target_alpha_unscaled","bd_target_beta_unscaled","bd_threshold","bd_label_alpha","bd_label_beta","bd_target_alpha","bd_correct")

    def __init__(
            self, hyper: HyperParameters, db: RamDB, training_data : TrainingData,  setup, seed, run_id, record_level: RecordLevel):
        #create_weight_tables(db, ATAM["gladiator"])
        # ─── SET 1: Global Shared Objects ───────────────────────────────
        self.record_level:      RecordLevel         = record_level
        self.hyper:              HyperParameters    = hyper
        self.db:                 RamDB              = db
        self.training_data:      TrainingData       = training_data
        self.config:             Config             = Config(self , hyper=self.hyper, db=self.db, training_data=self.training_data, )
        self.setup                                  = setup
        #self.lego_selector:     LegoSelector       = LegoSelector()

        # ─── SET 2: Core Stable Metrics (Always Present) ────────────────
        self.mae:               float               = None
        self.lowest_mae:        float               = 6.9e69
        self.lowest_mae_epoch:  int                 = 0
        self.run_id:            int                 = run_id
        self.gladiator:         str                 = setup["gladiator"]
        self.seed:              int                 = seed
        self.time_start:        datetime            = datetime.now()
        self.time_end:          datetime            = None
        self.last_epoch:        int                 = 0
        self.converge_cond:     str                 = None

        # ─── SET 3: Old dictionary converted
        self.bd_target_alpha:   float               = None
        self.bd_target_beta:    float               = None
        self.bd_target_alpha_unscaled:  float       = None
        self.bd_target_beta_unscaled:   float       = None
        self.bd_threshold:      float               = None
        self.bd_label_alpha:    float               = None
        self.bd_label_beta:     float               = None
        self.bd_target_alpha:   float               = None
        self.bd_correct:        int                 = 0

    def record_finish_time(self):
        self.time_end = datetime.now()

    def display(self):
        print("\n📊 Training Run Summary")
        print("────────────────────────")
        print(f"Timestamp:         {self.timestamp}")
        print(f"Runtime (s):       {self.runtime_seconds:.2f}")
        print(f"Final Error:       {self.final_error:.5f}")
        print(f"Seed:              {self.seed}")
        print(f"Run ID:            {self.run_id}")
        print(f"Phase ID:          {self.phase_id}")
        print(f"Sweep Group:       {self.sweep_group_id}")
        print(f"Is Baseline?       {self.is_baseline}")
        print(f"Is Reference?      {self.is_reference}")
        print("\n🧬 Config Metadata")
        print("────────────────────────")
        for k, v in self.config_metadata.items():
            print(f"{k:20}: {v}")

    def should_record(self, minimum_level: RecordLevel) -> bool:
        #return True
        return self.record_level.value >= minimum_level.value

    @property
    def time_seconds(self) -> float:
        if self.time_start is not None and self.time_end is not None:
            return (self.time_end - self.time_start).total_seconds()
        return -1.0

    @property
    def accuracy_regression(self) -> Optional[float]:
        """
        Returns a regression-style 'accuracy' score as:
        1 - (MAE / mean absolute target), capped between 0 and 1.
        Useful for comparing performance across problem types.

        Returns:
            float: Regression accuracy, or None if data is missing.
        """
        #print(f"accuracy_regression {self.training_data.problem_type}")
        mae = self.get("mae")
        mean_target = self.training_data.mean_absolute_target
        if mae is None or mean_target == 0 : return 0            # Avoid divide-by-zero or missing data
        return  (1.0 - (mae / mean_target)) * 100                      #return max(0.0, min(1.0, accuracy))  # Clamp to [0, 1]

    @property
    def accuracy_bd(self) -> float:
        #print("accuracy_bd")
        bd_correct = self.bd_correct
        samples = self.training_data.sample_count
        return (bd_correct / samples ) * 100

    @property
    def accuracy(self) -> float:
        if self.training_data.problem_type == "Binary Decision":
            return self.accuracy_bd
        else:
            return self.accuracy_regression

