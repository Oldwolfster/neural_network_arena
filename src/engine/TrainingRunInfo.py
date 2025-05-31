from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
#from src.engine.Reporting import generate_reports, create_weight_tables
from src.ArenaSettings import HyperParameters
from src.Legos.LegoSelector import LegoSelector
from src.engine.Config import Config
from src.engine.RamDB import RamDB
from src.engine.TrainingData import TrainingData


class TrainingRunInfo:
    def __init__(self, hyper: HyperParameters, db: RamDB, training_data : TrainingData,  setup, seed, run_id):
        #create_weight_tables(db, ATAM["gladiator"])
        # ─── SET 1: Global Shared Objects ───────────────────────────────
        self.hyper:              HyperParameters    = hyper
        self.db:                 RamDB              = db
        self.training_data:      TrainingData       = training_data
        self.config:             Config             = Config(hyper=self.hyper,db=self.db, training_data=self.training_data, gladiator_name=setup["gladiator"])
        self.lego_selector:      LegoSelector       = LegoSelector()
        self.setup                                  = setup

        # ─── SET 2: Core Stable Metrics (Always Present) ────────────────
        self.gladiator_name:    str                 = setup["gladiator"]
        self.time_start:        datetime            = datetime.now
        self.time_end:          datetime            = None
        self.seed:              int                 = seed
        self.run_id:            int                 = run_id

        # ─── SET 3: Dictonary for holding everything else
        self.config_metadata: Dict[str, Any] = {}

    # ─── Idiot-Proof Helpers ────────────────────────────────────────
    def get(self, key, default=None):
        """
        Safely retrieve a value from config_metadata.

        Args:
            key (str): The metadata key to retrieve.
            default (Any, optional): Value to return if key is not present. Defaults to None.

        Returns:
            Any: The stored value, or default if not found.
        """
        return self.config_metadata.get(key, default)

    def set(self, key, value):
        self.config_metadata[key] = value

    def set_if_lower(self, key, value) -> bool:
        old = self.config_metadata.get(key, float("inf"))
        if value < old:
            self.config_metadata[key] = value
            return True
        return False

    def set_if_higher(self, key, value) -> bool:
        old = self.config_metadata.get(key, float("-inf"))
        if value > old:
            self.config_metadata[key] = value
            return True
        return False

    def record_finish_time(self):
        self.time_end = datetime.now

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

    @property
    def time_seconds(self) -> float:
        if self.time_start is not None and self.time_end is not None:
            return (self.time_end - self.time_start).total_seconds()
        return -1.0
