from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.ArenaSettings import HyperParameters
from src.Legos.LegoSelector import LegoSelector
from src.engine.Config import Config
from src.engine.RamDB import RamDB
from src.engine.TrainingData import TrainingData



class TrainingRunInfo:
    def __init__(self, hyper: HyperParameters, db: RamDB, training_data : TrainingData, config: Config, gladiator_name:str):
        # ─── SET 1: Global Shared Objects ───────────────────────────────
        self.hyper:              HyperParameters         = hyper
        self.db:                 RamDB                   = db
        self.training_data:      TrainingData            = training_data
        self.config:             Config                  = config
        self.lego_selector:      LegoSelector            = LegoSelector()

        # ─── SET 2: Core Stable Metrics (Always Present) ────────────────
        self.gladiator_name: str                         = gladiator_name

        #coming soon timestart:          datetime = datetime.now
        #coming soon timeend:            datetime = None
        #coming soon runtime_seconds:    Optional[float] = None
        #coming soon final_error:        Optional[float] = None
        #coming soon seed:               Optional[int] = None
        #coming soon run_id:             Optional[int] = None

        # ─── SET 3: Dimensional Coordinates (i.e. Run Settings) ─────────
        #coming soon config_metadata:                Dict[str, Any] = field(default_factory=dict)
        #coming soon self.config_metadata_phases:    Dict[str, int] = {}  # For phase indexing (temp until smarter)

        # ─── SET 4: Execution Meta (Control Layers) ─────────────────────
        #phase_id:       Optional[int] = None     # e.g., 101 for "LR Sweep Phase"
        #sweep_group_id: Optional[int] = None     # used to group together settings during sweeps
        #is_baseline:    bool = False             # Was this a default fallback?
        #is_reference:   bool = False             # Is this a winning config for reuse?

    # ─── Idiot-Proof Helpers ────────────────────────────────────────
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
