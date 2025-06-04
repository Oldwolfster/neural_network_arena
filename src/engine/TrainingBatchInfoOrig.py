from datetime import datetime
from typing import List
from typing import Dict
from typing import Any
from itertools import product
import os
from src.Legos.LegoLister import LegoLister
import json
from src.engine.SQL import get_db_connection

# test = lister.list_legos("hidden_activation")
# test = lister.list_legos("output_activation")
# test = lister.list_legos("optimizer")
# test = lister.list_legos("scaler") # works a bit different....
# test = lister.list_legos("initializer")

class TrainingBatchInfo:
    def __init__(self, gladiators, arenas, dimensions: Dict[str, List[Any]]):
        self.gladiators = gladiators
        self.conn       = get_db_connection()
        self.lister     = LegoLister()
        self.arenas     = arenas
        self.dimensions = dimensions
        self.setups     = []
        self.check_for_continue()
        self.build_run_instructions()
        self.cfg_count  = len(self.setups)
        self.write_setups_to_db()
        self.setups     = None
        self.id_of_last = None

    def expand_wildcards(self):
        for key, val in self.dimensions.items():
            if val == "*":
                legos = self.lister.list_legos(key)
                # → Convert “legos” from a dict(name→obj) into a list of obj’s
                self.dimensions[key] = list(legos.values())

    def check_for_continue(self):
        self.create_training_batch_tasks_table()
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM training_batch_tasks
            WHERE status = 'pending'
        ''')
        pending_count = cursor.fetchone()[0]

        if pending_count > 0:
            print(f"📌 Resuming existing batch with {pending_count} pending setups...")
            self.setups = []  # skip building new ones
        else:
            print("🧪 No existing batch found — starting a new one.")


    def build_run_instructions(self):
        self.expand_wildcards()
        config_keys = list(self.dimensions.keys())
        config_values = list(self.dimensions.values())
        combos = list(product(*config_values))

        for gladiator in self.gladiators:
            lr_flag = self.model_explicitly_sets_lr(gladiator)
            for arena in self.arenas:
                for combo in combos:
                    config_dict = dict(zip(config_keys, combo))
                    config_dict["gladiator"] = gladiator
                    config_dict["arena"] = arena
                    config_dict["lr_specified"] = lr_flag
                    self.setups.append(config_dict)

    def model_explicitly_sets_lr(self, gladiator_name: str) -> bool:
        for root, _, files in os.walk("coliseum/gladiators"):
            for file in files:
                if file == f"{gladiator_name}.py":
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        for line in f:
                            if "config.learning_rate" in line and not line.strip().startswith("#"):
                                return True
                    return False
        raise FileNotFoundError(f"❌ Could not find file for gladiator '{gladiator_name}' (expected '{gladiator_name}.py') in 'coliseum/gladiators'.")

    def create_training_batch_tasks_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_batch_tasks (
                pk INTEGER PRIMARY KEY AUTOINCREMENT,
                gladiator TEXT,
                arena TEXT,
                config TEXT,  -- JSON-serialized config dict
                status TEXT DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def write_setups_to_db(self):
        self.create_training_batch_tasks_table()
        cursor = self.conn.cursor()

        for setup in self.setups:
            gladiator = setup["gladiator"]
            arena = setup["arena"]
            config_json = json.dumps(setup)

            cursor.execute('''
                INSERT INTO training_batch_tasks (gladiator, arena, config)
                VALUES (?, ?, ?)
            ''', (gladiator, arena, config_json))
        self.conn.commit()

    def mark_done_and_get_next_config(self):
        cursor = self.conn.cursor()

        # Step 1: If we just finished one, mark it done
        if self.id_of_last:
            cursor.execute('''
                UPDATE training_batch_tasks
                SET status = 'done'
                WHERE pk = ?
            ''', (self.id_of_last,))
            self.conn.commit()

        # Step 2: Get the next pending setup
        cursor.execute('''
            SELECT pk, config FROM training_batch_tasks
            WHERE status = 'pending'
            ORDER BY pk ASC
            LIMIT 1
        ''')
        row = cursor.fetchone()

        if not row:
            return None  # All done

        self.id_of_last = row[0]
        config_dict = json.loads(row[1])
        return config_dict
