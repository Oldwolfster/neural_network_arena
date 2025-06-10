from datetime import datetime
from typing import List, Dict, Any
from itertools import product
import os
import json
from src.Legos.LegoLister import LegoLister
from src.engine.SQL import get_db_connection

class TrainingBatchInfo:
    def __init__(self, gladiators, arenas, dimensions: Dict[str, List[Any]]):
        self.conn           = get_db_connection()
        self.initialized    = False
        self.gladiators     = gladiators
        self.arenas         = arenas
        self.dimensions     = dimensions
        self.lister         = LegoLister()
        self                . create_table_if_needed()
        self                . set_ids()
        print               (f"self.id_of_current  = {self.id_of_current  } and self.id_of_last  = {self.id_of_last  }")
        self                . prepare_batch_table()
        print               (f"self.id_of_current  = {self.id_of_current  } and self.id_of_last  = {self.id_of_last  }")

    def run_sql(self,sql) -> int:
        with self.conn:
            result = self.conn.execute(sql).fetchone()
            return result[0] if result and result[0] is not None else 0

    def mark_done_and_get_next_config(self):
        self.run_sql(f"UPDATE training_batch_tasks SET status = 'done' WHERE pk = {self.id_of_current}")
        if self.initialized: self.id_of_current +=1 #accounts for first one which would be one to high otherwise.
        self.initialized = True
        if self.id_of_current > self.id_of_last:
            print("🎉 All tasks complete.\n")
            return None
        else:
            #config = self.run_sql(f"SELECT config FROM training_batch_tasks  WHERE pk = {self.id_of_current}")
            #return json.loads(config)
            raw = self.run_sql(f"SELECT config FROM training_batch_tasks WHERE pk = {self.id_of_current}")
            config = json.loads(raw)
            return self.inflate_config(config)


    def inflate_config(self, setup: Dict[str, Any]) -> Dict[str, Any]:
        for key, val in setup.items():
            if isinstance(val, str) and key in self.lister.registry:
                setup[key] = self.lister.get_lego(key, val)
        return setup


    def prepare_batch_table(self):
        if self.id_of_current == 0:
            print("🧪 No existing batch found — starting a new one.")
            _BatchGenerationLogic(self.conn, self.gladiators, self.arenas, self.dimensions).generate()
            self.set_ids()
        else:
            print("📌 Resuming existing batch.")

    def set_ids(self):
        self.id_of_current  = self.run_sql("SELECT pk FROM training_batch_tasks WHERE status = 'pending' ORDER BY pk ASC LIMIT 1")
        self.id_of_last     = self.run_sql("SELECT MAX(pk) FROM training_batch_tasks")

    def create_table_if_needed(self):
        self.create_table_if_needed_batch_run()
        self.create_table_if_needed_batch_task

    def create_table_if_needed(self):
        with self.conn:
            self.conn.execute('''
                --CREATE TABLE IF NOT EXISTS training_batch_tasks (
                CREATE TABLE IF NOT EXISTS training_batch_tasks (
                    pk INTEGER PRIMARY KEY AUTOINCREMENT,
                    gladiator TEXT,
                    arena TEXT,
                    config TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def create_table_if_needed_batch_task(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS batch_task (
                    batch_task_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_run_id INTEGER,           -- FK to batch_run(batch_run_id)
                    arena TEXT,                     -- Which arena this task is for
                    gladiator TEXT,                 -- Gladiator variant (champ + tweaks)
                    config TEXT,                    -- JSON config for this specific task
                    status TEXT DEFAULT 'pending', -- 'pending', 'running', 'completed', 'error'
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            ''')



    def create_table_if_needed_batch_run(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS batch_run (
                    batch_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,                   -- "AutoML Defaults v1", "Hinge Sweep"
                    description TEXT,            -- Longer human-readable summary
                    gladiators TEXT,              -- Base Champ class name
                    arenas TEXT,                     -- Which arena this task is for
                    config TEXT,                 -- JSON config (rule snapshot, hyperparams, etc.)
                    is_autogen BOOLEAN,          -- True if rule-generated (AutoML), false if manual
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            ''')

class _BatchGenerationLogic:
    def __init__(self, conn, gladiators, arenas, dimensions):
        self.conn       = conn
        self.gladiators = gladiators
        self.arenas     = arenas
        self.dimensions = dimensions
        self.lister     = LegoLister()

    def generate(self):
        self.expand_wildcards()
        setups = self.build_setups()

        with self.conn:
            self.conn.execute("DELETE FROM training_batch_tasks")  # ensure clean slate
            self.conn.execute("DELETE FROM sqlite_sequence WHERE name = 'training_batch_tasks'") # Reset counter
            for setup in setups:
                print(f"setup = {setup}")
                self.conn.execute('''
                    INSERT INTO training_batch_tasks (gladiator, arena, config)
                    VALUES (?, ?, ?)
                ''', (setup["gladiator"], setup["arena"], json.dumps(setup)))

    def expand_wildcards(self):
        for key, val in self.dimensions.items():
            if val == "*":
                legos = self.lister.list_legos(key)
                self.dimensions[key] = list(legos.values())

    def build_setups(self) -> List[Dict[str, Any]]:
        keys   = list(self.dimensions.keys())
        values = list(self.dimensions.values())
        combos = list(product(*values))
        setups = []

        for arena in self.arenas:
            for gladiator in self.gladiators:
                lr_flag = self.model_sets_lr(gladiator)

                for combo in combos:
                    #Worked in ram, not when writing to table --> config = dict(zip(keys, combo))
                    config = {}
                    for k, v in zip(keys, combo):
                        if k in self.lister.registry:
                            config[k] = str(v)  # will call __repr__()
                        else:
                            config[k] = v
                    config.update({
                        "gladiator": gladiator,
                        "arena": arena,
                        "lr_specified": lr_flag
                    })
                    setups.append(config)
        return setups

    def model_sets_lr(self, gladiator_name: str) -> bool:
        for root, _, files in os.walk("coliseum/gladiators"):
            if f"{gladiator_name}.py" in files:
                path = os.path.join(root, f"{gladiator_name}.py")
                with open(path, 'r', encoding='utf-8') as f:
                    return any(
                        "config.learning_rate" in line and not line.strip().startswith("#")
                        for line in f
                    )
        raise FileNotFoundError(f"❌ Could not find file for gladiator '{gladiator_name}'.")

