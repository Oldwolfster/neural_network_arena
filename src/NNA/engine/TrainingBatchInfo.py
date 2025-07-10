
from typing import List, Dict, Any
from itertools import product
import os
import json
from src.NNA.Legos.LegoLister import LegoLister
from src.NNA.engine.SQL import get_db_connection

class TrainingBatchInfo:
    def __init__(self, gladiators, arenas, batch_notes, dimensions: Dict[str, List[Any]]):
        self.conn           = get_db_connection()
        self.initialized    = False
        self.gladiators     = gladiators
        self.arenas         = arenas
        self.batch_notes    = batch_notes

        self.dimensions     = dimensions
        self.lister         = LegoLister()
        self.batch_id       = None
        #print               (f"self.id_of_current  = {self.id_of_current  } and self.id_of_last  = {self.id_of_last  }")
        self                . start_new_batch()
        self                . set_ids()
        print               (f"self.id_of_current  = {self.id_of_current  } and self.id_of_last  = {self.id_of_last  }")

    def run_sql(self,sql) -> int:
        with self.conn:
            result = self.conn.execute(sql).fetchone()
            return result[0] if result and result[0] is not None else 0

    def mark_done_and_get_next_config(self):

        sql= (f"UPDATE {self.batch_run_table_name} SET status = 'done' WHERE pk = {self.id_of_current}")
        print(f"sql={sql}")
        self.run_sql(sql)
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


    def start_new_batch(self):
        #if self.id_of_current == 0:
        #    print("🧪 No existing batch found — starting a new one.")
        self.batch_id = self.insert_batch_metadata()  # ✅ Here’s your new batch ID
        self                . create_batch_tables()
        _BatchGenerationLogic(self.conn, self.gladiators, self.arenas, self.batch_id ,self.dimensions).generate()
        self.set_ids()
        #else:
        #    print("📌 Resuming existing batch.")

    def insert_batch_metadata(self) -> int:
        self.create_table_batch_master()

        # ✨ Safely stringify values for JSON
        safe_dimensions = {}
        for key, values in self.dimensions.items():
            safe_dimensions[key] = []
            for val in values:
                if key in self.lister.registry:
                    safe_dimensions[key].append(str(val))
                else:
                    safe_dimensions[key].append(val)

        dimensions_str = json.dumps(safe_dimensions)

        with self.conn:
            cursor = self.conn.execute('''
                INSERT INTO batch_master (dimensions, notes)
                VALUES (?, ?)
            ''', (dimensions_str, self.batch_notes))
            return cursor.lastrowid

    def set_ids(self):
        self.id_of_current  = self.run_sql("SELECT pk FROM training_batch_tasks WHERE status = 'pending' ORDER BY pk ASC LIMIT 1")
        self.id_of_last     = self.run_sql("SELECT MAX(pk) FROM training_batch_tasks")

    def create_batch_tables(self):
        self.create_table_training_batch_tasks()
        self.create_view_training_batch_tasks()

    def create_view_training_batch_tasks(self):
        table_name = f"batch_runs_{self.batch_id}"
        with self.conn:
            # Drop existing view to avoid conflict
            self.conn.execute("DROP VIEW IF EXISTS training_batch_tasks")
            # Create view alias pointing to batch-specific table
            self.conn.execute(f'''
                CREATE VIEW training_batch_tasks AS
                SELECT * FROM {table_name}
            ''')

    @property
    def batch_run_table_name(self)-> str :
        return f"batch_runs_{self.batch_id}"

    def create_table_training_batch_tasks(self, ):
        #table_name = f"batch_runs_{batch_id}"
        with self.conn:
            self.conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.batch_run_table_name} (
                    pk INTEGER PRIMARY KEY AUTOINCREMENT,                    
                    gladiator TEXT,
                    arena TEXT,
                    config TEXT,
                    status TEXT DEFAULT 'pending',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    accuracy REAL,
                    final_mae REAL,
                    best_mae REAL,
                    runtime_seconds REAL,
                    
                    architecture TEXT,        
                    loss_function TEXT,
                    hidden_activation TEXT,
                    output_activation TEXT,
                    weight_initializer TEXT,
                    normalization_scheme TEXT,
                    learning_rate REAL,
                    epoch_count INTEGER,
                    convergence_condition TEXT,        
                    problem_type TEXT,
                    sample_count INTEGER,            
                    
                    target_min REAL,                -- either min numeric or count of smaller class
                    target_max REAL,                -- either max numeric or count of larger class            
                    target_min_label TEXT,          -- e.g., "Repay" or "0"
                    target_max_label TEXT,          -- e.g., "Default" or "1"            
                    target_mean REAL,               -- mean of target values (esp useful in regression)
                    target_stdev REAL,               -- standard deviation of targets
                    notes TEXT,                      -- Optional remarks (e.g., 'testing AdamW with tanh glitch patch')
                    seed INTEGER                    
                )                
            ''')

    def ORIGINALcreate_table_training_batch_tasks(self):
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

    def create_table_batch_master(self):
        with self.conn:
            self.conn.execute('''
            CREATE TABLE IF NOT EXISTS batch_master (
                pk INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                dimensions TEXT,  -- JSON string like {"seed":[1,2], "activation":["ReLU",...]}
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            );

            ''')

class _BatchGenerationLogic:
    def __init__(self, conn, gladiators, arenas, batch_id, dimensions):
        self.conn       = conn
        self.gladiators = gladiators
        self.arenas     = arenas
        self.dimensions = dimensions
        self.batch_id   = batch_id
        self.lister     = LegoLister()

    def training_batch_tasks_name(self,batch_id)-> str :
        return f"batch_runs_{batch_id}"

    def generate(self):
        self.expand_wildcards()
        setups = self.build_setups()

        with self.conn:
            #self.conn.execute("DELETE FROM training_batch_tasks")  # ensure clean slate
            #self.conn.execute("DELETE FROM sqlite_sequence WHERE name = 'training_batch_tasks'") # Reset counter
            for setup in setups:
                print(f"setup = {setup}")
                table_name = self.training_batch_tasks_name(self.batch_id)
                self.conn.execute(f'''
                    INSERT INTO {table_name} (gladiator, arena, config)
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
        import os

        this_dir = os.path.dirname(__file__)
        gladiator_dir = os.path.join(this_dir, "..", "coliseum", "gladiators")
        gladiator_dir = os.path.abspath(gladiator_dir)

        for root, _, files in os.walk(gladiator_dir):
            if f"{gladiator_name}.py" in files:
                path = os.path.join(root, f"{gladiator_name}.py")
                with open(path, 'r', encoding='utf-8') as f:
                    return any(
                        "config.learning_rate" in line and not line.strip().startswith("#")
                        for line in f
                    )
        raise FileNotFoundError(f"❌ Could not find file for gladiator '{gladiator_name}'.")

