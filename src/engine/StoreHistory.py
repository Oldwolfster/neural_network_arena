import sqlite3
import os
import datetime
import ast  # For safely evaluating strings back to data structures
from tabulate import tabulate

from src.engine.Config import Config
from src.engine.Utils_DataClasses import NNA_history, ModelInfo
from datetime import datetime

def record_results(TRI, record_level):
    if record_level == 0: return
    config = TRI.config
    lowest_mae = TRI.get("lowest_mae")
    total_error = TRI.get("total_error_for_epoch")
    random_seed = TRI.seed

    TRI.config                  . configure_popup_headers()# MUST OCCUR AFTER CONFIGURE MODEL SO THE OPTIMIZER IS SET
    TRI                         . record_finish_time()

    model_info                  = ModelInfo(TRI.run_id,  TRI.gladiator_name, TRI.config .seconds, TRI.config .cvg_condition, TRI.config .architecture, TRI.config .training_data.problem_type )
    TRI.db.add     (model_info)              #Writes record to ModelInfo table

    conn = get_db_connection()
    create_snapshot_table(conn)
    log_entry = NNA_history.from_config(TRI, config, lowest_mae,total_error, random_seed)
    insert_snapshot(conn, log_entry)
    conn.close()


def insert_snapshot(conn, snapshot: NNA_history):
    cursor = conn.cursor()
    #run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now()

    cursor.execute('''
    INSERT INTO NNA_history (
        run_id, best_mae, timestamp, arena_name, gladiator_name, architecture, problem_type, 
        loss_function_name, hidden_activation_name, output_activation_name, 
        weight_initializer_name, normalization_scheme, seed, learning_rate, 
        epoch_count, convergence_condition, runtime_seconds, final_error
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        snapshot.run_id, snapshot.best_mae, timestamp, snapshot.arena_name, snapshot.gladiator_name,
        repr(snapshot.architecture), snapshot.problem_type
        , snapshot.loss_function_name,        snapshot.hidden_activation_name, snapshot.output_activation_name,
        snapshot.weight_initializer_name, snapshot.normalization_scheme,
        snapshot.seed, snapshot.learning_rate, snapshot.epoch_count, snapshot.convergence_condition,
        snapshot.runtime_seconds, snapshot.final_error
    ))
    conn.commit()
    #print(f"Snapshot saved with run_id: {run_id}")


def list_snapshots_in_console(result_rows: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT seed as _seed,* FROM NNA_history ORDER BY timestamp DESC LIMIT {result_rows} ")

    rows = cursor.fetchall()
    headers = [description[0] for description in cursor.description]
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    conn.close()

import csv

def list_snapshots(result_rows: int, filename="snapshots.csv"):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT seed AS _seed, *
        FROM NNA_history
        ORDER BY timestamp DESC
        LIMIT {result_rows}
    """)

    rows = cursor.fetchall()
    headers = [description[0] for description in cursor.description]

    # 👇 Generate timestamp
    from pathlib import Path
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_folder = Path("..") / "gladiator_matches"
    log_folder.mkdir(parents=True, exist_ok=True)  # Ensure folder exists

    filename = log_folder / f"GladiatorMatches_logfile_{timestamp}.csv"

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"✅ Exported {len(rows)} rows to {filename}")
    conn.close()

    import os

    # Right after conn.close()
    os.startfile(filename)

def create_snapshot_table(conn):
    cursor = conn.cursor()
    cursor.execute('''    
    CREATE TABLE IF NOT EXISTS NNA_history (
        timestamp DATETIME,
        run_id INTEGER,
        runtime_seconds REAL,
        best_mae REAL,
        final_error REAL,        
        arena_name TEXT,
        gladiator_name TEXT,
        architecture TEXT,        
        loss_function_name TEXT,
        hidden_activation_name TEXT,
        output_activation_name TEXT,
        weight_initializer_name TEXT,
        normalization_scheme TEXT,
        learning_rate REAL,
        epoch_count INTEGER,
        convergence_condition TEXT,        
        problem_type TEXT,
        seed INTEGER,
        pk INTEGER PRIMARY KEY AUTOINCREMENT
    )
    ''')
    conn.commit()


def get_db_connection(db_name='arena_history.db', subfolder='history'):
    """
    Connects to an SQLite database located in the specified subfolder within the parent directory of this script.
    If the subfolder does not exist, it is created.

    Parameters:
    - db_name (str): Name of the database file.
    - subfolder (str): Name of the subfolder where the database file is located.

    Returns:
    - conn: SQLite3 connection object.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    grandparent_dir = os.path.abspath(os.path.join(parent_dir, '..'))
    subfolder_path = os.path.join(grandparent_dir, subfolder)
    try:
        os.makedirs(subfolder_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {subfolder_path}: {e}")
        raise
    db_path = os.path.join(subfolder_path, db_name)
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database at {db_path}: {e}")
        raise
