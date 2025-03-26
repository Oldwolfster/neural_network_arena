import sqlite3
import os
import datetime
import ast  # For safely evaluating strings back to data structures
from tabulate import tabulate
from src.engine.Utils_DataClasses import ReproducibilitySnapshot


def record_snapshot(snapshot: ReproducibilitySnapshot):
    conn = get_db_connection()
    create_snapshot_table(conn)
    insert_snapshot(conn, snapshot)
    conn.close()


def insert_snapshot(conn, snapshot: ReproducibilitySnapshot):
    cursor = conn.cursor()
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.datetime.now()

    cursor.execute('''
    INSERT INTO reproducibility_snapshots (
        run_id, timestamp, arena_name, gladiator_name, architecture, problem_type, 
        loss_function_name, hidden_activation_name, output_activation_name, 
        weight_initializer_name, normalization_scheme, seed, learning_rate, 
        epoch_count, convergence_condition, runtime_seconds, final_error
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        run_id, timestamp, snapshot.arena_name, snapshot.gladiator_name,
        repr(snapshot.architecture), snapshot.problem_type, snapshot.loss_function_name,
        snapshot.hidden_activation_name, snapshot.output_activation_name,
        snapshot.weight_initializer_name, snapshot.normalization_scheme,
        snapshot.seed, snapshot.learning_rate, snapshot.epoch_count, snapshot.convergence_condition,
        snapshot.runtime_seconds, snapshot.final_error
    ))
    conn.commit()
    print(f"Snapshot saved with run_id: {run_id}")


def list_snapshots():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT * FROM reproducibility_snapshots ORDER BY timestamp
    ''')
    rows = cursor.fetchall()
    headers = [description[0] for description in cursor.description]
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    conn.close()


def create_snapshot_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reproducibility_snapshots (
        run_id TEXT PRIMARY KEY,
        timestamp DATETIME,
        arena_name TEXT,
        gladiator_name TEXT,
        architecture TEXT,
        problem_type TEXT,
        loss_function_name TEXT,
        hidden_activation_name TEXT,
        output_activation_name TEXT,
        weight_initializer_name TEXT,
        normalization_scheme TEXT,
        seed INTEGER,
        learning_rate REAL,
        epoch_count INTEGER,
        convergence_condition TEXT,
        runtime_seconds REAL,
        final_error REAL
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
