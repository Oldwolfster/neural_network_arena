import sqlite3
import os
import datetime
import ast  # For safely evaluating strings back to data structures


def record_training_data(training_data: list[tuple[float, ...]]):
    print(f"TRAINING DATA:::::::::::::::::\n{training_data}")
    conn = get_db_connection()
    create_table(conn)
    insert_training_data(conn, training_data)
    conn.close()


def insert_training_dataDEPRECATED_USE_RAND_SEED_INSTEAD(conn, training_data: list[tuple[float, ...]]):
    cursor = conn.cursor()
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.datetime.now()
    data_string = repr(training_data)  # Convert the list to a string representation

    cursor.execute('''
    INSERT INTO training_data (run_id, timestamp, data_string)
    VALUES (?, ?, ?)
    ''', (run_id, timestamp, data_string))
    conn.commit()
    print(f"Training data saved with run_id: {run_id}")
    return run_id


def list_runs():
    print("previous runs")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT run_id, timestamp FROM training_data ORDER BY timestamp
    ''')
    runs = cursor.fetchall()
    print("Available runs:")
    for run in runs:
        print(f"Run ID: {run[0]}, Timestamp: {run[1]}")


def create_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS training_data (
        run_id TEXT PRIMARY KEY,
        timestamp DATETIME,
        data_string TEXT
    )
    ''')
    conn.commit()


def retrieve_training_data(run_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT data_string FROM training_data WHERE run_id = ?
    ''', (run_id,))
    result = cursor.fetchone()
    if result:
        data_string = result[0]
        training_data = ast.literal_eval(data_string)  # Safely convert string back to list
        return training_data
    else:
        print(f"No data found for run_id: {run_id}")
        return None


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
    import os

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory of the script directory
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Get the parent directory of the script directory
    grandparent_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    # Construct the path to the subfolder in the parent directory
    subfolder_path = os.path.join(grandparent_dir, subfolder)

    # Ensure the subfolder exists
    try:
        os.makedirs(subfolder_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {subfolder_path}: {e}")
        raise

    # Construct the full path to the database file
    db_path = os.path.join(subfolder_path, db_name)

    # Connect to the database using the full path
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database at {db_path}: {e}")
        raise
