import sqlite3
import os
import datetime


from src.NNA.engine.SQL import get_db_connection

from src.NNA.engine.Utils_DataClasses import NNA_history, ModelInfo, RecordLevel

from datetime import datetime

import csv
from pathlib import Path

def record_results(TRI):
    TRI.record_finish_time()
    if not TRI.should_record(RecordLevel.SUMMARY):
        return

    config = TRI.config

    if TRI.should_record(RecordLevel.FULL):
        TRI.config.configure_popup_headers()  # MUST OCCUR AFTER CONFIGURE MODEL SO THE OPTIMIZER IS SET
        model_info = ModelInfo(
            TRI.run_id,
            TRI.gladiator,
            TRI.time_seconds,
            TRI.converge_cond,
            TRI.config.architecture,
            TRI.config.training_data.problem_type
        )
        TRI.db.add(model_info)  # Writes record to ModelInfo table

    conn = get_db_connection()
    create_snapshot_table(conn)
    log_entry = NNA_history.from_config(TRI, config)
    insert_snapshot(conn, log_entry)
    conn.close()


def insert_snapshot(conn, snapshot: NNA_history):
    cursor = conn.cursor()
    # run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now()

    cursor.execute('''
        INSERT INTO NNA_history (
            timestamp, 
            run_id,
            runtime_seconds, 
            gladiator, 
            arena, 
            accuracy, 
            best_mae, 
            final_mae,            
            architecture,             
            loss_function, 
            hidden_activation, 
            output_activation, 
            weight_initializer, 
            normalization_scheme,            
            learning_rate, 
            epoch_count, 
            convergence_condition,
            problem_type,
            sample_count,
            target_min,
            target_max,
            target_min_label,
            target_max_label,
            target_mean,
            target_stdev,
            notes,
            rerun_config,  
            seed
             
        ) VALUES (?,  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,
        snapshot.run_id,
        snapshot.runtime_seconds,
        snapshot.gladiator,
        snapshot.arena,
        snapshot.accuracy,
        snapshot.best_mae,
        snapshot.final_mae,
        repr(snapshot.architecture),
        snapshot.loss_function,
        snapshot.hidden_activation,
        snapshot.output_activation,
        snapshot.weight_initializer,
        snapshot.normalization_scheme,
        snapshot.learning_rate,
        snapshot.epoch_count,
        snapshot.convergence_condition,
        snapshot.problem_type,
        snapshot.sample_count,
        snapshot.target_min,
        snapshot.target_max,
        snapshot.target_min_label,
        snapshot.target_max_label,
        snapshot.target_mean,
        snapshot.target_stdev,
        snapshot.notes,
        snapshot.rerun_config,
        snapshot.seed,
    ))
    conn.commit()

    # Write the same snapshot to CSV (create if not exists, otherwise append)
    export_snapshot_to_csv(snapshot, timestamp)


def export_snapshot_to_csv(snapshot: NNA_history, timestamp, csv_filename='arena_history.csv', subfolder='history'):
    """
    Appends a single NNA_history snapshot to a CSV file. If the CSV does not exist, it is created with headers.

    Parameters:
    - snapshot (NNA_history): The snapshot data to write.
    - timestamp (datetime): The timestamp associated with this snapshot.
    - csv (str): The name of the CSV file (default: 'arena_history.csv').
    - subfolder (str): The subfolder (relative to the grandparent directory) where the CSV is stored.
    """
    # Determine the subfolder path (same logic as get_db_connection)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    grandparent_dir = os.path.abspath(os.path.join(parent_dir, '..'))
    subfolder_path = os.path.join(grandparent_dir, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)

    csv_path = os.path.join(subfolder_path, csv_filename)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # If the file is new, write header row first
        if not file_exists:
            headers = [
                'timestamp',
                'run_id',
                'runtime_seconds',
                'gladiator',
                'arena',
                'accuracy',
                'best_mae',
                'final_mae',
                'architecture',
                'loss_function',
                'hidden_activation',
                'output_activation',
                'weight_initializer',
                'normalization_scheme',
                'learning_rate',
                'epoch_count',
                'convergence_condition',
                'problem_type',
                'sample_count',
                'target_min',                # either min numeric or count of smaller class
                'target_max',
                'target_min_label',          # e.g., "Repay" or "0"
                'target_max_label',          # e.g., "Default" or "1"
                'target_mean',               # mean of target values (esp useful in regression)
                'target_stdev',              #-- standard deviation of targets
                'notes',    # Optional remarks (e.g., 'testing AdamW with tanh glitch patch')
                'rerun_config',# Serialized config for re-running this experiment

                'seed'

            ]
            writer.writerow(headers)

        # Write the snapshot's data
        writer.writerow([
            timestamp.isoformat(),
            snapshot.run_id,
            snapshot.runtime_seconds,
            snapshot.gladiator,
            snapshot.arena,
            snapshot.accuracy,
            snapshot.best_mae,
            snapshot.final_mae,
            repr(snapshot.architecture),
            snapshot.loss_function,
            snapshot.hidden_activation,
            snapshot.output_activation,
            snapshot.weight_initializer,
            snapshot.normalization_scheme,
            snapshot.learning_rate,
            snapshot.epoch_count,
            snapshot.convergence_condition,
            snapshot.problem_type,
            snapshot.sample_count,
            snapshot.target_min,
            snapshot.target_max,
            snapshot.target_min_label,
            snapshot.target_max_label,
            snapshot.target_mean,
            snapshot.target_stdev,
            snapshot.notes,
            snapshot.rerun_config,
            snapshot.seed
        ])


def create_snapshot_table(conn):
    cursor = conn.cursor()
    cursor.execute('''    
        CREATE TABLE IF NOT EXISTS NNA_history (
            timestamp DATETIME,
            run_id INTEGER,
            runtime_seconds REAL,
            gladiator TEXT,      
            arena TEXT,
            accuracy REAL,
            best_mae REAL,
            final_mae REAL,
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
              
            rerun_config TEXT,               -- Serialized config for re-running this experiment
            seed INTEGER,
            pk INTEGER PRIMARY KEY AUTOINCREMENT
        )
    ''')
    conn.commit()
