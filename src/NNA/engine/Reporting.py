import json
from tabulate import tabulate
from typing import List

from src.ArenaSettings import HyperParameters
from src.NNA.engine.Neuron import Neuron
from src.NNA.engine.RamDB import RamDB
from src.NNA.engine.Utils import smart_format
from src.NNA.engine.Utils_DataClasses import Iteration
from src.NNA.Legos.WeightInitializers import *


def generate_reports(db : RamDB, training_data  ):
    summary_report_launch(db)
    print(training_data.get_list())
    #db.query_print("SELECT * FROM WeightAdjustments")
""" 
    db.query_print(  # Examines weight table
       
SELECT 
    W.*, 
    --N.bias_before, 
    --N.bias, 
    json_extract(N.weights_before, '$[0]') AS first_weight_value_before,
    json_extract(N.weights, '$[0]') AS first_weight_value
FROM Neuron N
JOIN Weight W 
    ON N.model = W.model_id
    AND N.epoch_n = W.epoch
    AND N.iteration_n = W.iteration
    AND N.nid = W.nid
WHERE W.weight_id = 1 and w.nid=2
ORDER BY N.model, N.epoch_n, N.iteration_n, N.nid, W.weight_id;

        """

   # )


def create_weight_adjustments_table(db: RamDB, run_id: int, update_or_finalize: str, arg_count=12):
    """
    Creates a dedicated WeightAdjustments_<run_id> table with arg_1..arg_N fields.
    """
    table_name = f"WeightAdjustments_{update_or_finalize}_{run_id}"
    fields = ",\n".join([f"    arg_{i+1} REAL DEFAULT NULL" for i in range(arg_count)])

    sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch        INTEGER NOT NULL,
            iteration    INTEGER NOT NULL,
            nid          INTEGER NOT NULL,
            -- model_id     TEXT NOT NULL, removed - model is part of table name... why have column with 1 unique value??
            weight_index INTEGER NOT NULL,
            batch_id     INTEGER NOT NULL DEFAULT 0,
            {fields}
            
        );
    """

    db.execute(sql)

    db.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_batch_lookup_{update_or_finalize}_{run_id}
        ON {table_name} (epoch, batch_id, nid, weight_index);
    """)



def prep_RamDB():
    db=RamDB()

    #Create dummy records to create table so we can create the view
    dummy_iteration = Iteration(run_id=0, epoch=0, iteration=0, inputs="", target=0.1, prediction=0.1, inputs_unscaled="", target_unscaled=0.1, prediction_unscaled=0.1, prediction_raw=0.1, loss=0.1, loss_gradient=0.1, loss_function="dummy", accuracy_threshold=0.0)
    dummy_neuron = Neuron(0,1,0.0,Initializer_Tiny ,0)
    db.add(dummy_iteration)

    db.add(dummy_neuron,exclude_keys={"activation", "output_neuron"}, run_id=0, epoch_n = 0, iteration_n = 0 )
    #db.execute("CREATE INDEX idx_model_epoch_iteration ON Neuron (model, epoch_n, iteration_n);")
    db.execute("CREATE INDEX idx_epoch_iteration ON Neuron (run_id, epoch_n, iteration_n);")
    db.execute("CREATE INDEX idx__iteration ON Iteration (run_id,  iteration);")


    epoch_create_view_epochSummary(db)
    db.execute("DELETE FROM Iteration")     #Delete dummy records
    db.execute("DELETE FROM Neuron")        #Delete dummy records
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS Weight (
            run_id INT NOT NULL,
            epoch INTEGER NOT NULL,
            iteration INTEGER NOT NULL,
            nid INTEGER NOT NULL,
            weight_id INTEGER NOT NULL,
            value_before REAL NOT NULL,
            value REAL NOT NULL,            
            PRIMARY KEY (run_id, epoch, iteration, nid, weight_id)       
        );""")

    db.execute("""
        CREATE TABLE IF NOT EXISTS ErrorSignalCalcs (
            run_id       INTEGER NOT NULL,
            epoch        INTEGER NOT NULL,
            iteration    INTEGER NOT NULL,            
            nid          INTEGER NOT NULL,
            weight_id    INTEGER NOT NULL,
            arg_1        REAL NOT NULL,
            op_1         TEXT NOT NULL CHECK (op_1 IN ('+', '-', '*', '/', '=')),  
            arg_2        REAL NOT NULL,
            op_2         TEXT NOT NULL CHECK (op_2 IN ('+', '-', '*', '/', '=')),
            arg_3        REAL DEFAULT NULL,
            op_3         TEXT DEFAULT NULL CHECK (op_3 IN ('+', '-', '*', '/', '=')),
            result       REAL NOT NULL,
            PRIMARY KEY (run_id, epoch, iteration,  nid,weight_id)  -- Ensures unique calculations per neuron per step
        );""")

    #this will no longer run with engine refactoredto TurboForge... instead atomic creates individually
    #for gladiator in gladiators: # Each model gets its own isolated table
    #    create_weight_adjustments_table(db, gladiator, "update")
    #    create_weight_adjustments_table(db, gladiator, "finalize")

    return db

def create_weight_tables(db, run_id):
    create_weight_adjustments_table(db, run_id, "update")
    create_weight_adjustments_table(db, run_id, "finalize")
    delete_records         (db, run_id) # in case it had been run by LR sweep

def delete_records(db, run_id, possible_columns=None):
        """
        Deletes records across all tables where one of the possible columns matches the given gladiator.

        Args:
            db: Your database connection or wrapper.
            gladiator (str): The model ID or name to delete.
            possible_columns (list of str, optional): Columns to check, in order of preference.
        """
        if possible_columns is None:
            possible_columns = ['run_id', 'model', 'gladiator']

        # Delete tables that have run_id in name rather than waste a column
        table_name = f"WeightAdjustments_update_{run_id}"
        db.execute(f"DELETE FROM {table_name}")
        table_name = f"WeightAdjustments_finalize_{run_id}"
        db.execute(f"DELETE FROM {table_name}")

        # Get list of all table names
        tables = db.query("SELECT name FROM sqlite_master WHERE type='table';")

        for table_row in tables:
            #ez_debug(table_row=table_row)
            table_name = table_row['name']

            # Get column names for this table
            columns = db.query(f"PRAGMA table_info({table_name});", as_dict = False)
            column_names = [col[1] for col in columns]

            # Find first matching column
            matching_column = next((col for col in possible_columns if col in column_names), None)

            if matching_column:
                #print(f"🧹 Deleting from {table_name} where {matching_column} = '{gladiator}'")
                #db.execute(f"DELETE FROM {table_name} WHERE {matching_column} = ?", (gladiator,))
                db.execute(f"DELETE FROM {table_name} WHERE {matching_column} = '{run_id}'")


def summary_report_launch(db: RamDB):   #S.*, I.* FROM EpochSummary S
    # run_id           │   epoch │   correct │   wrong │   mean_absolute_error │   mean_squared_error │   root_mean_squared_error │ weights                                    │ biases              │   seconds │
    # round((S.correct * 1.0 / (S.correct + S.wrong)) * 100,2) AS [Accuracy],
    SQL = """
        SELECT  S.run_id as [Gladiator\nComparison], ROUND(I.seconds, 2) AS [Run\nTime],
                S.epoch[Epoch of\nConv], s.correct[Correct], s.wrong[Wrong], 
                Accuracy,
                s.mean_absolute_error[Mean\nAbs Err], s.root_mean_squared_error[RMSE], s.weights[Weights], s.biases[Biases] 
        FROM EpochSummary S
        JOIN (
            Select run_id,max(epoch) LastEpoch
            FROM EpochSummary 
            GROUP BY run_id
            ) M
        On S.run_id = M.run_id and S.epoch = M.LastEpoch
        JOIN ModelInfo I 
        ON S.run_id = I.run_id        
        """
    print(f"GLADIATOR COMPARISON ================================================")
    summary_overview = db.query_print(SQL, print_source=False)


def epoch_report_launch(db: RamDB):
    print("EPOCH SUMMARY ****************************************************")
    db.query_print("SELECT * FROM EpochSummary", surpress_call_stack=True)

def epoch_create_view_epochSummary(db: RamDB):
    SQL = """
        CREATE VIEW IF NOT EXISTS EpochSummary AS
        SELECT DISTINCT
            m.run_id,
            m.epoch,
            m.correct,
            round((m.correct * 1.0 / (m.correct + m.wrong)) * 100,3) AS [Accuracy],
            m.wrong,
            m.mean_absolute_error,
            m.mean_absolute_error_unscaled,
            m.mean_squared_error,
            m.root_mean_squared_error,
            n.combined_weights AS weights,
            n.combined_biases AS biases
        FROM (
            SELECT 
                run_id,
                epoch,
                SUM(is_true) AS correct,
                SUM(is_false) AS wrong,
                AVG(absolute_error) AS mean_absolute_error,
                AVG(absolute_error_unscaled) AS mean_absolute_error_unscaled,
                SUM(squared_error) / COUNT(*) AS mean_squared_error,
                SQRT(SUM(squared_error) / COUNT(*)) AS root_mean_squared_error
            FROM Iteration
            GROUP BY run_id, epoch
        ) m
        LEFT JOIN (
            SELECT 
                run_id,
                epoch_n AS epoch,
                GROUP_CONCAT(nid || ': ' || weights, '\n') AS combined_weights,
                GROUP_CONCAT(nid || ': ' || bias, '\n') AS combined_biases
            FROM Neuron
            WHERE (run_id, epoch_n, iteration_n) IN (
                SELECT 
                    run_id, epoch, MAX(iteration) AS max_iteration
                FROM Iteration
                GROUP BY run_id, epoch
            )
            GROUP BY run_id, epoch_n
        ) n
        ON m.run_id = n.run_id AND m.epoch = n.epoch
        ORDER BY m.epoch;
    """
    try:
        db.execute(SQL)
    except Exception as e:
        print(f"An error occurred: {e}")




#############################NEURON DETAIL REPORT *************************
#############################NEURON DETAIL REPORT *************************
#############################NEURON DETAIL REPORT *************************
def neuron_report_launch(db: RamDB):
    #db.query_print("SELECT * FROM Iteration")
    #db.query_print("Select * From Neuron")
    SQL = """
    SELECT  *
    FROM    Iteration I
    JOIN    Neuron N
    ON      I.run_id  = N.run_id 
    AND     I.epoch     = N.epoch_n
    AND     I.iteration = N.iteration_n
    ORDER BY epoch, iteration, model, nid 
    """

    data = db.query(SQL)
    print("NEURON LEVEL REPORT==========================================================================")
    #print(data)
    #results = neuron_report_organize_info(data)
    #iteration_report = tabulate(results, headers="keys", tablefmt="fancy_grid")
    #print(iteration_report)
    # Organize and tabulate mini-reports
    grouped_reports = neuron_report_organize_info(data)
    for mini_report in grouped_reports:
        print(tabulate(mini_report, headers="keys", tablefmt="fancy_grid"))


def neuron_report_organize_info(query_results):
    """
    Organize detailed neuron information for the report.
    Groups rows by Model, Epoch, and Iteration, and formats them as mini-reports.

    :param query_results: List of dictionaries, results from a query joining neuron and iteration data.
    :return: A list of dictionaries, each representing a neuron row for a given model, epoch, and iteration.

    The report includes:
    - Model, Neuron Context (Epoch / Iteration / Neuron ID)
    - Weight * Input, Bias, Raw Sum, Output, and New Weights
    - Summary-level fields: Target, Prediction, Error, and Loss (left blank for neuron rows)
    """
    report_data = []
    current_idx = 0

    while current_idx < len(query_results):
        # Extract the next group of rows
        group, next_idx = neuron_report_extract_group(query_results, current_idx)
        current_idx = next_idx  # Update the starting index
        mini_report = neuron_report_format_output(group)
        report_data.append(mini_report)
    return report_data


def neuron_report_format_output(rows_for_an_iteration):
    """
    Converts the data from the query to the desired report format for a given iteration

    :param rows_for_an_iteration: List of dictionaries, all neurons for an iteration and iteration summary
    :return: A list of dictionaries, each representing a neuron row for a given model, epoch, and iteration.
    """

    mini_report = []
    for row in rows_for_an_iteration:
        # Extract basic information
        model       = row['model_id']
        context     = f"{row['epoch']} / {row['iteration']} / {row['nid']}"
        weight_inp  = neuron_report_build_prediction_logic(row)
        new_weights = neuron_report_build_new_weights_logic(row)

        # Extract bias and raw sum
        bias        = row.get('bias_before', 0)
        weights     = json.loads(row.get('weights_before', '[]'))
        inputs      = json.loads(row.get('inputs', '[]'))
        raw_sum     = sum(w * inp for w, inp in zip(weights, inputs)) + bias
        output_val  = row.get('output')
        output      = f"{row['activation']}: {smart_format(output_val)}"

        # Append to the report data
        mini_report.append({
            "Model"             : model,
            "Epc / Iter / Nrn"  : context,
            "Weight * Input"    : weight_inp,
            "Bias"              : smart_format(bias),
            "Raw Sum"           : smart_format(raw_sum),
            "Output"            : output,
            "New Weights"       : new_weights,

        })

    # Add summary row
    summary_row = neuron_report_build_iteration_summary(rows_for_an_iteration)
    mini_report.append(summary_row)
    return mini_report


def neuron_report_build_iteration_summary(rows_for_an_iteration):
    """
    Create a summary row for the given group of rows (all neurons for a single iteration).

    :param rows_for_an_iteration: List of dictionaries for a single Model, Epoch, and Iteration.
    :return: A single dictionary representing the summary row.
    """
    #print(rows_for_an_iteration)
    # Use the first row to extract shared fields
    prediction  = smart_format(rows_for_an_iteration[0].get("prediction"))

    target      = smart_format(rows_for_an_iteration[0].get("target"))
    error       = smart_format(rows_for_an_iteration[0].get("prediction")-rows_for_an_iteration[0].get("target"))
    loss        = smart_format(rows_for_an_iteration[0].get("loss"))
    summary_row = {
        "Model"            : "Iteration Summary",
        "Epc / Iter / Nrn" : "Pred - Targ = Err",
        "Weight * Input"   : f"{prediction} - {target} = {error}",
        "Bias"             : "",  # Blank for summary
        "Raw Sum"          : "Loss",
        "Output"           : "MSE",  #  Placeholder for now
        "New Weights"      : loss,  # Currently only MSE
    }
    return summary_row


def neuron_report_extract_group(query_results, start_idx):
    """
    Extract a group of rows from query_results with the same Model, Epoch, and Iteration.

    :param query_results: List of dictionaries, results from a query joining neuron and iteration data.
    :param start_idx: The current starting index in query_results.
    :return: (group, next_idx) where:
        - group is a list of rows for the same Model, Epoch, and Iteration.
        - next_idx is the index to start processing the next group.
    """
    group = []
    initial_key = (
        query_results[start_idx]['model_id'],
        query_results[start_idx]['epoch'],
        query_results[start_idx]['iteration']
    )

    for idx in range(start_idx, len(query_results)):
        row = query_results[idx]
        current_key = (row['model_id'], row['epoch'], row['iteration'])

        if current_key != initial_key:
            return group, idx  # End of the current group

        group.append(row)

    return group, len(query_results)  # End of all rows


def neuron_report_build_new_weights_logic(row):
    """
    Build new weights logic for a single neuron (row).
    Loops through new weights, generating labeled entries.
    """
    nid = row.get('nid')  # Get neuron ID
    new_weights = json.loads(row.get('weights', '[]'))  # Deserialize new weights

    if not new_weights:
        raise ValueError(f"No new weights found for neuron ID {nid}")

    # Generate new weights logic
    new_weights_lines = [
        f"W{i+1} {smart_format(w)}" for i, w in enumerate(new_weights)
    ]

    # Combine into a single multi-line string
    return "\n".join(new_weights_lines)


def neuron_report_build_prediction_logic(row):
    """
    Build prediction logic for a single neuron (row).
    Loops through weights and inputs, generating labeled calculations.
    """
    return
    nid = row.get('nid')  # Get neuron ID
    weights = json.loads(row.get('weights_before', '[]'))  # Deserialize weights
    inputs = json.loads(row.get('inputs', '[]'))  # Deserialize inputs

    # Validate lengths of weights and inputs
    if len(weights) != len(inputs):
        raise ValueError(f"Mismatch in length of weights ({len(weights)}) and inputs ({len(inputs)})")

    # Generate prediction logic
    predictions = []
    for i, (w, inp) in enumerate(zip(weights, inputs), start=1):
        label = f"W{i}I{i}"  # Update label to match new specs
        calculation = f"{label} {smart_format(w)} * {smart_format(inp)} = {smart_format(w * inp)}"
        predictions.append(calculation)

    # Combine multi-line predictions into a single string
    return "\n".join(predictions)

