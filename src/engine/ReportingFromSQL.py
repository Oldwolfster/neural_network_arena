import json
import sqlite3
from tabulate import tabulate
from src.engine.RamDB import RamDB
from src.engine.Utils import smart_format


def generate_reports(db : RamDB):
    generate_iteration_report(db)


def generate_iteration_report(db: RamDB):
    #db.query_print("SELECT * FROM Iteration")
    #db.query_print("Select * From Neuron")
    SQL = """
    SELECT  *
    FROM    Iteration I
    JOIN    Neuron N
    ON      I.model_id  = N.model 
    AND     I.epoch     = N.epoch_n
    AND     I.iteration = N.iteration_n
    ORDER BY epoch, iteration, model, nid 
    """

    data = db.query(SQL)
    print("NEURON LEVEL REPORT==========================================================================")
    #print(data)
    results = neuron_report_organize_info(data)
    iteration_report = tabulate(results, headers="keys", tablefmt="fancy_grid")
    print(iteration_report)



def neuron_report_organize_info(rows):
    """
    Organize detailed neuron information for the report.
    Includes Model, Neuron Context, Weight * Input, Bias, Raw Sum, Output, and New Weights.
    """
    report_data = []

    for row in rows:
        # Extract basic information
        model = row['model_id']
        context = f"{row['epoch']} / {row['iteration']} / {row['nid']}"
        prediction_logic = build_prediction_logic(row)
        new_weights_logic = build_new_weights_logic(row)  # Add new weights logic

        # Calculate bias and raw sum
        bias = row.get('bias_before', 0)  # Default to 0 if bias is missing
        weights = json.loads(row.get('weights_before', '[]'))
        inputs = json.loads(row.get('inputs', '[]'))
        raw_sum = sum(w * inp for w, inp in zip(weights, inputs)) + bias

        # Determine output (activation function currently linear)
        output = f"{row['activation']}: {smart_format(raw_sum)}"  # Use activation function name from the row

        # Append to the report data
        report_data.append({
            "Model": model,
            "Epc / Iter / Nrn": context,
            "Weight * Input": prediction_logic,
            "Bias": smart_format(bias),
            "Raw Sum": smart_format(raw_sum),
            "Output": output,
            "New Weights": new_weights_logic  # Add new weights to the report
        })

    return report_data



def build_new_weights_logic(row):
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


def build_prediction_logic(row):
    """
    Build prediction logic for a single neuron (row).
    Loops through weights and inputs, generating labeled calculations.
    """
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

