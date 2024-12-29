import json
import sqlite3

from tabulate import tabulate

from src.engine.RamDB import RamDB


def generate_reports(db : RamDB):
    generate_iteration_report(db)


def generate_iteration_report(db: RamDB):
    db.query_print("SELECT * FROM Iteration")
    db.query_print("Select * From Neuron")
    SQL = """
    SELECT  *
    FROM    Iteration I
    JOIN    Neuron N
    ON      I.model_id  = N.model 
    AND     I.epoch     = N.epoch_n
    AND     I.iteration = N.iteration_n    
    """

    data = db.query(SQL)
    print(data)
    results = extract_on_iteration(data)
    iteration_report = tabulate(results, headers="keys", tablefmt="fancy_grid")
    print(iteration_report)

def extract_on_iteration(data):
    """
    Extract rows grouped by iteration and aggregate prediction logic.
    Includes model, epoch, iteration, prediction details, and target.
    HayabusaTwoNeurons  1  1    N1W1: 10.0 * 6.553336792757052 = 65.53336792757052                 179111
                                N1W2: 10.0 * 5.468733529198841 = 54.687335291988404
                                N2W1: 20.0 * 6.553336792757052 = 131.06673585514105
                                N2W2: 20.0 * 5.468733529198841 = 109.37467058397681
    HayabusaTwoNeurons  2  1    N1W1: 29.487664358972886 * 6.553336792757052 = 193.24259577612779
                                N1W2: 26.262378518905944 * 5.468733529198841 = 143.6219499628523


    """
    output = []
    last_iteration = None
    last_epoch = None
    current_rows = []

    for row in data:
        if (row['iteration'] != last_iteration or row['epoch'] != last_epoch ) and current_rows:
            # Process the current group when iteration changes
            aggregated_prediction = "\n".join(build_prediction_logic(r) for r in current_rows)
            output.append({
                'model': current_rows[0]['model_id'],
                'epoch': current_rows[0]['epoch'],
                'iteration': current_rows[0]['iteration'],
                'prediction_details': aggregated_prediction,
                'target': current_rows[0]['target']
            })
            current_rows = []  # Reset for the next group

        # Add the current row to the group
        current_rows.append(row)
        last_iteration = row['iteration']
        last_epoch = row['epoch']

    # Handle the final group
    if current_rows:
        aggregated_prediction = "\n".join(build_prediction_logic(r) for r in current_rows)
        output.append({
            'model': current_rows[0]['model_id'],
            'epoch': current_rows[0]['epoch'],
            'iteration': current_rows[0]['iteration'],
            'prediction_details': aggregated_prediction,
            'target': current_rows[0]['target']
        })

    return output


def build_prediction_logic(row):
    """
    Build prediction logic for a single neuron (row).
    Loops through weights and inputs, generating labeled calculations.
    """
    nid = row.get('nid')  # Get neuron ID
    weights = json.loads(row.get('weights_before', '[]'))  # Deserialize weights
    inputs = json.loads(row.get('inputs', '[]'))  # Deserialize inputs

    if len(weights) != len(inputs):
        raise ValueError(f"Mismatch in length of weights ({len(weights)}) and inputs ({len(inputs)})")

    predictions = []
    for i, (w, inp) in enumerate(zip(weights, inputs), start=1):
        label = f"N{nid}W{i}:"
        calculation = f"{label} {w} * {inp} = {w * inp}"
        predictions.append(calculation)

    return "\n".join(predictions)

    #db.list_tables(2)
    """
    Generate the step detail report using tabulate.
    
    # Fetch the data from the database
    formatted_rows = db.query("SELECT * FROM IterationData")
    print(f"Formatted Rows: {formatted_rows}")

    # Use "keys" for headers to match dictionary structure
    headers = "keys"

    # Generate the tabulated report
    report = tabulate(formatted_rows, headers=headers, tablefmt="fancy_grid")
    print(report)
    """

def fetch_report_data(ramDb: sqlite3.Connection):
    """
    Fetch data from the database for generating the step detail report.
    epoch, step, prediction, target, error, neuron_id, weights_before, weights_after, bias_before, bias_after, output_before, output_after = row
    """
    sql = '''
        SELECT
            i.epoch,
            i.step,
            i.prediction,
            i.target,
            i.target - i.prediction AS error,
            n.neuron_id,
            n.weights,
            n.bias,
            n.output
        FROM iterations AS i
        JOIN neurons AS n
        ON i.model_id = n.model_id AND i.epoch = n.epoch AND i.step = n.step
        ORDER BY i.epoch, i.step, n.neuron_id
    '''

    return ramDb.execute(sql).fetchall()



def format_report_data(report_data):
    """
    Format the fetched report data for tabulate.
    """
    formatted_rows = []
    for row in report_data:
        epoch, step, prediction, target, error, neuron_id, weights_before, weights_after, bias_before, bias_after, output_before, output_after = row

        # Deserialize weights
        weights_before = json.loads(weights_before)
        weights_after = json.loads(weights_after)

        # Generate weight * input + bias strings
        weight_input_bias = [
            f"N{neuron_id}W{i+1} {w_before} -> {w_after}"
            for i, (w_before, w_after) in enumerate(zip(weights_before, weights_after))
        ]

        # Add formatted row
        formatted_rows.append([
            f"Epoch {epoch} / Step {step}",
            "\n".join(weight_input_bias),
            prediction,
            target,
            error,
            f"{bias_before} -> {bias_after}",
            f"{output_before} -> {output_after}"
        ])
    return formatted_rows


def format_report_dataOrigi(report_data):
    """
    Format the fetched report data for tabulate.
    """
    formatted_rows = []
    for row in report_data:
        epoch, step, prediction, target, error, neuron_id, weights, bias, output = row

        # Deserialize weights (assuming JSON storage)
        weights = json.loads(weights)

        # Generate weight * input + bias string
        weight_input_bias = [
            f"N{neuron_id}W{i+1} {w} * X + {bias}"
            for i, w in enumerate(weights)
        ]

        formatted_rows.append([
            f"Epoch {epoch} / Step {step}",
            "\n".join(weight_input_bias),
            prediction,
            target,
            error,
            f"Neuron {neuron_id} Output: {output}"
        ])
    return formatted_rows

