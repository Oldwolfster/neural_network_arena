import json
import sqlite3

from tabulate import tabulate

from src.engine.RamDB import RamDB


def generate_reports(db : RamDB):
    generate_iteration_report(db)


def generate_iteration_report(db: RamDB):
    db.query_print("SELECT * FROM IterationData")
    db.query_print("Select * From Neuron")
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
