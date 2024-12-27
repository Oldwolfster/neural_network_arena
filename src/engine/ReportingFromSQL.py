import json
import sqlite3

from tabulate import tabulate

def generate_reports(ramDb: sqlite3.Connection):
    generate_iteration_report(ramDb)


def generate_iteration_report(ramDb: sqlite3.Connection):
    """
    Generate the step detail report using tabulate.
    """
    report_data = fetch_report_data(ramDb)
    formatted_rows = format_report_data(report_data)

    headers = [
        "Step Detail",
        "Weight * Input + Bias",
        "Prediction",
        "Target",
        "Error",
        "Neuron Output"
    ]

    report = tabulate(formatted_rows, headers=headers, tablefmt="fancy_grid")
    print(report)


def fetch_report_data(ramDb: sqlite3.Connection):
    """
    Fetch data from the database for generating the step detail report.
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

