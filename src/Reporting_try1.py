from tabulate import tabulate
from Graphing import *
from colorama import Fore, Style
from typing import List, Dict
from MetricsMgr import MetricsMgr, get_all_epoch_summaries
from Utils import *

iteration_log = []

def print_results(mgr_list : List[MetricsMgr], training_data, display_graphs, display_iteration_logs, display_train_data, display_epoch_sum, epoch_count : int, iteration_count: int):
    problem_type = determine_problem_type(training_data)
    prepare_data(mgr_list)
    if display_iteration_logs:
        print_logs(mgr_list, epoch_count, iteration_count)
    print_epoch_summaries(mgr_list, problem_type)

    if display_train_data:
        print(f"Training Data: {training_data}")

def prepare_data(mgr_list : List[MetricsMgr], epoch_count: int, iteration_count: int):
    max_rows = epoch_count * iteration_count  # Each MetricsMgr can have this many rows.

    # Create a list of count of iteration and name for each manager's metrics
    lengths = [len(mgr.metrics) for mgr in mgr_list]
    model_names = [mgr.name for mgr in mgr_list]

    iteration_log = []
    for row in range(max_rows):                             # Loop through each iteration
        for mgr_idx, mgr in enumerate(mgr_list):            # Loop through each model
            append_iteration_row(iteration_log, model_names[mgr_idx], mgr.metrics[row])


def append_iteration_row(iteration_log: list, model_name: str, data_row):
    data = data_row.to_list()
    iteration_log.append([
        model_name,
        data[0],  # Epoch
        data[1],  # Iteration
        smart_format(data[2]),  # Input
        smart_format(data[3]),  # Target
        smart_format(data[4]),  # Prediction
        smart_format(data[5]),  # Error
        smart_format(data[10]),  # Old Weight
        smart_format(data[11]),  # New Weight
        smart_format(data[12]),  # Old Bias
        smart_format(data[13]),  # New Bias
    ])


def print_logs(mgr_list: List[MetricsMgr], epoch_count: int, iteration_count: int):
    headers = ["Model", "Epoch", "Iter#", "Input", "Target", "Prediction", "Error", "Old Weight", "New Weight", "Old Bias", "New Bias"]
    repeat_header_interval = 10
    row_counter = 0  # Keep track of the number of rows printed
    max_rows = epoch_count * iteration_count  # Each MetricsMgr can have this many rows.

    # Create a list of count of iteration and name for each manager's metrics
    lengths = [len(mgr.metrics) for mgr in mgr_list]
    model_names = [mgr.name for mgr in mgr_list]

    correlated_log = []                                     # Create new list to merge and format data
    for row in range(max_rows):                             # Loop through each iteration
        for mgr_idx, mgr in enumerate(mgr_list):            # Loop through each model
            if row < lengths[mgr_idx]:
                add_extra_headers(correlated_log, row_counter, repeat_header_interval, headers)
                append_data_row(correlated_log, model_names[mgr_idx], mgr.metrics[row])
                row_counter += 1
    print(tabulate(correlated_log, headers=headers, tablefmt="fancy_grid"))


def add_extra_headers(correlated_log: list, row_counter: int, repeat_header_interval: int, headers: list):
    if row_counter % repeat_header_interval == 0 and row_counter > 0:
        correlated_log.append(headers)


def append_data_row(correlated_log: list, model_name: str, data_row):
    data = data_row.to_list()
    correlated_log.append([
        model_name,
        data[0],  # Epoch
        data[1],  # Iteration
        smart_format(data[2]),  # Input
        smart_format(data[3]),  # Target
        smart_format(data[4]),  # Prediction
        smart_format(data[5]),  # Error
        smart_format(data[10]),  # Old Weight
        smart_format(data[11]),  # New Weight
        smart_format(data[12]),  # Old Bias
        smart_format(data[13]),  # New Bias
    ])


#def print_epoch_summaries(all_summaries: Dict[str, List[EpochSummary]]):
def print_epoch_summaries(mgr_list: List[MetricsMgr]):
    all_summaries = get_all_epoch_summaries(mgr_list)
    for model_name, summaries in all_summaries.items():
        print(f"\nEpoch Summaries for {model_name}:")
        headers = ["Epoch", "Total Samples", "Correct", "Accuracy", "MAE", "MSE"]
        table_data = []
        for summary in summaries:
            accuracy = summary.correct_predictions / summary.total_samples
            mae = summary.total_absolute_error / summary.total_samples
            mse = summary.total_squared_error / summary.total_samples
            table_data.append([
                summary.epoch,
                summary.total_samples,
                summary.correct_predictions,
                f"{accuracy:.2%}",
                f"{mae:.4f}",
                f"{mse:.4f}"
            ])
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

"""
            # Check if row counter has reached the repeat interval
            if row_counter % repeat_header_interval == 0 and row_counter > 0:
                correlated_log.append(headers)

            if row < lengths[mgr_idx]:  # Use precomputed length
                data_row = mgr.metrics[row].to_list()  # Get iteration data from mgr
                correlated_log.append([
                    model_names[mgr_idx],
                    data_row[0],
                    data_row[1],
                    smart_format(data_row[2]),
                    smart_format(data_row[3]),
                    smart_format(data_row[4]),
                    smart_format(data_row[5]),
                    smart_format(data_row[10]),  # Old Weight
                    smart_format(data_row[11]),
                    smart_format(data_row[12]),
                    smart_format(data_row[13]),
                ])
                row_counter += 1  # Increment the counter after each data row

    # Print the final result
    print(tabulate(correlated_log, headers=headers, tablefmt="fancy_grid"))
    """




