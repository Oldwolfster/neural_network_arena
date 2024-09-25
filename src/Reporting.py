from tabulate import tabulate
from Graphing import *
from colorama import Fore, Style
from typing import List
from MetricsMgr import MetricsMgr
from Utils import *


def print_results(mgr_list : List[MetricsMgr], training_data, display_graphs, display_logs, display_train_data, display_epoch_sum, epoch_count : int, iteration_count: int):
    problem_type = determine_problem_type(training_data)

    if display_logs:
        print_logs(mgr_list, epoch_count, iteration_count)

    if display_train_data:
        print(f"Training Data: {training_data}")



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




def determine_problem_type(data):
    """
    Examine training data to determine if it's binary decision or regression
    """
    # Extract unique values from the second element of each tuple
    unique_values = set(item[1] for item in data)

    # If there are only two unique values, it's likely a binary decision problem
    if len(unique_values) == 2:
        return "Binary Decision"

    # If there are more than two unique values, it's likely a regression problem
    elif len(unique_values) > 2:
        return "Regression"

    # If there's only one unique value or the list is empty, it's inconclusive
    else:
        return "Inconclusive"
