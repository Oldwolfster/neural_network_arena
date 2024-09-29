from tabulate import tabulate
from Graphing import *
from colorama import Fore, Style
from typing import List, Dict
from MetricsMgr import MetricsMgr
from Utils import *
repeat_header_interval = 10

def print_results(mgr_list : List[MetricsMgr], training_data, display_graphs, display_iteration_logs, display_train_data, display_epoch_sum, epoch_count : int, iteration_count: int):
    problem_type = determine_problem_type(training_data)
    print_iteration_logs(mgr_list, epoch_count, iteration_count, display_iteration_logs)
    print_epoch_summaries(mgr_list, problem_type)
    print_model_comparison(mgr_list, problem_type, epoch_count, iteration_count)
    if display_train_data:
        print (training_data)

######################################################################################
############# First level of reporting, iteration details ############################
######################################################################################
def print_iteration_logs(mgr_list: List[MetricsMgr], epoch_count: int, iteration_count: int, display_iteration_logs):
    if not display_iteration_logs:
        return
    headers = ["Iteration Summary", "Epoch", "Iter#", "Input", "Target", "Prediction", "Error", "Old Weight", "New Weight", "Old Bias", "New Bias"]
    row_counter = 0  # Keep track of the number of rows printed
    max_rows = epoch_count * iteration_count  # Each MetricsMgr can have this many rows.

    # Create a list of count of iteration and name for each manager's metrics
    lengths = [len(mgr.metrics) for mgr in mgr_list]
    model_names = [mgr.name for mgr in mgr_list]

    repeat_header_interval = 10
    correlated_log = []                                     # Create new list to merge and format data
    for row in range(max_rows):                             # Loop through each iteration
        for mgr_idx, mgr in enumerate(mgr_list):            # Loop through each model
            if row < lengths[mgr_idx]:
                # Instead of adding headers in body we will print many tabulates add_extra_headers(correlated_log, row_counter, repeat_header_interval, headers)
                append_iteration_row(correlated_log, model_names[mgr_idx], mgr.metrics[row])
                row_counter += 1

    chunks = list(chunk_list(correlated_log, repeat_header_interval))    # Split correlated_log into sublists of length repeat_header_interval
    for chunk in chunks:                                                        # Print each chunk using tabulate
        print(tabulate(chunk, headers=headers, tablefmt="fancy_grid"))


def append_iteration_row(correlated_log: list, model_name: str, data_row):
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

######################################################################################
############# Second level of reporting, epoch details ###############################
######################################################################################

def print_epoch_summaries(mgr_list : List[MetricsMgr],problem_type):
    all_summaries = []
    for mgr in mgr_list:
        add_summary(mgr, all_summaries, problem_type)
    headers = ["Epoch Summary", "Epoch", "Final\nWeight","Total\nError", "Correct", "Wrong", "Accuracy","Mean\nAbs Err"]

    if problem_type == "Binary Decision":
        headers.extend(["TP", "TN", "FP", "FN","Prec\nision", "Recall", "F1\nScore", "Specif\nicity"])

    chunks = list(chunk_list(all_summaries, repeat_header_interval))    # Split correlated_log into sublists of length repeat_header_interval
    for chunk in chunks:                                                        # Print each chunk using tabulate
        print(tabulate(chunk, headers=headers, tablefmt="fancy_grid"))


def add_summary(mgr: MetricsMgr, epoch_summaries : list, problem_type):
    for summary in mgr.epoch_summaries:
        append_summary_row(summary, epoch_summaries, problem_type)


def append_summary_row(summary, epoch_summaries : list, problem_type):
    epoch_summary = [
        summary.model_name,
        summary.epoch,
        smart_format(summary.final_weight),
        smart_format(summary.total_absolute_error),
        summary.tp + summary.tn,
        summary.fp + summary.fn,
        f"{smart_format((summary.tp + summary.tn)/summary.total_samples * 100)}%",
        smart_format(summary.total_absolute_error/summary.total_samples)
    ]

    if problem_type == "Binary Decision":
        epoch_summary.extend([
            summary.tp,
            summary.tn,
            summary.fp,
            summary.fn,
            smart_format(precision(summary.tp, summary.fp)),
            smart_format(recall(summary.tp, summary.fn)),
            smart_format(f1(summary.tp, summary.fp, summary.fn)),
            smart_format(specificity(summary.tn, summary.fp))

    ])
    #print(epoch_summary)
    epoch_summaries.append(epoch_summary)

def collect_final_epoch_summary(mgr_list: List[MetricsMgr], problem_type: str) -> List:
    final_summaries = []
    for mgr in mgr_list:
        if mgr.epoch_summaries:  # Check if there are epoch summaries available
            last_summary = mgr.epoch_summaries[-1]  # Get the last epoch summary

            # Create a list starting with model name and runtime
            summary_row = [last_summary.model_name, smart_format(mgr.run_time)]

            # Explicitly slice and order the fields we need (using dot notation for clarity)
            summary_fields = [
                last_summary.epoch,
                last_summary.final_weight,
                last_summary.total_absolute_error,
                last_summary.tp + last_summary.tn,  # Correct
                last_summary.fp + last_summary.fn,  # Wrong
                f"{smart_format((last_summary.tp + last_summary.tn) / last_summary.total_samples * 100)}%",  # Accuracy
                smart_format(last_summary.total_absolute_error / last_summary.total_samples)  # Mean Abs Error
            ]

            # Append the sliced fields to the summary_row
            summary_row.extend(summary_fields)

            final_summaries.append(summary_row)
    return final_summaries




def collect_final_epoch_summaryorig(mgr_list: List[MetricsMgr], problem_type: str) -> List:
    final_summaries = []
    for mgr in mgr_list:
        if mgr.epoch_summaries:  # Check if there are epoch summaries available
            last_summary = mgr.epoch_summaries[-1]  # Get the last epoch summary
            append_summary_row(last_summary, final_summaries, problem_type)
    return final_summaries

######################################################################################
############# Third level of reporting, Model Comparision#############################
######################################################################################

def print_model_comparison(mgr_list : List[MetricsMgr],problem_type ,  epoch_count, iteration_count):
    # Collect and print final epoch summaries
    final_summaries = collect_final_epoch_summary(mgr_list, problem_type)
    headers = ["Model Summary", "Run Time", "Epc 2 Convg", "Final Weight", "Total Error", "Correct", "Wrong", "Accuracy", "Mean Abs Err"]
    if problem_type == "Binary Decision":
        headers.extend(["TP", "TN", "FP", "FN", "Precision", "Recall", "F1 Score", "Specificity"])
    print(f"Training Set Size: {iteration_count}\t Max Epochs:{epoch_count}")
    print(tabulate(final_summaries, headers=headers, tablefmt="fancy_grid"))



def precision(tp, fp) -> float:
    """how many of the predicted positive instances are actually positive."""
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(tp, fn) -> float:
    """how many of the actual positive instances are correctly predicted."""
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1(tp, fp, fn) -> float:
    """harmonic mean of precision and recall"""
    prec = precision(tp, fp)
    rec  = recall(tp, fn)
    if (prec + rec) > 0:
        return 2 * (prec * rec) / (prec + rec)
    return 0

def specificity(tn, fp) -> float:
    """how many of the actual negatives are correctly predicted"""
    return tn / (tn + fp) if (tn + fp) > 0 else 0






