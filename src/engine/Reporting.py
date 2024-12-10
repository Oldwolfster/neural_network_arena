import math

from tabulate import tabulate
from typing import List
from .MetricsMgr import MetricsMgr
from .Utils import *
from src.ArenaSettings import HyperParameters
from src.engine.graphs._GraphMaster import graph_master

repeat_header_interval = 10
MAX_ITERATION_LINES = 200
def print_results(mgr_list : List[MetricsMgr], training_data, hyper : HyperParameters, training_pit):
    problem_type = determine_problem_type(training_data)
    print_iteration_logs(mgr_list, hyper)
    print_epoch_summaries(mgr_list, problem_type)
    print_convergence(mgr_list)
    print(f"Data: {training_pit}\tTraining Set Size: {hyper.training_set_size}\t Max Epochs: {hyper.epochs_to_run}\tDefault Learning Rate: {hyper.default_learning_rate}")
    print_model_comparison(mgr_list, problem_type)
    if hyper.display_train_data:
        print (training_data)
    if hyper.display_graphs:
        graph_master(mgr_list)

######################################################################################
############# First level of reporting, iteration details ############################
######################################################################################
def print_iteration_logs(mgr_list: List[MetricsMgr], hyper : HyperParameters):
    if not hyper.display_logs:
        return
    headers = ["Step Detail", "Epoc\nStep",  "Weight * Input + Bias", "Prediction","Target", "Error", "New Weight/Chg",  "New\nBias"]
    row_counter = 0  # Keep track of the number of rows printed
    max_rows = hyper.epochs_to_run * hyper.training_set_size  # Each MetricsMgr can have this many rows.

    # Create a list of count of iteration and name for each manager's metrics
    lengths = [len(mgr.metrics) for mgr in mgr_list]
    model_names = [mgr.name for mgr in mgr_list]

    repeat_header_interval = 10
    correlated_log = []                                     # Create new list to merge and format data
    buffer = []  # Initialize an empty list to hold the output


    # Create correlate log (i.e. epoch 1 for each model, epoch 2 for each model, etc.)
    for row in range(max_rows):                             # Loop through each iteration
        #if row_counter > MAX_ITERATION_LINES:                              # Exit the loop if row_counter exceeds 2000

        for mgr_idx, mgr in enumerate(mgr_list):            # Loop through each Model's Metrics Mgr
            if row < lengths[mgr_idx]:
                # Instead of adding headers in body we will print many tabulates add_extra_headers(correlated_log, row_counter, repeat_header_interval, headers)
                append_iteration_row(correlated_log, model_names[mgr_idx], mgr.metrics[row], hyper)
                row_counter += 1

    chunks = list(chunk_list(correlated_log, repeat_header_interval))  # Split correlated_log into sublists of length repeat_header_interval
    for chunk in chunks:  # Prepare each chunk for printing
        buffer.append(tabulate(chunk, headers=headers, tablefmt="fancy_grid"))
        #print(tabulate(chunk, headers=headers, tablefmt="fancy_grid"))

    # Join all chunks into a single string with a newline separator and print once
    print("\n\n".join(buffer))


def print_iteration_logs20241208(mgr_list: List[MetricsMgr], hyper : HyperParameters):
    if not hyper.display_logs:
        return
    headers = ["Step Detail", "Epoc\nStep",  "Weight * Input + Bias", "Prediction","Target", "Error", "New Weight/Chg",  "New\nBias"]
    row_counter = 0  # Keep track of the number of rows printed
    max_rows = hyper.epochs_to_run * hyper.training_set_size  # Each MetricsMgr can have this many rows.

    # Create a list of count of iteration and name for each manager's metrics
    lengths = [len(mgr.metrics) for mgr in mgr_list]
    model_names = [mgr.name for mgr in mgr_list]

    repeat_header_interval = 10
    correlated_log = []                                     # Create new list to merge and format data
    buffer = []  # Initialize an empty list to hold the output
    for row in range(max_rows):                             # Loop through each iteration
        #if row_counter > MAX_ITERATION_LINES:                              # Exit the loop if row_counter exceeds 2000

        for mgr_idx, mgr in enumerate(mgr_list):            # Loop through each Model's Metrics Mgr
            if row < lengths[mgr_idx]:
                # Instead of adding headers in body we will print many tabulates add_extra_headers(correlated_log, row_counter, repeat_header_interval, headers)
                append_iteration_row(correlated_log, model_names[mgr_idx], mgr.metrics[row], hyper)
                row_counter += 1


        chunks = list(chunk_list(correlated_log, repeat_header_interval))  # Split correlated_log into sublists of length repeat_header_interval
        for chunk in chunks:  # Prepare each chunk for printing
            #buffer.append(tabulate(chunk, headers=headers, tablefmt="fancy_grid"))
            print(tabulate(chunk, headers=headers, tablefmt="fancy_grid"))

    # Join all chunks into a single string with a newline separator and print once

def append_iteration_row(correlated_log: list, model_name: str, data_row, hyper : HyperParameters,):
    data        = data_row.to_list()
    epoch       = data[0]
    inputs      = data[2]
    weights     = data[10]
    new_weights = data[11]
    bias        = data[12]
    new_bias    = data[13]

    if epoch < hyper.detail_log_min:
        return
    if epoch > hyper.detail_log_max:
        return


    # Concatenate each input-weight operation into a single cell with line breaks
    #weighted_terms = "\n".join(
        # originial f"{smart_format(weights[i])} * {smart_format(inputs[i])} = {smart_format(weights[i] * inputs[i])}"
#        f"{smart_format(weights[i])} * {smart_format(inputs[i])} + {smart_format(data[12])}"
#        for i in range(len(inputs))
#    )
    weighted_terms = "\n".join(
        f"{smart_format(weights[i])} * {smart_format(inputs[i])} + {smart_format(data[12]) if i == 0 else '0'}"
        for i, _ in enumerate(inputs)
    )

    # Concatenate new weights and amount of change into a single cell with line breaks
    new_weights = "\n".join(
        f"{smart_format(new_weights[i])} / ({'+' if (new_weights[i] - weights[i]) > 0 else ''}{smart_format(new_weights[i] - weights[i])})"
        for i in range(len(inputs))
    )
    new_bias = f"{smart_format(new_bias)} / ({'+' if (new_bias - bias) > 0 else ''}{smart_format(new_bias - bias)})"
    # TODO Add columnt for Activation function

    # Append a single row for this iteration, with concatenated input-weight terms
    correlated_log.append([
        model_name,
        f"{epoch} / {data[1]}",  # Epoch and iteration #
        weighted_terms,             # Concatenated Weighted Term calculations
        f"{smart_format(data[4])}", # Prediction
        smart_format(data[3]),      # Target
        smart_format(data[5]),       # Error
        new_weights,                # New Weight
        new_bias     # New Bias
    ])

######################################################################################
############# Display Convergence ###############################
######################################################################################

def print_convergence(mgr_list : List[MetricsMgr]):
    for mgr in mgr_list:
        if len(mgr.convergence_signal)==0:
            print(f"Convergence - Model:{mgr.name}\tDID NOT CONVERGE - HIT MAX EPOCHS")
        else:
            print(f"Convergence - Model:{mgr.name}\t{mgr.convergence_signal}")



######################################################################################
############# Second level of reporting, epoch details ###############################
######################################################################################

def print_epoch_summaries(mgr_list : List[MetricsMgr],problem_type):
    all_summaries = []
    for mgr in mgr_list:
        add_summary(mgr, all_summaries, problem_type)
    headers = ["Epoch Summary", "Epoch", "Final\nWeight", "Final\nBias","Correct", "Wrong", "Accuracy", "Mean\nAbs Err", "Mean\nSqr Err"]

    if problem_type == "Binary Decision":
        headers.extend(["TP", "TN", "FP", "FN","Prec\nision", "Recall", "F1\nScore", "Specif\nicity"])
    else: # Regression adds RMSE
        headers.extend(["RMSE"])

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
        #smart_format(summary.final_weight[0]),  #TODO handle multiple weights
        # Concatenate final weights and their changes into a single cell with line breaks
        "\n".join(
            f"{smart_format(summary.final_weight[i])}"
            for i in range(len(summary.final_weight))
        ),
        smart_format(summary.final_bias),
        summary.tp + summary.tn,
        summary.fp + summary.fn,
        f"{smart_format((summary.tp + summary.tn)/summary.total_samples * 100)}%",
        smart_format(summary.total_absolute_error/summary.total_samples),
        smart_format(summary.total_squared_error/summary.total_samples)
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
    else: # Regression

        epoch_summary.extend([smart_format(math.sqrt( summary.total_squared_error))])


    #print(epoch_summary)
    epoch_summaries.append(epoch_summary)


######################################################################################
############# Third level of reporting, Model Comparision#############################
######################################################################################


def collect_final_epoch_summary_ONLYONEWEIGHT(mgr_list: List[MetricsMgr], problem_type: str) -> List:
    final_summaries = []
    for mgr in mgr_list:
        if mgr.epoch_summaries:  # Check if there are epoch summaries available
            last_summary = mgr.epoch_summaries[-1]  # Get the last epoch summary

            # Create a list starting with model name and runtime
            summary_row = [last_summary.model_name, smart_format(mgr.run_time)]

            # Explicitly slice and order the fields we need (using dot notation for clarity)
            summary_fields = [
                last_summary.epoch,
                last_summary.final_weight[0],
                last_summary.final_bias,
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
                #last_summary.final_weight[0],
                "\n".join(
                    f"{smart_format(last_summary.final_weight[i])}"
                    for i in range(len(last_summary.final_weight))
                ),
                last_summary.final_bias,
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



def print_model_comparison(mgr_list : List[MetricsMgr],problem_type):
    # Collect and print final epoch summaries
    final_summaries = collect_final_epoch_summary(mgr_list, problem_type)
    headers = ["Gladiator Comparision", "Run Time", "Epoch\nof Conv", "Final\nWeight", "Final\nBias", "Total\nError", "Correct", "Wrong", "Accuracy", "Mean\nAbs Err"]
    if problem_type == "Binary Decision":
        headers.extend(["TP", "TN", "FP", "FN", "Precision", "Recall", "F1 Score", "Specificity"])
    else: # Regression
        headers.extend(["RMSE"])
#    print(f"Training Set Size: {iteration_count}\t Max Epochs: {epoch_count}\tDefault Learning Rate: {default_learning_rate}")
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






