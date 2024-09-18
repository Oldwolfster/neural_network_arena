from tabulate import tabulate
from Graphing import *
from colorama import Fore, Style

logs_to_print = -1      # It will print this many of the first and this many of the last iteration logs


def give_me_a_line():
    print(f"{'=' * 129}")

def print_results(metrics_list, training_data, display_graphs, display_logs, display_train_data,epochs_to_run, training_set_size):
    problem_type = determine_problem_type(training_data)

    if display_logs == True:
         print_logs(metrics_list)

    print_epoch_summary(metrics_list)

    if problem_type == "Binary Decision":
        print_binary_grid(metrics_list,epochs_to_run, training_set_size)
    else:
        print_regression_grid(metrics_list, epochs_to_run, training_set_size)

    if display_train_data:
        print(f"Training Data: {training_data}")
    if display_graphs == True:
        for metrics in metrics_list:
            all_graphs(metrics)


def print_epoch_summary(metrics_list, repeat_header_interval=8):
    # Define headers
    headers = ["epoch", "PAP", "MAPE", "R^2", "Mean Absolute Error", " Total Absolute Error", "Mean Square Error", "Final Weight", "Final Bias"]

    # Prepare data
    data = []

    row_counter = 0  # To track when to insert the headers

    # Iterate over each metrics object in metrics_list
    for metrics in metrics_list:
        # Iterate over each epoch summary in the metrics object
        print(f"\n\nEpoch Summary for {metrics.name}")
        for epoch_summary in metrics.epoch_summaries:
            epochNum, iteration_count, total_pap, MAPE, R2, TAE, MAE, MSE, weight, bias = epoch_summary

            # Insert header row every 'repeat_header_interval' rows
            if row_counter % repeat_header_interval == 0:
                data.append(headers)

            # Append formatted row data
            data.append([
                epochNum,
                f"{(total_pap / iteration_count):,.0f}%",  # Prediction Accuarcy Percent
                f"{MAPE:,.6f}",  # Mean Absolute Percentage Error
                f"{R2:,.6f}",  # R squared
                f"{MAE:,.6f}",         # Mean Absolute Error
                f"{TAE:,.6f}",  # Format Total Absolute Error as a nicely readable integer
                f"{MSE:,.6f}",         # Mean squuared error
                f"{weight:.6f}",      # Format weight to 4 decimal places
                f"{bias:.6f}" if bias is not None else "N/A",  # Handle potential None for bias

            ])
            row_counter += 1
            # Print table using the 'tabulate' module

        print(tabulate(data, headers="firstrow", tablefmt="grid"))
        data.clear()
        row_counter = 0

def print_regression_grid(metrics_list,epochs_to_run, training_set_size):
    # Prepare headers
    print(f"\nEpochs:{epochs_to_run}\tTraining Samples: {training_set_size}")
    headers = ["Neural Network Arena", "Run Time", "Conv at", "PAP", "Mean Abs Err", " Total Abs Err", "Mean Sqr Err", "Error Squared"]

    # Prepare data
    data = []
    total_loss_for_epoch = []    # MAE  Mean Absolute Error - sum the abs value of the loss....
    for metrics in metrics_list:
        # Calculate metrics


        pap = f"{(metrics.total_pap / metrics.iteration_count):,.0f}%"
        errors = f"{metrics.errors[-1]:,.2f}"
        errors_squared = f"{metrics.errors_squared[-1]:,.2f}"
        mse = f"{(metrics.errors_squared[-1] / metrics.iteration_count):,.0f}"
        iteration_count = metrics.iteration_count
        recall = .69
        f1_score = .69

        # Append row data
        data.append([
            metrics.name,
            f"{metrics.run_time:.2f}",
            f"{metrics.converge_epoch:.0f}",
            pap,
            f"{metrics.converge_mae_final:,.2f}",
            errors,
            mse,
            errors_squared,
        ])

    # Print table
    print(tabulate(data, headers=headers, tablefmt="grid"))
    give_me_a_line()
    # for name, losses in total_loss_for_epoch:
    #   print(f"Total Loss By Epoch for {name}: {losses}")


def print_binary_grid(metrics_list,epochs_to_run, training_set_size):
    # Prepare headers
    print(f"\nEpochs:{epochs_to_run}\tTraining Samples: {training_set_size}")
    headers = ["Neural Network Arena", "Run Time", "Correct", "Wrong/Loss", "Accuracy", "Precision", "Recall", "F1 Score", "Epoch to Converge", "SAE at Conv"]

    # Prepare data
    data = []
    total_loss_for_epoch = []    # MAE  Mean Absolute Error - sum the abs value of the loss....
    for metrics in metrics_list:
        # Calculate metrics
        accuracy = metrics.accuracy * 100
        precision = metrics.precision * 100
        recall = metrics.recall * 100
        f1_score = metrics.f1_score * 100
        total_loss_for_epoch.append((metrics.name, metrics.losses))
        # Append row data
        data.append([
            metrics.name,
            f"{metrics.run_time:.2f}",
            metrics.correct,
            metrics.wrong,
            # f"{metrics.total_loss_for_epoch:.0f}",
            f"{accuracy:.2f}%",
            f"{precision:.2f}%",
            f"{recall:.2f}%",
            f"{f1_score:.2f}%",
            metrics.epochs_to_converge,
            metrics.MAE_at_convergence
        ])

    # Print table
    print(tabulate(data, headers=headers, tablefmt="grid"))
    give_me_a_line()
    # for name, losses in total_loss_for_epoch:
    #   print(f"Total Loss By Epoch for {name}: {losses}")



# Define color codes
colors = [
    "\033[97m",  # White
    "\033[94m",  # Blue
    "\033[91m",  # Red
    "\033[92m",  # Green
    "\033[93m",  # Yellow

    "\033[95m",  # Magenta
    "\033[96m",  # Cyan

]

# Function to reset color
reset_color = "\033[0m"

def print_logsAlmost(metrics_list):
    headers = ["Epoch", "Iteration", "Input", "Target", "Prediction", "PAP", "Error", "Old Weight", "New Weight",
               "Model"]
    all_data = []

    for idx, metrics in enumerate(metrics_list):
        color = colors[idx % len(colors)]  # Cycle through colors if more metrics than colors
        for row in metrics.log_data:
            colored_row = [f"{color}{cell}{Style.RESET_ALL}" for cell in row]
            colored_row.append(f"{color}{metrics.name}{Style.RESET_ALL}")
            all_data.append(colored_row)

    print(tabulate(all_data, headers=headers, tablefmt="grid"))


from itertools import zip_longest
from tabulate import tabulate
from colorama import Style  # Assuming you're using colorama for color resets

from itertools import zip_longest
from tabulate import tabulate
from colorama import Style  # Assuming you're using colorama for color resets

from itertools import zip_longest
from tabulate import tabulate
from colorama import Style  # Assuming you're using colorama for color resets


def print_logs(metrics_list):
    headers = ["Epoch", "Iteration", "Input", "Target", "Prediction", "PAP", "Error", "Old Weight", "New Weight", "Old Bias", "New Bias",
               "Model"]
    row_counter = 0  # Keep track of the number of rows printed

    # Use zip_longest to iterate over rows in parallel from each metrics list
    max_length = max(len(metrics.log_data) for metrics in metrics_list)

    # Initialize an empty list to hold rows between header reprints
    buffer = []

    for rows in zip_longest(*[metrics.log_data for metrics in metrics_list], fillvalue=None):
        if row_counter % 10 == 0 and buffer:
            # Print accumulated rows with headers
            print(tabulate(buffer, headers=headers, tablefmt="grid"))
            buffer = []  # Clear buffer after printing

        for idx, row in enumerate(rows):
            if row:  # If the row exists (zip_longest fills with None if shorter lists)
                color = colors[idx % len(colors)]  # Cycle through colors if more metrics than colors
                colored_row = [f"{color}{cell}{Style.RESET_ALL}" for cell in row]
                colored_row.append(f"{color}{metrics_list[idx].name}{Style.RESET_ALL}")
                buffer.append(colored_row)

        row_counter += 1

    # Print any remaining rows with headers
    if buffer:
        print(tabulate(buffer, headers=headers, tablefmt="grid"))


def print_logsLastTextVersion(metrics_list):
    # Merge all log lists into a single list
    merged_logs = []
    for idx, metrics in enumerate(metrics_list):
        color = colors[idx % len(colors)]  # Assign color based on model index
        for log_entry in metrics.log_list:
            # Store the log entry, the model name, and color in a tuple
            # log_entry is assumed to be a list where epoch and iteration are the first two elements
            merged_logs.append((log_entry, metrics.name, color))

    # Sort the merged logs by epoch (first element) and iteration (second element)
    sorted_logs = sorted(merged_logs, key=lambda x: (x[0][0], x[0][1]))

    # Print the sorted logs with color-coding
    for log_entry, model_name, color in sorted_logs:
        # Format the row with appropriate coloring and spacing
        formatted_row = "\t".join([f"{item}" for item in log_entry])
        print(f"{color}{formatted_row}\t{model_name}{reset_color}")




def print_logsLastTextVersion(metrics_list):
    max_length = max(len(metrics.log_list) for metrics in metrics_list)
    for i in range(max_length):
        for idx, metrics in enumerate(metrics_list):
            if i < len(metrics.log_list):
                color = colors[idx % len(colors)]  # Cycle through colors if more metrics than colors
                print(f"{color}{metrics.log_list[i]}\t{metrics.name}{reset_color}")




def print_footer(training_data, metrics_list):
    print(training_data)

    for metrics in metrics_list:
        mae_for_who = metrics.name
        calculate_anomaly_rate(training_data, metrics)
        anomalies = metrics.anomalies
        anom_percent = anomalies / metrics.data_count * 100

        print(f"{mae_for_who}  # Anomalies:\t{anomalies:.0f} out of {metrics.data_count:.0f}\t {anom_percent}%")
        error_percent = 100-metrics.percents[-1]
        if anomalies > 0:
            error_rate_per_anomaly = error_percent / anomalies
            print(f"Error_rate per Anomaly:\t{error_rate_per_anomaly:.0f}%) # \tAnomalies:{anomalies}\tError Rate Per Anomaly:{error_rate_per_anomaly:.2f}")
        else:
            print("No anomalies!!!")

    #if show_graph == 1:
    #    plot_loss_and_weight_stability(metrics)


def determine_problem_type(data):
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


def calculate_anomaly_rate(training_data, metrics):
    count = 0
    anoms = 0
    for i, (credit_score, result) in enumerate(training_data):
        count += 1
        if is_it_an_anomaly(credit_score, result):
            anoms += 1

    metrics.anomalies = anoms
    metrics.data_count = count


def is_it_an_anomaly(credit_score, result):
    if credit_score >= 50 and result == 0:
        return True
    return False


