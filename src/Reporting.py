from tabulate import tabulate
from Graphing import *
from colorama import Fore, Style

logs_to_print = -1      # It will print this many of the first and this many of the last iteration logs


def give_me_a_line():
    print(f"{'=' * 129}")


def print_results(mgr_list, training_data, display_graphs, display_logs, display_train_data, display_epoch_sum, epochs_to_run, training_set_size):
    problem_type = determine_problem_type(training_data)

    if display_logs == True:
         print_logs(mgr_list)

    if display_epoch_sum == True:
        print_epoch_summary(mgr_list)

    print(f"\nTraining Run Summary:\tProblem Type: {problem_type}\tEpochs:{epochs_to_run}\tTraining Samples: {training_set_size}")
    # if problem_type == "Binary Decision":
    print_binary_summary(mgr_list,epochs_to_run, training_set_size)
    # else:
    give_me_a_line()
    print_regression_summary(mgr_list, epochs_to_run, training_set_size)

    if display_train_data:
        print(f"Training Data: {training_data}")
    if display_graphs == True:
        for metrics in mgr_list:
            all_graphs(metrics)


def print_epoch_summary(mgr_list, repeat_header_interval=8):
    # Define headers
    headers = ["epoch", "PAP", "MAPE", "R^2", "Mean Absolute Error", " Total Absolute Error", "Mean Square Error", "Final Weight", "Final Bias"]

    # Prepare data
    data = []

    row_counter = 0  # To track when to insert the headers

    # Iterate over each metrics object in mgr_list
    for metrics in mgr_list:
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

def print_regression_summary(mgr_list,epochs_to_run, training_set_size):
    # Prepare headers
    headers = ["Model", "Run Time", "Cvg Epoch",  "Mean Abs Err", " Total Abs Err", "Mean Sqr Err", "Sum Sqr Err"]

    # Prepare data
    data = []
    total_loss_for_epoch = []    # MAE  Mean Absolute Error - sum the abs value of the loss....
    for metrics in mgr_list:
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
            .69,
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


def print_binary_summary(mgr_list,epochs_to_run, training_set_size):
    # Prepare headers
    print(f"\nEpochs:{epochs_to_run}\tTraining Samples: {training_set_size}")
    headers = ["Neural Network Arena", "Run Time", "Correct", "Wrong/Loss", "Accuracy", "Precision", "Recall", "F1 Score", "Epoch to Converge", "SAE at Conv"]

    # Prepare data
    data = []
    total_loss_for_epoch = []    # MAE  Mean Absolute Error - sum the abs value of the loss....
    for metrics in mgr_list:
        # Calculate metrics
        accuracy = metrics.accuracy * 100
        precision = metrics.precision * 100
        recall = metrics.recall * 100
        f1_score = metrics.f1_score * 100
        #total_loss_for_epoch.append((metrics.name, metrics.losses))
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
            f"{metrics.converge_epoch:.0f}",
            0, #TODO metrics.MAE_at_convergence
        ])
    print(tabulate(data, headers=headers, tablefmt="grid")) # Print table




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


from itertools import zip_longest
from tabulate import tabulate
from colorama import Style  # Assuming you're using colorama for color resets

from itertools import zip_longest
from tabulate import tabulate
from colorama import Style  # Assuming you're using colorama for color resets

from itertools import zip_longest
from tabulate import tabulate
from colorama import Style  # Assuming you're using colorama for color resets


def print_logs(mgr_list):
    headers = ["Epoch", "Iteration", "Input", "Target", "Prediction", "Error", "Old Weight", "New Weight", "Old Bias", "New Bias",
               "Model"]
    row_counter = 0  # Keep track of the number of rows printed

    # Use zip_longest to iterate over rows in parallel from each metrics list
    max_length = max(len(metrics.log_data) for metrics in mgr_list)

    # Initialize an empty list to hold rows between header reprints
    buffer = []

    for rows in zip_longest(*[metrics.log_data for metrics in mgr_list], fillvalue=None):
        if row_counter % 10 == 0 and buffer:
            # Print accumulated rows with headers
            print(tabulate(buffer, headers=headers, tablefmt="grid"))
            buffer = []  # Clear buffer after printing

        for idx, row in enumerate(rows):
            if row:  # If the row exists (zip_longest fills with None if shorter lists)
                color = colors[idx % len(colors)]  # Cycle through colors if more metrics than colors
                colored_row = [f"{color}{cell}{Style.RESET_ALL}" for cell in row]
                colored_row.append(f"{color}{mgr_list[idx].name}{Style.RESET_ALL}")
                buffer.append(colored_row)

        row_counter += 1

    # Print any remaining rows with headers
    if buffer:
        print(tabulate(buffer, headers=headers, tablefmt="grid"))

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

