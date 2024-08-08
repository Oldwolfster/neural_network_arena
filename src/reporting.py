from tabulate import tabulate
from graphing import *

logs_to_print = -1      # It will print this many of the first and this many of the last iteration logs


def give_me_a_line():
    print(f"{'=' * 129}")

def print_results(metrics_list, training_data, display_graphs):
    # print_logs(metrics_list)
    print_grid(metrics_list)
    print(f"Training Data: {training_data}")
    if display_graphs == True:
        for metrics in metrics_list:
            all_graphs(metrics)


def print_grid(metrics_list):
    # Prepare headers
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


def print_logs(metrics_list):
    for metrics in metrics_list:
        # Print logs for each metrics object
        print(f"Logs for {metrics.name}:")
        if logs_to_print < 0:  # Sentinel value to print all logs.
            for log in metrics.log_list:
                print(log)

        for log in metrics.log_list[:logs_to_print]:
            print(log)
            # give_me_a_line()

        for log in metrics.log_list[-logs_to_print:]:
            print(log)
            # give_me_a_line()


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
