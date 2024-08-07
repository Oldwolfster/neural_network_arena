import matplotlib.pyplot as plt
import numpy as np

def all_graphs(metrics):
    plot_training_metrics(metrics)
    plot_convergence(metrics)
    #plot_precision_recall(metrics)
    plot_weight_distribution(metrics)

def plot_training_metrics(metrics):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f'Training Metrics for {metrics.name}', fontsize=16)

    # Plot 1: Loss over epochs
    axs[0, 0].plot(metrics.losses)
    axs[0, 0].set_title('Loss over Epochs')
    #axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_xlabel(' ')
    axs[0, 0].set_ylabel('Total Loss')

    # Plot 2: Accuracy over epochs
    axs[0, 1].plot(metrics.percents)
    axs[0, 1].set_title('Accuracy over Epochs')
    axs[0, 1].set_xlabel(' ')
    axs[0, 1].set_ylabel('Accuracy (%)')

    # Plot 3: Weight changes within the last epoch
    axs[1, 0].plot(metrics.weights_this_epoch)
    axs[1, 0].set_title('Weight Changes (Last Epoch)')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Weight')

    # Plot 4: Confusion Matrix
    cm = [[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]]
    axs[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axs[1, 1].set_title('Confusion Matrix')
    axs[1, 1].set_xlabel('Predicted')
    axs[1, 1].set_ylabel('Actual')
    for i in range(2):
        for j in range(2):
            axs[1, 1].text(j, i, str(cm[i][j]), ha='center', va='center')
    axs[1, 1].set_xticks([0, 1])
    axs[1, 1].set_yticks([0, 1])
    axs[1, 1].set_xticklabels(['Negative', 'Positive'])
    axs[1, 1].set_yticklabels(['Negative', 'Positive'])

    plt.tight_layout()
    plt.show()

def plot_convergence(metrics):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics.losses)
    plt.title(f'Convergence of {metrics.name}')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.axvline(x=metrics.epochs_to_converge, color='r', linestyle='--', label='Convergence Point')
    plt.legend()
    plt.show()

def plot_precision_recall(metrics):
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 1, 100)
    plt.plot(x, x, 'r--', label='Random Classifier')
    plt.plot(metrics.recall, metrics.precision, 'bo', label='Model Performance')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {metrics.name}')
    plt.legend()
    plt.show()

def plot_weight_distribution(metrics):
    plt.figure(figsize=(10, 6))
    plt.hist(metrics.weights_this_epoch, bins=30)
    plt.title(f'Weight Distribution in Last Epoch for {metrics.name}')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.show()

# Usage:
# Assuming 'metrics' is an instance of your Metrics class after training
# plot_training_metrics(metrics)
# plot_convergence(metrics)
# plot_precision_recall(metrics)
# plot_weight_distribution(metrics)