import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# Function to forward pass through your network using stored weights
def forward_pass(x1, x2, weights_dict):
    """
    Perform forward pass through network using stored weights from database

    Args:
        x1, x2: Input values (the two XOR inputs)
        weights_dict: Dictionary containing weights and biases

    Returns:
        Final output after sigmoid activation
    """
    # First hidden neuron (ID: 0-0)
    h1_input = weights_dict['h1_bias'] + weights_dict['h1_w1'] * x1 + weights_dict['h1_w2'] * x2
    h1_output = np.tanh(h1_input)

    # Second hidden neuron (ID: 0-1)
    h2_input = weights_dict['h2_bias'] + weights_dict['h2_w1'] * x1 + weights_dict['h2_w2'] * x2
    h2_output = np.tanh(h2_input)

    # Output neuron (ID: 1-0)
    o_input = weights_dict['o_bias'] + weights_dict['o_w1'] * h1_output + weights_dict['o_w2'] * h2_output
    o_output = 1 / (1 + np.exp(-o_input))  # sigmoid

    return o_output

# Connect to your SQLite database
def get_epochs_data_original(db_path):
    """
    Retrieve weight data for each epoch from the database

    Args:
        db_path: Path to SQLite database

    Returns:
        List of dictionaries, each containing weights for one epoch
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Adjust this query based on your actual database schema
    cursor.execute("""
        SELECT 
            epoch, 
            h1_bias, h1_w1, h1_w2,
            h2_bias, h2_w1, h2_w2,
            o_bias, o_w1, o_w2
        FROM 
            weights_table
        ORDER BY 
            epoch
    """)

    epochs_data = []
    for row in cursor.fetchall():
        epoch, h1_bias, h1_w1, h1_w2, h2_bias, h2_w1, h2_w2, o_bias, o_w1, o_w2 = row
        epochs_data.append({
            'epoch': epoch,
            'h1_bias': h1_bias, 'h1_w1': h1_w1, 'h1_w2': h1_w2,
            'h2_bias': h2_bias, 'h2_w1': h2_w1, 'h2_w2': h2_w2,
            'o_bias': o_bias, 'o_w1': o_w1, 'o_w2': o_w2
        })

    conn.close()
    return epochs_data

def get_epochs_data(db_path):
    """
    Retrieve weight data for each epoch from the database

    Args:
        db_path: Path to SQLite database

    Returns:
        List of dictionaries, each containing weights for one epoch
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get distinct epochs to iterate through
    cursor.execute("SELECT DISTINCT epoch FROM Weight ORDER BY epoch")
    epochs = cursor.fetchall()

    epochs_data = []

    for epoch_row in epochs:
        epoch = epoch_row[0]

        # Create a dictionary to store weights for this epoch
        epoch_data = {'epoch': epoch}

        # Query for hidden layer 1 weights
        cursor.execute("""
            SELECT nid, weight_id, value 
            FROM Weight 
            WHERE epoch = ? AND nid LIKE 'h1%'
            ORDER BY nid, weight_id
        """, (epoch,))
        h1_weights = cursor.fetchall()

        # Query for hidden layer 2 weights
        cursor.execute("""
            SELECT nid, weight_id, value 
            FROM Weight 
            WHERE epoch = ? AND nid LIKE 'h2%'
            ORDER BY nid, weight_id
        """, (epoch,))
        h2_weights = cursor.fetchall()

        # Query for output layer weights
        cursor.execute("""
            SELECT nid, weight_id, value 
            FROM Weight 
            WHERE epoch = ? AND nid LIKE 'o%'
            ORDER BY nid, weight_id
        """, (epoch,))
        o_weights = cursor.fetchall()

        # Process hidden layer 1 weights
        for nid, weight_id, value in h1_weights:
            if weight_id == 0:  # bias
                epoch_data['h1_bias'] = value
            else:
                epoch_data[f'h1_w{weight_id}'] = value

        # Process hidden layer 2 weights
        for nid, weight_id, value in h2_weights:
            if weight_id == 0:  # bias
                epoch_data['h2_bias'] = value
            else:
                epoch_data[f'h2_w{weight_id}'] = value

        # Process output layer weights
        for nid, weight_id, value in o_weights:
            if weight_id == 0:  # bias
                epoch_data['o_bias'] = value
            else:
                epoch_data[f'o_w{weight_id}'] = value

        epochs_data.append(epoch_data)

    conn.close()
    return epochs_data

# Plot decision boundary for a specific epoch
def plot_decision_boundary(weights_dict, epoch_num, ax=None):
    """
    Plot the decision boundary for a specific set of weights

    Args:
        weights_dict: Dictionary containing weights and biases
        epoch_num: Epoch number (for title)
        ax: Matplotlib axis to plot on (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Create a meshgrid of points
    x1_range = np.linspace(-0.5, 1.5, 100)
    x2_range = np.linspace(-0.5, 1.5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Calculate output for each point
    Z = np.zeros_like(X1)
    for i in range(len(x1_range)):
        for j in range(len(x2_range)):
            Z[j, i] = forward_pass(X1[j, i], X2[j, i], weights_dict)

    # Plot decision boundary
    contour = ax.contourf(X1, X2, Z, levels=20, cmap='RdYlGn', alpha=0.8)

    # Plot the XOR points
    xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_outputs = np.array([0, 1, 1, 0])

    for i, (x, y) in enumerate(xor_inputs):
        color = 'lime' if xor_outputs[i] == 1 else 'red'
        ax.scatter(x, y, c=color, s=100, edgecolor='black', zorder=10)

    # Calculate and show how many points are correctly classified
    correct = 0
    for i, (x, y) in enumerate(xor_inputs):
        output = forward_pass(x, y, weights_dict)
        predicted = 1 if output > 0.5 else 0
        if predicted == xor_outputs[i]:
            correct += 1

    # Add details to plot
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('Input X1')
    ax.set_ylabel('Input X2')
    ax.set_title(f'Epoch {epoch_num}: Decision Boundary (Correct: {correct}/4)')

    # Add XOR truth labels
    for i, (x, y) in enumerate(xor_inputs):
        ax.annotate(f"{int(xor_outputs[i])}",
                   (x, y),
                   textcoords="offset points",
                   xytext=(0,10),
                   ha='center')

    # Return for colorbar if needed
    return contour

# Create animation of decision boundary evolving through epochs
def create_decision_boundary_animation(epochs_data, output_file='xor_decision_boundary.mp4'):
    """
    Create animation showing how decision boundary evolves during training

    Args:
        epochs_data: List of dictionaries with weights for each epoch
        output_file: File to save animation to
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame):
        ax.clear()
        weights = epochs_data[frame]
        contour = plot_decision_boundary(weights, weights['epoch'], ax)
        return [contour]

    # Sample epochs if there are too many
    if len(epochs_data) > 100:
        sample_indices = np.linspace(0, len(epochs_data)-1, 100, dtype=int)
        sampled_epochs = [epochs_data[i] for i in sample_indices]
    else:
        sampled_epochs = epochs_data

    ani = animation.FuncAnimation(fig, update, frames=len(sampled_epochs), blit=False)
    ani.save(output_file, writer='ffmpeg', fps=5)
    plt.close()

    print(f"Animation saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Path to your SQLite database
    db_path = "path_to_your_database.db"

    # Get weights for each epoch
    epochs_data = get_epochs_data(db_path)

    # Create animation
    create_decision_boundary_animation(epochs_data)

    # Or plot specific epochs (e.g., first, middle, breakthrough, final)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    epochs_to_plot = [
        0,  # First epoch
        len(epochs_data) // 3,  # Early training
        len(epochs_data) // 2,  # Middle of training
        len(epochs_data) - 1  # Final epoch
    ]

    for i, epoch_idx in enumerate(epochs_to_plot):
        plot_decision_boundary(epochs_data[epoch_idx], epochs_data[epoch_idx]['epoch'], axes[i])

    plt.tight_layout()
    plt.savefig('xor_decision_boundaries.png')
    plt.show()