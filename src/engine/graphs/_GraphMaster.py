from src.engine.RamDB import RamDB
from src.engine.graphs.Multiline_Weight_Bias_ErrorRamDB import plot_multi_scale
from src.engine.graphs.Weight_Change_Analysis import weight_change_analysis
from src.engine.graphs.WeightsForXOR import plot_weights_and_mae
import re


def parse_weights_or_biases(data: str, is_weight=True):
    """
    Parses weights or biases from a string in the format:
    '0: [value1, value2]\n1: [value3, value4]' (weights) or
    '0: value1\n1: value2' (biases).

    Args:
        data (str): String containing the weights or biases.
        is_weight (bool): Whether to parse weights (default) or biases.

    Returns:
        dict: A dictionary mapping labels (e.g., 'N0W1') to values.
    """
    parsed = {}
    lines = data.split('\n')
    for line in lines:
        match = re.match(r'(\d+): \[?([-+]?[\d.]+)(?:, ([-+]?[\d.]+))?\]?', line)
        if match:
            neuron = int(match.group(1))  # Directly use the neuron number
            if is_weight:
                parsed[f'N{neuron}W1'] = float(match.group(2))  # First weight
                if match.group(3):
                    parsed[f'N{neuron}W2'] = float(match.group(3))  # Second weight
            else:
                parsed[f'N{neuron}Bias'] = float(match.group(2))
    return parsed


def query_for_report(db):
    SQL = "SELECT epoch, weights, biases, mean_absolute_error FROM EpochSummary"
    results = db.query(SQL)
    #print("Query Results>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #print(results)


    report_data = []
    for row in results:
        weights = parse_weights_or_biases(row['weights'], is_weight=True)
        biases = parse_weights_or_biases(row['biases'], is_weight=False)
        report_entry = {
            'epoch': row['epoch'],
            'weights': weights,  # Parsed weights with neuron labels
            'biases': biases,    # Parsed biases with neuron labels
            'mean_absolute_error': row['mean_absolute_error'],
        }
        report_data.append(report_entry)

    #print ("REPORT DATA))))))))))))))))))))))))))")
    #print(report_data)
    return report_data


def graph_master(db: RamDB):
    models= db.query("Select distinct model_id from Iteration", None,False)
    #db.list_tables()
    #print (models)
    #db.query_print("Select * from Neuron")
    for model in models:
        title = f"{model}\n"
        report_data = query_for_report(db)
        #epoch_summaries = db.query("SELECT * FROM EpochSummary where model_id = ? ORDER BY epoch",model)
        #print (epoch_summaries)
        #run_multiline_weight_bias_error(mgr.epoch_summaries, title)
        #weight_change_analysis(mgr.epoch_summaries)
        plot_multi_scale(report_data, title)
    #headers = ["Epoch Summary", "Epoch", "Final\nWeight", "Final\nBias","Correct", "Wrong", "Accuracy", "Mean\nAbs Err", "Mean\nSqr Err"]



