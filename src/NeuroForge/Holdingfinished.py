from src.engine.RamDB import RamDB


def get_weight_min_max(self, db: RamDB, model_id: str, neuron_id: int):
    """
    Retrieves:
    1. The global maximum absolute weight across all epochs and neurons.
    2. The maximum absolute weight for each individual weight index across all epochs.

    Returns:
        global_max (float): The single highest absolute weight in the entire model.
        max_per_weight (list): A list of max absolute weights for each weight index.
    """

    # ✅ Query 1: Get the highest absolute weight overall
    SQL_GLOBAL_MAX = """
        SELECT MAX(ABS(value)) AS global_max
        FROM (
            SELECT json_each.value AS value
            FROM Neuron, json_each(Neuron.weights)
            WHERE model = ? and nid = ?
        )
    """
    global_max_result = db.query(SQL_GLOBAL_MAX, (model_id, neuron_id))
    global_max = global_max_result[0]['global_max'] if global_max_result and global_max_result[0]['global_max'] is not None else 1.0

    # ✅ Query 2: Get the max absolute weight per weight index
    SQL_MAX_PER_WEIGHT = """
        SELECT key, MAX(ABS(value)) AS max_weight
        FROM (
            SELECT json_each.key AS key, json_each.value AS value
            FROM Neuron, json_each(Neuron.weights)
            WHERE model = ? and nid = ?
        )
        GROUP BY key
        ORDER BY key ASC
    """
    max_per_weight_result = db.query(SQL_MAX_PER_WEIGHT, (model_id, neuron_id))

    # Convert result to a list, ensuring order by index (key)
    max_per_weight = []
    for row in max_per_weight_result:
        index = row['key']
        weight = row['max_weight']
        # Ensure correct placement in the list
        while len(max_per_weight) <= index:
            max_per_weight.append(0)  # Initialize missing indices
        max_per_weight[index] = weight

    return global_max, max_per_weight