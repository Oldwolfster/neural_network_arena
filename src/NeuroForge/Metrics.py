# src/NeuroForge/Metrics.py

def get_max_error(db) -> int:
    """Retrieve highest abs(error)"""
    sql = "SELECT MAX(abs(error_signal)) as error_signal FROM Neuron"
    rs = db.query(sql)
    return rs[0].get("error_signal")

def get_max_epoch(db) -> int:
    """Retrieve highest epoch."""
    sql = "SELECT MAX(epoch) as max_epoch FROM Iteration"
    rs = db.query(sql)
    return rs[0].get("max_epoch")

def get_max_weight(db) -> float:
    """Retrieve highest weight magnitude."""
    sql = """
        SELECT MAX(ABS(value)) AS max_weight
        FROM (SELECT json_each.value AS value FROM Neuron, json_each(Neuron.weights))
    """
    rs = db.query(sql)
    return rs[0].get("max_weight")

def get_max_iteration(db) -> int:
    """Retrieve highest iteration"""
    sql = "SELECT MAX(iteration) as max_iteration FROM Iteration"
    rs = db.query(sql)
    return rs[0].get("max_iteration")
