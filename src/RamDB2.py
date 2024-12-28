import json
import sqlite3

import numpy as np


class RamDB:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:', isolation_level=None)
        self.cursor = self.conn.cursor()
        self.tables = {}  # To track schemas for validation

    def _infer_schema(self, obj):
        """
        Infer SQLite schema from a Python object.
        """
        schema = {}
        for attr, value in vars(obj).items():
            if isinstance(value, int):
                schema[attr] = "INTEGER"
            elif isinstance(value, float):
                schema[attr] = "REAL"
            elif isinstance(value, str):
                schema[attr] = "TEXT"
            elif isinstance(value, (list, dict)):
                schema[attr] = "TEXT"  # Serialize as JSON
            elif isinstance(value, np.ndarray):  # Handle numpy arrays
                schema[attr] = "TEXT"  # Serialize as JSON
            else:
                raise TypeError(f"Unsupported field type: {type(value)} for attribute '{attr}'")
        return schema
    def _create_table(self, table_name, schema):
        """
        Create a SQLite table dynamically with context fields prioritized and a composite primary key.
        """
        # Define context fields and reorder schema
        context_fields = ['epoch', 'iteration']
        reordered_schema = {key: schema[key] for key in context_fields if key in schema}
        reordered_schema.update({key: schema[key] for key in schema if key not in context_fields})

        # Add composite primary key
        columns = ", ".join([f"{name} {type}" for name, type in reordered_schema.items()])
        primary_key = "PRIMARY KEY (epoch, iteration, nid)"
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns}, {primary_key});"

        # Execute table creation
        self.cursor.execute(sql)
        self.tables[table_name] = schema  # Track schema


    def add(self, obj, **context):
        """
        Add an object to the database with required context fields (e.g., epoch, iteration).
        Automatically creates the table if it does not exist.
        """
        # Determine the table name from the object's class
        table_name = obj.__class__.__name__

        # Infer the schema from the object
        schema = self._infer_schema(obj)

        # Merge context fields into the schema
        for key, value in {"epoch": context.get('epoch'), "iteration": context.get('iteration')}.items():
            if key and value is not None:
                if isinstance(value, int):
                    schema[key] = "INTEGER"
                elif isinstance(value, float):
                    schema[key] = "REAL"
                elif isinstance(value, str):
                    schema[key] = "TEXT"
                else:
                    raise TypeError(f"Unsupported context field type: {type(value)} for '{key}'")

        # Create the table if it doesn't exist
        if table_name not in self.tables:
            self._create_table(table_name, schema)
            self.tables[table_name] = schema  # Track schema

        # Merge object attributes and context fields
        data = {**vars(obj), **context}

        # Reorder fields for insertion
        ordered_data = {key: data[key] for key in schema.keys()}

        # Serialize fields (e.g., numpy arrays, lists) into JSON-friendly formats
        ordered_data = {
            key: (json.dumps(value.tolist()) if isinstance(value, np.ndarray) else
                  json.dumps(value) if isinstance(value, (list, dict)) else
                  value)
            for key, value in ordered_data.items()
        }

        # Prepare SQL INSERT statement
        columns = ", ".join(ordered_data.keys())
        placeholders = ", ".join(["?"] * len(ordered_data))
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders});"

        # Execute the insert
        self.cursor.execute(sql, tuple(ordered_data.values()))



    def query(self, sql, as_dict=True):
        """
        Execute a SQL query and return the results.
        """
        try:
            self.cursor.execute(sql)
            rows = self.cursor.fetchall()
            if as_dict:
                column_names = [description[0] for description in self.cursor.description]
                return [dict(zip(column_names, row)) for row in rows]
            return rows  # Default to tuples
        except sqlite3.Error as e:
            raise RuntimeError(f"SQL query failed: {e}")



class Neuron:
    """
    Represents a single neuron with weights, bias, and an activation function.
    """

    def __init__(self, nid: int, input_count: int, learning_rate: float, layer_id: int = 0):
        self.nid = nid
        self.layer_id = layer_id  # Add layer_id to identify which layer the neuron belongs to
        self.input_count = input_count
        self.weights = np.array([(nid + 1) * 0 for _ in range(input_count)], dtype=np.float64)


N1 = Neuron(1, 1, .001, 1)
N2 = Neuron(2, 2, .001, 1)
N3 = Neuron(3, 3, .001, 2)

db = RamDB()
# Add a neuron with context
db.add(N1, epoch=1, iteration=10)
db.add(N2, epoch=1, iteration=20)
db.add(N3, epoch=2, iteration=30)
result = db.query("SELECT * FROM Neuron")
result2 = db.query("SELECT * FROM Neuron", False)
print(result)
print(result2)
