import sqlite3

'''
Let's do this right.  Using sqlite db in ram, gave
    1) Great performance
    2) Extreme flexility in reporting, as you can just use SQL.
However:
    1) There's lots of boiler plate and wiring up fields which as you change your mind, get's to be a pita
    
Objective:
build a helper called RamDB.  aPI still up in the air, but the jist of it is...
1) On instantiation, it creates a sqlite connection in ram
2) Accepts instance of object(or reference to).  will create 
    1) A "holding place/buffer" for accumulating inserts to batch.
    2) A table.
    3) Does so based on interrogating object, not me manually keeping lists in sync 
3) Accepts instances of the object to write, puts them in buffer.
4) Accepts command to store the buffer. 
'''

def prepSQL() -> sqlite3.Connection:
    conn = sqlite3.connect(':memory:', isolation_level=None)
    ramDB = conn.cursor()

    # Create the iterations table
    ramDB.execute('''
        CREATE TABLE IF NOT EXISTS iterations (            
            model_id TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            step INTEGER NOT NULL,
            inputs TEXT NOT NULL,
            target REAL NOT NULL,
            prediction REAL NOT NULL,
            loss REAL NOT NULL,
            PRIMARY KEY (model_id, epoch, step)
        );
    ''')

    # Create the neurons table
    ramDB.execute('''
        CREATE TABLE IF NOT EXISTS neurons (            
            model_id TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            step INTEGER NOT NULL,
            neuron_id INTEGER NOT NULL,               -- Unique ID for each neuron
            layer_id INTEGER NOT NULL,                -- Identifies the layer the neuron belongs to
            weights_before TEXT NOT NULL,             -- Serialized weights PRIOR TO UPDATE (e.g., JSON or CSV)
            weights TEXT NOT NULL,                    -- Serialized weights (e.g., JSON or CSV)
            bias_before REAL NOT NULL,                -- Current bias PRIOR TO UPDATE for the neuron
            bias REAL NOT NULL,                       -- Current bias for the neuron
            output REAL NOT NULL,                     -- Output of the neuron (post-activation)
            PRIMARY KEY (model_id, epoch, step, neuron_id),
            FOREIGN KEY (model_id, epoch, step) REFERENCES iterations (model_id, epoch, step)
        );
    ''')

    return ramDB


def record_neuron(self, iteration, neuron):
    """
    Record a single neuron's data, including before and after states.
    """
    self.metrics_neuron.append({
        'model_id': self.model_id,
        'epoch': iteration.epoch,
        'step': iteration.step,
        'neuron_id': neuron.nid,
        'layer_id': neuron.layer_id,
        'weights_before': dumps(neuron.weights_before.tolist()),  # Serialize weights before
        'weights_after': dumps(neuron.weights.tolist()),  # Serialize weights after
        'bias_before': neuron.bias_before,
        'bias_after': neuron.bias,
        'output': neuron.output
    })

    def write_neurons(self):
        """
        Batch write all accumulated neuron data to the database.
        """
        if not self.metrics_neuron:
            return  # Nothing to write
        sql_neurons = '''
            INSERT INTO neurons (model_id, epoch, step, neuron_id, layer_id, weights_before, weights, bias_before, bias, output)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        values_neurons = [
            (
                neuron['model_id'],
                neuron['epoch'],
                neuron['step'],
                neuron['neuron_id'],
                neuron['layer_id'],
                neuron['weights_before'],
                neuron['weights_after'],
                neuron['bias_before'],
                neuron['bias_after'],
                neuron['output']
            )
            for neuron in self.metrics_neuron
        ]
        self.ramDb.executemany(sql_neurons, values_neurons)
        self.metrics_neuron.clear()


# Create a connection to an in-memory SQLite database
    ramDb = prepSQL()