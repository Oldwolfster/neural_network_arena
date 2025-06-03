"""
        CREATE TABLE IF NOT EXISTS NNA_history (
            timestamp DATETIME,
            run_id INTEGER,
            runtime_seconds REAL,
            gladiator TEXT,
            arena TEXT,
            accuracy REAL,
            best_mae REAL,
            final_mae REAL,
            architecture TEXT,
            loss_function TEXT,
            hidden_activation TEXT,
            output_activation TEXT,
            weight_initializer TEXT,
            normalization_scheme TEXT,
            learning_rate REAL,
            epoch_count INTEGER,
            convergence_condition TEXT,
            problem_type TEXT,
            sample_count INTEGER,
            seed INTEGER
            pk INTEGER PRIMARY KEY AUTOINCREMENT
        )
"""