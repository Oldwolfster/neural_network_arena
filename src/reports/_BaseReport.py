import sqlite3
from tabulate import tabulate

class BaseReport:
    """Base class for SQL-driven reports with reusable query execution and tabulation."""
    
    def __init__(self, *args):
        """
        Initialize the report with a reference to the database.
        :param RamDB: Database connection object
        """
        self.db = args[0]

    def run_sql_report(self, sql: str, params: list = None, limit: int = 10):
        """
        Execute a SQL query, retrieve results, and display them in a tabulated format.
        
        :param sql: SQL query string with optional placeholders.
        :param params: List of parameters for the SQL query.
        :param limit: Number of records to display per batch.
        """
        try:
            result_set = self.db.query(sql, params or [], as_dict=True)
            if not result_set:
                print("No data returned.")
                return
            
            headers = result_set[0].keys()  # Extract column names from first record

            # Loop through the records in batches of `limit`
            for i in range(0, len(result_set), limit):
                batch = result_set[i:i + limit]
                print(tabulate(batch, headers="keys", tablefmt="fancy_grid"))

        except Exception as e:
            print(f"Error running report: {e}")
