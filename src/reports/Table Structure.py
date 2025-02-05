from src.Reports._BaseReport import BaseReport
from src.engine.RamDB import RamDB


class ReportingMadeEasy(BaseReport):
    def purpose(self) -> str:
        return "📝 Show fields and 3 sample records of neuron table."

    def what_to_look_for(self) -> str:
        return """If all activations look the same for every input, something is wrong.
                  If hidden layer activations are all close to 1 or -1, sigmoid/tanh is saturating.
                    If output is always ~0.5, we have bad weight initialization.        
                """

    def report_logic(self, *args):
        """
        This method is invoked when user selects this report from Report Menu
        """


        SQL =   """
                
                """
        self.run_sql_report("SELECT * FROM Neuron LIMIT 3;")
        self.run_sql_report("SELECT * FROM Iteration LIMIT 3;")

    def __init__(self, *args):
        super().__init__(*args)
