from src.reports._BaseReport import BaseReport
from src.NNA.engine.RamDB import RamDB


class ReportingMadeEasy(BaseReport):
    def purpose(self) -> str:
        return "📝 Purpose: Verify that weights and bias are actually changing and not oscillating or stagnating."

    def what_to_look_for(self) -> str:
        return """If all activations look the same for every input, something is wrong.
                  If hidden layer activations are all close to 1 or -1, sigmoid/tanh is saturating.
                    If output is always ~0.5, we have bad weight initialization.        
                """

    def report_logic(self, *args):
        """
        This method is invoked when user selects this report from Report Menu
        """
        sql =   """
            SELECT 
            epoch_n, iteration_n, layer_id, nid, 
            weights_before, weights_after, bias_before, bias_after
            FROM Neuron 
            -- WHERE model = ?
            ORDER BY epoch_n, iteration_n, layer_id, nid;
                """
        self.run_sql_report("SELECT * FROM EpochSummary")

    def __init__(self, *args):
        super().__init__(*args)
        self.dbRam = args[0]
