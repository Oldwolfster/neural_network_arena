from src.reports._BaseReport import BaseReport
from src.NNA.engine.RamDB import RamDB


class ReportingMadeEasy(BaseReport):
    def purpose(self) -> str:
        return "📝 Purpose: Verify that each forward pass computes correctly by logging neuron activations across the network."

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
                SELECT 
                    epoch_n, iteration_n, layer_id, nid, 
                    activation_value AS output, 
                    raw_sum, bias_before, weights_before
                FROM Neuron 
                -- WHERE model = ?
                ORDER BY epoch_n, iteration_n, layer_id, nid;
                """
        self.run_sql_report(SQL)

    def __init__(self, *args):
        super().__init__(*args)
