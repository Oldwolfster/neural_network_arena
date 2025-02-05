from src.Reports._BaseReport import BaseReport
from src.engine.RamDB import RamDB
class ReportingMadeEasy(BaseReport):
    def purpose(self) -> str:
        return "📝 Purpose: Verify that loss gradient, activation gradient, and error signals are calculated correctly."

    def what_to_look_for(self) -> str:
        return """
If activation gradients are all tiny (~0.0001), vanishing gradient is happening.
If error signals are way bigger than loss gradients, weight updates may be unstable.
If loss gradient is flat, loss isn’t affecting training.       
                """
    def report_logic(self, *args):
        """
        This method is invoked when user selects this report from Report Menu
        """
        SQL =   """
                SELECT 
                    epoch_n, iteration_n, layer_id, nid, 
                    activation_gradient, error_signal, loss_gradient
                FROM Neuron 
                --WHERE model = ?
                ORDER BY epoch_n, iteration_n, layer_id, nid;
                """
        self.run_sql_report(SQL)

    def __init__(self, *args):
        super().__init__(*args)
        self.dbRam = args[0]
