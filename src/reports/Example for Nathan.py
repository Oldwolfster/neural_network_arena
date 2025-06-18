from src.reports._BaseReport import BaseReport
from src.NNA.engine.RamDB import RamDB

class ReportingMadeEasy(BaseReport):
    def __init__(self, *args):
        super().__init__(*args)


    def report_logic(self, *args):
        """
        This method is invoked when user selects this report from Report Menu
        """
        self.run_sql_report("SELECT * FROM Neuron")



