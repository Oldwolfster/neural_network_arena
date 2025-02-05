from src.Reports._BaseReport import BaseReport
from src.engine.RamDB import RamDB

class ReportingMadeEasy(BaseReport):
    def __init__(self, *args):
        super().__init__(*args)
        self.dbRam = args[0]
    def run_report(self):
        """
        This method is invoked when user selects this report from Report Menu
        """
        self.run_sql_report()

    def get_parameters():
        print("Get parameters")


