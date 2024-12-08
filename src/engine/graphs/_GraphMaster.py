from src.engine.graphs.Multiline_Weight_Bias_Error import run_multiline_weight_bias_error
from src.engine.graphs.Weight_Change_Analysis import weight_change_analysis
from src.engine.MetricsMgr import MetricsMgr
from typing import List

def graph_master(mgr_list : List[MetricsMgr]):

    for mgr in mgr_list:
        title = f"{mgr.name}\n"
        run_multiline_weight_bias_error(mgr.epoch_summaries, title)
        #weight_change_analysis(mgr.epoch_summaries)

    #headers = ["Epoch Summary", "Epoch", "Final\nWeight", "Final\nBias","Correct", "Wrong", "Accuracy", "Mean\nAbs Err", "Mean\nSqr Err"]



