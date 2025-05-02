from typing import List

import pygame

from src.NeuroForge import Const
from src.NeuroForge.Popup_Base import Popup_Base
from src.engine.Config import Config
from src.Legos.Optimizers import BatchMode
from src.engine.Utils import draw_rect_with_border, draw_text_with_background, ez_debug, check_label_collision, get_text_rect, beautify_text, smart_format

class ArchitecturePopup(Popup_Base):
    def __init__(self, model, configs: List[Config]):
        super().__init__()

    def header_text(self):
        return "Model Architecture Comparison"


    def draw_dividers(self, surf, col_w):

        y = self.y_coord_for_row(1)
        #pygame.draw.line(surf, Const.COLOR_BLACK, (0,y),(ARCH_W,y),2)
    # no highlights required here


    def content_to_display(self):
        configs = Const.configs
        def get_labels():
            return [

                ("Model Definition",     ""),  # spacer
                ("",                     "Neuron Layout"),
                ("",                     "Initializer"),
                ("",                     "Hidden Activation"),
                ("",                     "Output Activation"),
                ("",                     ""),  # spacer

                ("Training Setup",       ""),
                ("",                     "Optimizer"),
                ("",                     "Batch Mode"),
                ("",                     "Batch Size"),
                ("",                     "Learning Rate"),
                ("",                     "Input Scaler"),
                ("",                     "ROI Mode"),
                ("",                     ""),  # spacer

                ("Training Outcome",     ""),
                ("",                     "Training Time (s)"),
                ("",                     "Final Epoch"),
                ("",                     "Best Error"),
                ("",                     "Best Error @ Epoch"),
                ("",                     "Convergence"),
            ]

        def architecture(bp) -> str:

            if bp == [1]:
                return "Perceptron"
            else:
                layers = ", ".join(str(x) for x in bp)
                return f"{layers} Neurons/Layer"

        def describe(cfg):
            return [
                "",  # Training Setup header, no value
                architecture(cfg.architecture),
                cfg.initializer.name,
                cfg.hidden_activation.name,
                cfg.output_activation.name,
                "",

                "",  # Training Setup header, no value
                cfg.optimizer.name,
                beautify_text(BatchMode(cfg.batch_mode).name),
                f"{cfg.batch_size} (Samples)",
                smart_format(cfg.learning_rate),
                beautify_text(cfg.input_scaler.name),
                beautify_text(cfg.roi_mode.name),
                "",

                "",  # Training Outcome header
                smart_format(cfg.seconds),
                f"{cfg.final_epoch} (Epochs)",
                smart_format(cfg.lowest_error),
                cfg.lowest_error_epoch,
                cfg.cvg_condition,
            ]


        labels = get_labels()  # returns list of (group, prop_name)
        rows = [ [group, label] + [describe(cfg)[i] for cfg in configs] for i, (group, label) in enumerate(labels) ]
        return [list(col) for col in zip(*rows)]

