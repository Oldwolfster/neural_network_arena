from src.NeuroForge.DisplayModel__Neuron_Base import DisplayModel__Neuron_Base
from src.engine.Utils_DataClasses import ez_debug

class DisplayModel__NeuronScaler(DisplayModel__Neuron_Base):
    def _from_base_constructor(self):
        """Called from DisplayModel_Neuron_Base constructor"""
        ez_debug(text_ver = self.text_version)
        if self.text_version == "Verbose":
            self.banner_text = "Input Scaler"
        else:
            self.banner_text = "Scaler"


