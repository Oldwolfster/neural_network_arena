# src/engine/LegoLister.py
import inspect
import importlib

class LegoLister:
    def __init__(self):
        self.registry = {
            "loss_function": ("src.Legos.LossFunctions", "StrategyLossFunction"),
            "hidden_activation": ("src.Legos.ActivationFunctions", "StrategyActivationFunction"),
            "output_activation": ("src.Legos.ActivationFunctions", "StrategyActivationFunction"),
            "optimizer": ("src.Legos.Optimizers", "StrategyOptimizer"),
            "initializer": ("src.Legos.WeightInitializers", "ScalerWeightInitializer"),
            # Add more as needed
        }
    
    def list_legos(self, config_property: str):
        if config_property not in self.registry:
            raise ValueError(f"No lego registered for '{config_property}'")

        module_path, base_class_name = self.registry[config_property]
        module = importlib.import_module(module_path)
        base_class = getattr(module, base_class_name)

        legos = {
            name: obj for name, obj in inspect.getmembers(module)
            if isinstance(obj, base_class)
        }

        print(f"\n🔎 Found legos for '{config_property}':")
        for k, v in legos.items():
            print(f"  {k}: {v}")
        return legos


    def list_all(self):
        return {
            prop: self.list_legos(prop)
            for prop in self.registry
        }

    def list_legos_orig(self, config_property: str):
        if config_property not in self.registry:
            raise ValueError(f"No lego registered for '{config_property}'")

        if config_property == "loss_function":
            from src.Legos.LossFunctions import Loss_Hinge
            return {"Loss_Hinge": Loss_Hinge}

        module_path, class_name = self.registry[config_property]
        module = importlib.import_module(module_path)
        base_class = getattr(module, class_name)

        return {
            name: obj for name, obj in inspect.getmembers(module)
            if isinstance(obj, base_class)
        }
