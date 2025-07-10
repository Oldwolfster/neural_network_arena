# src/engine/LegoLister.py
import inspect
import importlib

class LegoLister:
    def __init__(self):
        self._cached_legos = {}  # new cache for dual maps
        self.registry = {
            "loss_function": ("src.Legos.LossFunctions", "StrategyLossFunction"),
            "hidden_activation": ("src.Legos.ActivationFunctions", "StrategyActivationFunction"),
            "output_activation": ("src.Legos.ActivationFunctions", "StrategyActivationFunction"),
            "optimizer": ("src.Legos.Optimizers", "StrategyOptimizer"),
            "initializer": ("src.Legos.WeightInitializers", "ScalerWeightInitializer"),
            # Add more as needed
        }

        self.registry = {
            "loss_function": ("NNA.Legos.LossFunctions", "StrategyLossFunction"),
            "hidden_activation": ("NNA.Legos.ActivationFunctions", "StrategyActivationFunction"),
            "output_activation": ("NNA.Legos.ActivationFunctions", "StrategyActivationFunction"),
            "optimizer": ("NNA.Legos.Optimizers", "StrategyOptimizer"),
            "initializer": ("NNA.Legos.WeightInitializers", "ScalerWeightInitializer"),
        }


    
    def list_legos(self, config_property: str):
        if config_property not in self.registry:
            raise ValueError(f"No lego registered for '{config_property}'")

        module_path, base_class_name = self.registry[config_property]
        module = importlib.import_module(module_path)
        base_class = getattr(module, base_class_name)

        by_var_name = {}
        by_display_name = {}

        for name, obj in inspect.getmembers(module):
            if isinstance(obj, base_class):
                by_var_name[name] = obj
                by_display_name[getattr(obj, "name", name)] = obj  # fallback to var name if no .name

        self._cached_legos[config_property] = {
            "by_var": by_var_name,
            "by_name": by_display_name
        }

        print(f"\n🔎 Found legos for '{config_property}' why is it doing this on resume?:")
        for k in by_display_name:
            print(f"  {k}")

        return by_display_name  # Return old default for compatibility


    def list_all(self):
        return {
            prop: self.list_legos(prop)
            for prop in self.registry
        }

    def get_lego(self, config_property: str, name: str):
        if config_property not in self._cached_legos:
            self.list_legos(config_property)  # populate cache

        lego_map = self._cached_legos[config_property]

        # Try lookup by display name first (safe for DB)
        if name in lego_map["by_name"]:
            return lego_map["by_name"][name]

        # Fallback to internal var name
        if name in lego_map["by_var"]:
            return lego_map["by_var"][name]

        all_names = list(lego_map["by_name"].keys()) + list(lego_map["by_var"].keys())
        raise ValueError(f"❌ Unknown lego name '{name}' for '{config_property}'. Available: {all_names}")




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
