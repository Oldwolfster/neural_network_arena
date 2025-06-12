from src.NeuroForge import Const
from src.NeuroForge.DisplayModel import DisplayModel


class ModelGenerator:
    model_shapes    = {}  # {model_id: (layer_count, max_neurons)}
    split_direction = None
    model_positions = {}
    model_count     = 0
    models          = []
    display_models  = []
    @staticmethod
    def create_models():
        ModelGenerator.divide_real_estate()
        ModelGenerator.instantiate_DisplayModels()
        return ModelGenerator.display_models

    @staticmethod
    def instantiate_DisplayModels():
        """Create DisplayModel instances for each model."""
        ModelGenerator.display_models = []

        for TRI in Const.TRIs:  # Loop through stored Config objects
            if TRI.run_id not in ModelGenerator.model_positions:
                continue  # Skip models not assigned a position

            model_position = ModelGenerator.model_positions[TRI.run_id]  # Fetch position
            #print(f"model_position = {model_position}") #print graphs position
            display_model = DisplayModel(TRI, model_position)  # Pass both config & position
            display_model.initialize_with_model_info()
            ModelGenerator.display_models.append(display_model)

    @staticmethod
    def divide_real_estate():
        """Entry point to generate model layout based on model count."""
        ModelGenerator.get_model_shapes()

        ModelGenerator.model_count = len(ModelGenerator.model_shapes)
        if ModelGenerator.model_count == 1:
            LayoutSelector_single_model.determine_layout()
        elif 2 <= ModelGenerator.model_count <= 4:
            LayoutSelector_2_or_3.determine_layout()
        else:
            print(f"🚨 Unsupported model count: {ModelGenerator.model_count}")  # Placeholder for future expansion
        #print(f"{ModelGenerator.model_count} Models:Calculated Positions: {ModelGenerator.model_positions}")


    @staticmethod
    def get_model_shapes():
        """Extracts shape details for each model to determine visualization layout."""
        ModelGenerator.model_shapes = {}  # Reset stored data

        for TRI in Const.TRIs:
            layer_count = len(TRI.config.architecture)

            # Enforce minimum 2 neurons per layer for visualization purposes
            padded_architecture = [max(2, n) for n in TRI.config.architecture]
            max_neurons = max(padded_architecture)

            ModelGenerator.model_shapes[TRI.run_id] = (layer_count, max_neurons)

class LayoutSelector_single_model:
    @staticmethod
    def determine_layout():
        """Centers a single model in the available visualization space."""
        model_id = list(ModelGenerator.model_shapes.keys())[0]  # Only one model exists

        ModelGenerator.model_positions = {
            model_id: {
                "left": Const.MODEL_AREA_PIXELS_LEFT,
                "top": Const.MODEL_AREA_PIXELS_TOP,
                "width": Const.MODEL_AREA_PIXELS_WIDTH,
                "height": Const.MODEL_AREA_PIXELS_HEIGHT,
            }
        }

        #print(f"Single Model Positioned at: {ModelGenerator.model_positions}")

class LayoutSelector_2_or_3:
    @staticmethod
    def determine_layout():
        """Decides layout strategy for 2-3 models."""
        LayoutSelector_2_or_3.determine_horizontal_vs_vertical()
        LayoutSelector_2_or_3.determine_split()

    @staticmethod
    def determine_horizontal_vs_vertical():
        """Decides whether to split models horizontally or vertically based on architecture and available space."""
        model_shapes = ModelGenerator.model_shapes

        # Summarize total layers and max neurons across all models
        total_layers = sum(shape[0] for shape in model_shapes.values())
        total_max_neurons = sum(shape[1] for shape in model_shapes.values())

        # Available space dimensions
        available_width = Const.MODEL_AREA_PIXELS_WIDTH
        available_height = Const.MODEL_AREA_PIXELS_HEIGHT

        # Compute aspect ratios
        model_aspect_ratio = total_max_neurons / total_layers if total_layers > 0 else 1
        available_aspect_ratio = available_width / available_height

        # Decision logic: Compare aspect ratios
        ModelGenerator.split_direction = "vertical" if model_aspect_ratio > available_aspect_ratio else "horizontal"
        #print(f"Splitting models {ModelGenerator.split_direction.upper()} based on shape & available space.")

    @staticmethod
    def determine_split():
        """Allocates space dynamically based on model proportions and chosen split direction."""
        model_shapes = ModelGenerator.model_shapes
        model_ids = list(model_shapes.keys())

        total_size = (
            sum(shape[1] for shape in model_shapes.values())  # Max neurons per layer
            if ModelGenerator.split_direction == "horizontal"
            else sum(shape[0] for shape in model_shapes.values())  # Number of layers
        )

        if ModelGenerator.split_direction == "horizontal":
            base_top = Const.MODEL_AREA_PIXELS_TOP
            height_remaining = Const.MODEL_AREA_PIXELS_HEIGHT
            ModelGenerator.model_positions = {}

            for model_id in model_ids:
                percent_of_total = model_shapes[model_id][1] / total_size
                model_height = height_remaining * percent_of_total

                ModelGenerator.model_positions[model_id] = {
                    "left": Const.MODEL_AREA_PIXELS_LEFT,
                    "top": base_top,
                    "width": Const.MODEL_AREA_PIXELS_WIDTH,
                    "height": model_height,
                }

                base_top += model_height

        else:  # Vertical split
            base_left = Const.MODEL_AREA_PIXELS_LEFT
            width_remaining = Const.MODEL_AREA_PIXELS_WIDTH
            ModelGenerator.model_positions = {}

            for model_id in model_ids:
                percent_of_total = model_shapes[model_id][0] / total_size
                model_width = width_remaining * percent_of_total

                ModelGenerator.model_positions[model_id] = {
                    "left": base_left,
                    "top": Const.MODEL_AREA_PIXELS_TOP,
                    "width": model_width,
                    "height": Const.MODEL_AREA_PIXELS_HEIGHT,
                }
                base_left += model_width


