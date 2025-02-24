from typing import List
from typing import TYPE_CHECKING
#from src.NeuroForge import Display_Manager
from src.engine.ModelConfig import ModelConfig

# ==============================
# UI Constants
# ==============================
SCREEN_WIDTH  = 1200
SCREEN_HEIGHT = 900

MODEL_AREA_PERCENT_LEFT     = 0.14
MODEL_AREA_PERCENT_TOP      = 0.05
MODEL_AREA_PERCENT_WIDTH    = 0.72
MODEL_AREA_PERCENT_HEIGHT   = 0.91
MODEL_AREA_PIXELS_LEFT      = SCREEN_WIDTH * MODEL_AREA_PERCENT_LEFT
MODEL_AREA_PIXELS_TOP       = SCREEN_HEIGHT * MODEL_AREA_PERCENT_TOP
MODEL_AREA_PIXELS_WIDTH     = SCREEN_WIDTH * MODEL_AREA_PERCENT_WIDTH
MODEL_AREA_PIXELS_HEIGHT    = SCREEN_HEIGHT * MODEL_AREA_PERCENT_HEIGHT

MENU_ACTIVE   = False

# ==============================
# Global References
# ==============================
configs: List[ModelConfig] = []
if TYPE_CHECKING:
    from src.NeuroForge.Display_Manager import DisplayManager
    from src.NeuroForge.VCR             import VCR
dm: "DisplayManager" = None  # Lazy reference to avoid circular imports
vcr: "VCR" = None

# ==============================
# Training State
# ==============================
MAX_EPOCH       = 0
CUR_EPOCH       = 1
MAX_ITERATION   = 0
CUR_ITERATION   = 1
MAX_WEIGHT      = 0.0
MAX_ERROR       = 0.0

# ==============================
# Pygame Objects (Initialized Later)
# ==============================
SCREEN          = None
UI_MANAGER      = None
TOOL_TIP        = None
FONT            = None
DISPLAY_MODELS  = []

# ==============================
# UI Customization
# ==============================
JUMP_TO_EPOCH  = 0
COLOR_NEURONS  = True

# ==============================
# Colors
# ==============================
COLOR_WHITE             = (255, 255, 255)
COLOR_BLACK             = (0, 0, 0)
COLOR_SKY_BLUE          = (135, 206, 235)
COLOR_CRIMSON           = (220, 20, 60)
COLOR_FOREST_GREEN      = (34, 139, 34)
COLOR_BLUE              = (50, 50, 255)
COLOR_FOR_BANNER        = (0, 0, 255)
COLOR_FOR_SHADOW        = (30, 30, 100)  # Darker blue for depth
COLOR_FOR_BACKGROUND    = COLOR_WHITE
""" From original

#Epoch and VCR
VCR                     = None
max_epoch       :int    = 0
max_iteration   :int    = 0
max_weight      :float  = 0
max_error       :float  = 0
epoch           :int    = 1
iteration       :int    = 1
epoch_error     :float  = 0.0
error           :float  = 0.0
loss            :float  = 0.0
loss_grd        :float  = 0.0
avg_error       :float  = 0.0
avg_loss        :float  = 0.0
avg_loss_grd    :float  = 0.0

#Pygame objects
screen                  = None
tool_tip                = None
jump_to_epoch           = 0
color_neurons           = True
font                    = None
layerDeleteMe           = 1
neuronDeleteMe          = 1
display_models          = []
screen_width            = 1200
screen_height           = 900
menu_active             = False
# Colors
white                   = (255, 255, 255)
color_black             = (0, 0, 0)
sky_blue                = (135, 206, 235)
color_crimson           = (220,20,60)
color_greenforest       = (34,139,34)
color_blue              = (50, 50, 255)
banner_color            = (0, 0, 255)
"""
