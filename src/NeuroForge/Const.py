from typing import List
from typing import TYPE_CHECKING
#from src.NeuroForge import Display_Manager
from src.engine.ModelConfig import ModelConfig

# ==============================
# Global References
# ==============================
configs: List[ModelConfig] = []
if TYPE_CHECKING:
    from src.NeuroForge.Display_Manager import DisplayManager
dm: "DisplayManager" = None  # Lazy reference to avoid circular imports

if TYPE_CHECKING:
    from src.NeuroForge.VCR             import VCR
vcr: "VCR" = None


# ==============================
# UI Constants
# ==============================
SCREEN_WIDTH  = 1200
SCREEN_HEIGHT = 900

MODEL_AREA_PERCENT_LEFT     = 0.15
MODEL_AREA_PERCENT_TOP      = 0.05
MODEL_AREA_PERCENT_WIDTH    = 0.70
MODEL_AREA_PERCENT_HEIGHT   = 0.91
MODEL_AREA_PIXELS_LEFT      = SCREEN_WIDTH * MODEL_AREA_PERCENT_LEFT
MODEL_AREA_PIXELS_TOP       = SCREEN_HEIGHT * MODEL_AREA_PERCENT_TOP
MODEL_AREA_PIXELS_WIDTH     = SCREEN_WIDTH * MODEL_AREA_PERCENT_WIDTH
MODEL_AREA_PIXELS_HEIGHT    = SCREEN_HEIGHT * MODEL_AREA_PERCENT_HEIGHT

MENU_ACTIVE   = False

# ==============================
# Training State
# ==============================
MAX_EPOCH       = 0
CUR_EPOCH       = 1
MAX_ITERATION   = 0
CUR_ITERATION   = 1
MAX_WEIGHT      = 0.0
MAX_ACTIVATION  = 0.0
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
JUMP_TO_EPOCH       = 0
FONT_SIZE_WEIGHT    = 20
#COLOR_NEURONS  = True

# ==============================
# Colors
# ==============================
COLOR_BLACK             = (0, 0, 0)
COLOR_BLUE              = (50, 50, 255)
COLOR_BLUE_PURE         = (0, 0, 255)
COLOR_BLUE_MIDNIGHT     = (25, 25, 112)
COLOR_BLUE_STEEL        = (70, 130, 180)
COLOR_BLUE_SKY          = (135, 206, 235)
COLOR_CRIMSON           = (220, 20, 60)
COLOR_CYAN              = (0, 255, 255)
COLOR_GREEN             = (0, 128, 0)
COLOR_GREEN_FOREST      = (34, 139, 34)
COLOR_ORANGE            = (255, 165, 0)
COLOR_YELLOW_BRIGHT     = (255, 215, 0)
COLOR_WHITE             = (255, 255, 255)

COLOR_FOR_BANNER        = (0, 0, 255)
COLOR_FOR_SHADOW        = (30, 30, 100)  # Darker blue for depth
COLOR_FOR_BACKGROUND    = COLOR_WHITE
COLOR_FOR_BANNER_START  = COLOR_BLUE_MIDNIGHT
COLOR_FOR_BANNER_END    = COLOR_BLUE_STEEL
COLOR_FOR_NEURON_BODY   = COLOR_BLUE_PURE
COLOR_FOR_NEURON_TEXT   = COLOR_WHITE
COLOR_FOR_BAR_GLOBAL    = COLOR_ORANGE
COLOR_FOR_BAR_SELF      = COLOR_GREEN
COLOR_FOR_ACT_POSITIVE  = COLOR_GREEN
COLOR_FOR_ACT_NEGATIVE  = COLOR_CRIMSON
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
