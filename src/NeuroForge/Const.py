from typing import List
from typing import TYPE_CHECKING
#from src.NeuroForge import Display_Manager
from src.engine.Config import Config
from copy import deepcopy
from src.Legos.Scalers import *

# ==============================
# Global References
# ==============================
configs: List[Config] = []
if TYPE_CHECKING:
    from src.NeuroForge.Display_Manager import DisplayManager
dm: "DisplayManager" = None  # Lazy reference to avoid circular imports

if TYPE_CHECKING:
    from src.NeuroForge.VCR             import VCR
vcr: "VCR" = None


# ==============================
# UI Constants
# ==============================
SCREEN_WIDTH  = 1900
SCREEN_HEIGHT = 900 #900

MODEL_AREA_PERCENT_LEFT     = 0.10
MODEL_AREA_PERCENT_TOP      = 0.05
MODEL_AREA_PERCENT_WIDTH    = 0.80
MODEL_AREA_PERCENT_HEIGHT   = 0.91
MODEL_AREA_PIXELS_LEFT      = SCREEN_WIDTH * MODEL_AREA_PERCENT_LEFT
MODEL_AREA_PIXELS_TOP       = SCREEN_HEIGHT * MODEL_AREA_PERCENT_TOP
MODEL_AREA_PIXELS_WIDTH     = SCREEN_WIDTH * MODEL_AREA_PERCENT_WIDTH
MODEL_AREA_PIXELS_HEIGHT    = SCREEN_HEIGHT * MODEL_AREA_PERCENT_HEIGHT

MENU_ACTIVE                 = False
IS_RUNNING                  = True


# ==============================
# Training State
# ==============================
"""
def set_vcr_instance(instance):
    global vcr
    vcr = instance

def get_CUR_EPOCH():
    return vcr.epoch

def set_CUR_EPOCH(val):
    vcr.epoch = val

def get_CUR_ITERATION():
    return vcr.iteration

def set_CUR_ITERATION(val):
    vcr.iteration = val

CUR_EPOCH = property(get_CUR_EPOCH, set_CUR_EPOCH)
CUR_ITERATION = property(get_CUR_ITERATION, set_CUR_ITERATION)
"""


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
# Popup Const
# ==============================
TOOLTIP_WIDTH_MAX   = 1369 #969 #669
TOOLTIP_HEIGHT_MAX  = 869
TOOLTIP_WIDTH       = 1169 #969 #669
TOOLTIP_HEIGHT      = 469
TOOLTIP_PLACEMENT_X =  10
TOOLTIP_PLACEMENT_Y =  10
TOOLTIP_PADDING     =   5
TOOLTIP_FONT_TITLE  =  40
TOOLTIP_FONT_HEADER =  26
TOOLTIP_FONT_BODY   =  22
TOOLTIP_COL_WIDTH   =  60  # ✅ Standardized column width
TOOLTIP_ROW_HEIGHT  =  20  # ✅ Standardized row height
TOOLTIP_HEADER_PAD  =  39  # ✅ Consistent header spacing
TOOLTIP_COND_COLUMN =   7
TOOLTIP_ADJUST_PAD  =  20

# ==============================
# Popup Divider Line Consts
# ==============================
TOOLTIP_LINE_BEFORE_BACKPROP       = 6    # After forward prop ends
TOOLTIP_LINE_AFTER_ADJUST          = 15   # After orig/new before blame calc
#TOOLTIP_LINE_BEFORE_ACTIVATION     = 6    # Before Act Gradient in fwd pass
TOOLTIP_LINE_OVER_HEADER_Y        = 0   # Y position under header row
TOOLTIP_HEADER_DIVIDER_THICKNESS   = 2
TOOLTIP_COLUMN_DIVIDER_THICKNESS   = 1



# ==============================
# UI Customization
# ==============================
JUMP_TO_EPOCH       = 0
FONT_SIZE_WEIGHT    = 24
FONT_SIZE_SMALL     = 20
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
COLOR_GRAY_DIM          = (105, 105, 105)
COLOR_GRAY_DARK         = (64, 64, 64)

COLOR_GREEN             = (0, 128, 0)
COLOR_GREEN_FOREST      = (34, 139, 34)
COLOR_GREEN_JADE        = (60, 179, 113)
COLOR_GREEN_KELLY       = (34, 170, 34)
COLOR_RED_FIREBRICK     = (178, 34, 34)
COLOR_RED_BURGUNDY      = (139,   0, 0)
COLOR_ORANGE            = (255, 165, 0)
COLOR_YELLOW_BRIGHT     = (255, 215, 0)
COLOR_WHITE             = (255, 255, 255)
COLOR_CREAM             = (255, 255, 200)

#Below is Colors  by Purpose rather than color name.
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
COLOR_FOR_BAR1_POSITIVE  = COLOR_GREEN_KELLY
COLOR_FOR_BAR1_NEGATIVE  = COLOR_RED_FIREBRICK
COLOR_FOR_BAR2_POSITIVE = COLOR_GREEN_JADE
COLOR_FOR_BAR2_NEGATIVE = COLOR_RED_BURGUNDY
COLOR_eh             = (220, 255, 220)
COLOR_HIGHLIGHT_FILL    = COLOR_eh
COLOR_HIGHLIGHT_BORDER  = (218, 165, 32)

#Moved here to avoid clutter in NeuroForge opening.
def add_items_to_architecture_not_in_NNA(configs: List[Config]):
    return # Trying different approach
    for config in configs:  #Add scalers, thresholds, and anything else to architecture
        config.architecture_core = deepcopy(config.architecture)
        if config.input_scaler != Scaler_NONE:
            config.architecture.insert(0, 1) # Add one to front of architecture [2,2,1 ] becomes [1,2,2,1]
        
