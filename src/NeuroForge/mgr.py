import tkinter.messagebox as mb
# manages resources such as variables, fonts, etc.

#Epoch and VCR
VCR                     = None
max_epoch       :int    = 0
max_iteration   :int    = 0
max_weight      :float  = 0
max_error       :float  = 0
epoch           :int    = 1
iteration       :int    = 1
epoch_error     :float  = 0
summarized_epoch :int    = 1
error       :float  = 0
loss       :float  = 0
loss_grd   :float  = 0
avg_error       :float  = 0.0
avg_loss       :float  = 0
avg_loss_grd   :float  = 0
#Pygame objects
screen          = None
tool_tip        = None
jump_to_epoch   = 0
color_neurons   = True
font            = None
layerDeleteMe           = 1
neuronDeleteMe          = 1
display_models  = []
screen_width=1200
screen_height=900
menu_active = False
# Colors
white           = (255, 255, 255)
color_black         = (0, 0, 0)
sky_blue            = (135, 206, 235)
color_crimson       = (220,20,60) #crimson
color_greenforest   = (34,139,34) #forest Green
color_blue = (50, 50, 255)
banner_color = (0,0,255)

