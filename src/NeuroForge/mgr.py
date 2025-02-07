import tkinter.messagebox as mb
from datetime import time

# mgr.py
# manages resources such as variables, fonts, etc.


#Pygame objects
screen          = None
tool_tip        = None
jump_to_epoch   = 0
color_neurons   = True
font            = None
time_scale      = 'Iteration'
epoch           = 1
iteration       = 1
max_epoch       = 0
max_iteration   = 0
max_weight      = 0
max_error       = 0
layer           = 1
neuron          = 1
display_models  = []

vcr_rate       = 1  #0 means paused
vcr_direction   = 1

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

def validate_epoch_change():
    global epoch
    global jump_to_epoch
    global iteration

    #Check if epoch jump text box was used
    if jump_to_epoch !=0:
        epoch = jump_to_epoch
        jump_to_epoch = 0
    #print (f"in epoch validate: iteration = {iteration}")

    if iteration > max_iteration:
        epoch += 1
        iteration = 1


    # Check for out of range conditions
    if epoch == -1:
        mb.showinfo("Invalid input!", "Please enter a number!")
    if epoch < 1:  # Check if trying to move past the beginning
        mb.showinfo("Out of Range", "You cannot go past the first epoch!")
        epoch = 1
        iteration = 1

    if epoch > max_epoch:  # Check if trying to move past the end
        #DELETE MEmgr.scheduler.schedule("vcr", mgr.vcr_rate ,0)
        mb.showinfo("Out of Range", "You are at the end!")
        epoch = max_epoch
        iteration = max_iteration

    #print (f"end of epoch validate iteration={iteration}")
