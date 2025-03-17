import time
from src.neuroForge_original import mgr
import tkinter.messagebox as mb

class VCR:
    def __init__(self):
        self.vcr_rate = 5  # Speed of playback (0 = paused, ±1, ±2, etc.)
        self.direction = 1  # 1= Forward, 2 = reverse
        self.last_update_time = time.monotonic()
        self.status = "Playing"

    def play(self):
        self.status = "Playing"

    def pause(self):
        self.status = "Paused"

    def reverse(self):
        print(f"self.direction before ={self.direction}")
        self.direction *= -1    #reverse direction
        print(f"self.direction aftre ={self.direction}")
        self.play()

    def set_speed(self, speed: int):
        """Set playback speed (positive = forward, negative = reverse, 0 = pause)."""
        self.vcr_rate = speed
        self.play()

    def jump_to_epoch(self, epoch_str: str):
        """
        Handles user input for jumping to a specific epoch.

        Steps:
        1) Validates that `epoch_str` is a number.
        2) Checks if the number is within the valid epoch range (1 to mgr.max_epoch).
        3) If valid, updates `mgr.epoch` and resets `mgr.iteration` to 1.
        4) If invalid, prints/logs a message but does not update the epoch.

        Args:
            epoch_str (str): User-inputted epoch number.
        """
        try:
            epoch = int(epoch_str)  # ✅ Convert input to an integer
            if 1 <= epoch <= mgr.max_epoch:
                mgr.epoch = epoch
                mgr.iteration = 1  # Reset iteration when jumping

                #print(f"✅ Jumped to Epoch {epoch}")
            else:
                self.pause()
                mb.showinfo("⚠️ Epoch out of range!",f"Must be between 1 and {mgr.max_epoch}.")
                print(f"⚠️ Epoch out of range! Must be between 1 and {mgr.max_epoch}\nClick 'Play' to resume.")
        except ValueError:
            self.pause()
            mb.showinfo("⚠️ Invalid input!","Please enter a valid epoch number.\nClick 'Play' to resume.")


    def step_x_iteration(self, direction: int):
        """
        Move one iteration either direction

        Args:
            direction (int): 1 for forward -1 for back
        """
        mgr.iteration += direction  #1 or -1
        self.validate_epoch()

    def step_x_epochs(self, direction: int):
            """
            Move one iteration either direction

            Args:
                direction (int): 1 for forward -1 for back
            """

            mgr.epoch += direction  #1 or -1
            self.validate_epoch()


    def play_the_tape(self): #Handles auto-play when VNR is running.
        if self.status == "Playing":
            current_time = time.monotonic()
            seconds_per_frame = 1.0 / abs(self.vcr_rate)
            if current_time - self.last_update_time >= seconds_per_frame:
                #self.step(1 if self.vcr_rate > 0 else -1)
                self.switch_frame()
                self.last_update_time = current_time



    def switch_frame(self):
        if self.status != "Playing":
            return
        if self.direction > 0:
            mgr.iteration += 1
        else:
            mgr.iteration -= 1
        self.validate_epoch()

    def validate_epoch(self):        #print(f"validating- mgr.epoch: {mgr.epoch}\tmgr.max_epoch: {mgr.max_epoch}\tmgr.epoch: {mgr.iteration}\tmgr.iteration: {mgr.epoch}\t")
        #Is it past the end?
        if (mgr.epoch == mgr.max_epoch and mgr.iteration> mgr.max_iteration) or mgr.epoch > mgr.max_epoch :
            mgr.epoch = mgr.max_epoch               #Set it to the end
            mgr.iteration = mgr.max_iteration       #Set it to the end
            print("pausing")
            self.pause()

        #Is it in front of the beginning?
        if mgr.epoch <=0:
            mgr.epoch = 1
            mgr.iteration = 1
            self.pause()

        # flip epoch
        if mgr.iteration > mgr.max_iteration:
            mgr.epoch = min(mgr.epoch + 1, mgr.max_epoch)
            mgr.iteration = 1
        if mgr.iteration == 0:
            mgr.epoch = max(1, mgr.epoch - 1)
            mgr.iteration = mgr.max_iteration
