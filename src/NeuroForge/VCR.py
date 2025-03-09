import time
from src.NeuroForge import Const
import tkinter.messagebox as mb

class VCR:
    def __init__(self):
        """Initialize VCR playback control."""
        self.vcr_rate = 5  # Speed of playback (0 = paused, ±1, ±2, etc.)
        self.direction = 1  # 1 = Forward, -1 = Reverse
        self.last_update_time = time.monotonic()
        self.status = "Playing"  # Default to paused
        Const.CUR_ITERATION = Const.MAX_ITERATION

    def play(self):
        """Start playback."""
        self.status = "Playing"

    def pause(self):
        """Pause playback."""
        self.status = "Paused"

    def toggle_play_pause(self):
        """Toggle between play and pause modes."""
        if self.status == "Playing":
            self.pause()
        else:
            self.play()

    def reverse(self):
        """Reverse playback direction and start playing."""
        self.direction *= -1
        self.play()

    def set_speed(self, speed: int):
        """Set playback speed (positive = forward, negative = reverse, 0 = pause)."""
        self.vcr_rate = abs(speed) * self.direction
        if speed == 0:
            self.pause()
        else:
            self.play()

    def jump_to_epoch(self, epoch_str: str):
        """Handles user input for jumping to a specific epoch."""
        try:
            epoch = int(epoch_str)
            if 1 <= epoch <= Const.MAX_EPOCH:
                Const.CUR_EPOCH = epoch
                Const.CUR_ITERATION = 1  # Reset iteration when jumping
            else:
                self.pause()
                mb.showinfo("⚠️ Epoch out of range!", f"Must be between 1 and {Const.MAX_EPOCH}.")
        except ValueError:
            self.pause()
            mb.showinfo("⚠️ Invalid input!", "Please enter a valid epoch number.")

    def step_x_iteration(self, step: int):
        """Move a specified number of iterations forward or backward."""
        Const.CUR_ITERATION += step
        self.validate_epoch_or_iteration_change_and_sync_data()

    def step_x_epochs(self, step: int):
        """Move a specified number of epochs forward or backward."""
        Const.CUR_EPOCH += step
        #Const.CUR_ITERATION = 1  # Reset iteration when jumping epochs
        self.validate_epoch_or_iteration_change_and_sync_data()

    def play_the_tape(self):
        """Handles auto-play when VCR is running."""

        if self.status == "Playing":
            current_time = time.monotonic()
            seconds_per_frame = 1.0 / abs(self.vcr_rate)
            if current_time - self.last_update_time >= seconds_per_frame:
                self.switch_frame()
                self.last_update_time = current_time

    def switch_frame(self):
        """Advance or reverse a frame based on playback direction."""
        if self.status != "Playing":
            return
        self.step_x_epochs(self.direction)
        self.validate_epoch_or_iteration_change_and_sync_data()
        # Fetch the latest iteration data after stepping
        #Const.dm.get_iteration_dict()

    def validate_epoch_or_iteration_change_and_sync_data(self):
        """Ensure epoch and iteration values stay within valid bounds."""
        if Const.CUR_EPOCH > Const.MAX_EPOCH:
            Const.CUR_EPOCH = Const.MAX_EPOCH
            Const.CUR_ITERATION = Const.MAX_ITERATION
            self.pause()

        if Const.CUR_EPOCH < 1:
            Const.CUR_EPOCH = 1
            Const.CUR_ITERATION = 1
            self.pause()

        if Const.CUR_ITERATION > Const.MAX_ITERATION:
            Const.CUR_EPOCH = min(Const.CUR_EPOCH + 1, Const.MAX_EPOCH)
            Const.CUR_ITERATION = 1

        if Const.CUR_ITERATION < 1:
            Const.CUR_EPOCH = max(1, Const.CUR_EPOCH - 1)
            Const.CUR_ITERATION = Const.MAX_ITERATION

        # Fetch the latest iteration data after stepping
        Const.dm.query_dict_iteration()
        Const.dm.query_dict_epoch()
