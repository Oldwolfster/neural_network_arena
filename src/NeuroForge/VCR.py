import time
from bisect import bisect_left

from src.NeuroForge import Const
import tkinter.messagebox as mb

class VCR:
    def __init__(self):
        """Initialize VCR playback control."""
        self.vcr_rate = 5  # Speed of playback (0 = paused, ±1, ±2, etc.)
        self.direction = 1  # 1 = Forward, -1 = Reverse
        self.advance_by_epoch = 1
        self.last_update_time = time.monotonic()
        self.status = "Playing"  # Default to paused
        self.recorded_frames = []  # filled from dispaly manager
        self._cur_epoch = 1             #Moved here when we stopped recording every frame
        self._cur_iteration = 1         #Moved here when we stopped recording every frame

    def get_nearest_frame(self, requested_epoch, requested_iter):
            requested = (requested_epoch, requested_iter)
            frames = self.recorded_frames

            if not frames:
                return requested  # fallback: nothing to snap to

            if requested in frames:
                return requested

            # Binary search for the first frame > requested
            idx = bisect_left(frames, requested)

            if idx < len(frames):    # found one after
                return frames[idx]
            elif idx > 0:            # fallback to one before
                return frames[idx - 1]
            else:                    # no frames at all
                return frames[0]

    @property
    def CUR_EPOCH_MASTER(self):
        return self._cur_epoch

    @CUR_EPOCH_MASTER.setter
    def CUR_EPOCH_MASTER(self, val):
        e, i = self.get_nearest_frame(val, self._cur_iteration)
        self._cur_epoch = e
        self._cur_iteration = i

    @property
    def CUR_ITERATION(self):
        return self._cur_iteration

    @property
    def now(self):
        return f"Epoch:{self._cur_epoch} Sample:{self._cur_iteration}"


    @CUR_ITERATION.setter
    def CUR_ITERATION(self, val):
        e, i = self.get_nearest_frame(self._cur_epoch, val)
        self._cur_epoch = e
        self._cur_iteration = i



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
                Const.vcr.CUR_EPOCH = epoch
                Const.vcr.CUR_ITERATION = 1  # Reset iteration when jumping
            else:
                self.pause()
                mb.showinfo("⚠️ Epoch out of range!", f"Must be between 1 and {Const.MAX_EPOCH}.")
        except ValueError:
            self.pause()
            mb.showinfo("⚠️ Invalid input!", "Please enter a valid epoch number.")

    def step_x_iteration(self, step: int, pause_me=False):
        """Move a specified number of iterations forward or backward."""
        if pause_me:
            self.pause()

        # Normalize to previous/next epoch if needed
        if step < 0 and Const.vcr.CUR_ITERATION == 1:
            if Const.vcr.CUR_EPOCH_MASTER > 1:
                Const.vcr.CUR_EPOCH_MASTER -= 1
                Const.vcr.CUR_ITERATION = Const.MAX_ITERATION
            else:
                return  # Already at start
        elif step > 0 and Const.vcr.CUR_ITERATION == Const.MAX_ITERATION:
            if Const.vcr.CUR_EPOCH_MASTER < Const.MAX_EPOCH:
                Const.vcr.CUR_EPOCH_MASTER += 1
                Const.vcr.CUR_ITERATION = 1
            else:
                return  # Already at end
        else:
            Const.vcr.CUR_ITERATION += step

        # Snap to nearest real frame (some might be skipped)
        snapped_epoch, snapped_iter = self.get_nearest_frame(Const.vcr.CUR_EPOCH_MASTER, Const.vcr.CUR_ITERATION)
        Const.vcr.CUR_EPOCH_MASTER = snapped_epoch
        Const.vcr.CUR_ITERATION = snapped_iter

        self.validate_epoch_or_iteration_change_and_sync_data()


    def step_x_iterationorig(self, step: int, pause_me = False):
        """Move a specified number of iterations forward or backward."""
        if pause_me:
            self.pause()
        # Check if trying to move past end
        if step > 0 and Const.vcr.CUR_ITERATION == Const.MAX_ITERATION and Const.vcr.CUR_EPOCH_MASTER == Const.MAX_EPOCH:
            return
        # Check if trying to move before start
        if step < 0 and Const.vcr.CUR_ITERATION == 1 and Const.vcr.CUR_EPOCH_MASTER == 1:
            return
        Const.vcr.CUR_ITERATION += step
        self.validate_epoch_or_iteration_change_and_sync_data()

    def step_x_epochs(self, step: int, pause_me = False):
        """Move a specified number of epochs forward or backward."""
        if pause_me:
            self.pause()


        Const.vcr.CUR_EPOCH_MASTER += step
        #Const.vcr.CUR_ITERATION = 1  # Reset iteration when jumping epochs
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
        if self.advance_by_epoch == 1:
            self.step_x_epochs(self.direction)
        else:
            self.step_x_iteration(self.direction)
        self.validate_epoch_or_iteration_change_and_sync_data()
        # Fetch the latest iteration data after stepping
        #Const.dm.get_iteration_dict()

    def validate_epoch_or_iteration_change_and_sync_data(self):
        """Ensure epoch and iteration values stay within valid bounds."""
        if Const.vcr.CUR_EPOCH_MASTER > Const.MAX_EPOCH:
            Const.vcr.CUR_EPOCH_MASTER = Const.MAX_EPOCH
            Const.vcr.CUR_ITERATION = Const.MAX_ITERATION
            self.pause()

        if Const.vcr.CUR_EPOCH_MASTER < 1:
            Const.vcr.CUR_EPOCH_MASTER = 1
            Const.vcr.CUR_ITERATION = 1
            self.pause()

        if Const.vcr.CUR_ITERATION > Const.MAX_ITERATION:
            Const.vcr.CUR_EPOCH_MASTER = min(Const.vcr.CUR_EPOCH_MASTER + 1, Const.MAX_EPOCH)
            Const.vcr.CUR_ITERATION = 1

        if Const.vcr.CUR_ITERATION < 1:
            Const.vcr.CUR_EPOCH_MASTER = max(1, Const.vcr.CUR_EPOCH_MASTER - 1)
            Const.vcr.CUR_ITERATION = Const.MAX_ITERATION

        # Fetch the latest iteration data after stepping
        Const.dm.query_dict_iteration()
        Const.dm.query_dict_epoch()
