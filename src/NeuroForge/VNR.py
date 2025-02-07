import time

class VNR_Controller:
    def __init__(self):
        self.vcr_rate = 0  # Speed of playback (0 = paused, ±1, ±2, etc.)
        self.last_update_time = time.monotonic()

    def set_speed(self, speed: int):
        """Set playback speed (positive = forward, negative = reverse, 0 = pause)."""
        self.vcr_rate = speed

    def jump_to_epoch(self, epoch: int):
        """Jump to a specific epoch and reset iteration to 1."""
        mgr.epoch = max(1, min(epoch, mgr.max_epoch))
        mgr.iteration = 1
        print(f"Jumped to Epoch {mgr.epoch}")

    def step(self, direction: int):
        """Manually step forward (+1) or backward (-1)."""
        if direction > 0:
            mgr.iteration += 1
            if mgr.iteration > mgr.max_iteration:
                mgr.epoch = min(mgr.epoch + 1, mgr.max_epoch)
                mgr.iteration = 1
        else:
            mgr.iteration -= 1
            if mgr.iteration == 0:
                mgr.epoch = max(1, mgr.epoch - 1)
                mgr.iteration = mgr.max_iteration
        print(f"Stepped to Epoch {mgr.epoch}, Iteration {mgr.iteration}")

    def update(self):
        """Handles auto-play when VNR is running."""
        if self.vcr_rate == 0:
            return  # Paused

        current_time = time.monotonic()
        seconds_per_frame = 1.0 / abs(self.vcr_rate)

        if current_time - self.last_update_time >= seconds_per_frame:
            self.step(1 if self.vcr_rate > 0 else -1)
            self.last_update_time = current_time
