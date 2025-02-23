import time

import pygame

from src.NeuroForge import Const
import tkinter.messagebox as mb


class VCR:
    def __init__(self):
        self.vcr_rate = 5  # Speed of playback (0 = paused, ±1, ±2, etc.)
        self.direction = 1  # 1= Forward, 2 = reverse
        self.last_update_time = time.monotonic()
        self.status = "Playing"

    def process_event(self, event: pygame.event):
        return True
