import time
from transformers.generation.streamers import BaseStreamer

class TTFTStreamer(BaseStreamer):
    def __init__(self):
        self.start_time = time.time()
        self.first_token_time = None
        self.ttft = None

    def put(self, value):
        if self.first_token_time is None:
            self.first_token_time = time.time()
            self.ttft = self.first_token_time - self.start_time

    def end(self):
        pass
