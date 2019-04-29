import random

from collections import deque


class Memory():
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def add(self, record):
        self.memory.append(record)

    def sample(self, n):
        return random.sample(self.memory, n)
