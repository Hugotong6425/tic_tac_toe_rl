import random

import numpy as np

from collections import deque


class Memory():
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def add(self, record):
        train_data = [record['observation'], record['action'], record['reward'],
                      record['next_observation'], record['done']]
        self.memory.append(train_data)

    def sample(self, n):
        return random.sample(self.memory, n)

    def get_memory_size(self):
        return len(self.memory)
