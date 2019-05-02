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

    def sample(self, n, is_special_sample, special_percentage=0.5):
        #print('n:', n)
        #print('is_special_sample:', is_special_sample)
        #print('special_percentage:', special_percentage)
        #print('self.memory:', self.memory)

        if is_special_sample:
            # get terminal state data (those with reward != 0)
            index = np.array(self.memory)[:, 2] != 0
            terminal_state_data = np.array(self.memory)[index]
            #print('\nterminal_state_data:', terminal_state_data)

            # determinal number of terminal state get and normal data get
            special_num = int(n * special_percentage)
            if len(terminal_state_data) < special_num:
                special_num = len(terminal_state_data)
            normal_num = n - special_num

            #print('special_num:',special_num)
            #print('normal_num:',normal_num)

            #print('random.sample(list(terminal_state_data), special_num).shape:',np.array(random.sample(list(terminal_state_data), special_num)).shape)
            #print('\nrandom.sample(self.memory, normal_num).shape:',np.array(random.sample(self.memory, normal_num)).shape)

            train_data =  np.concatenate(
                (random.sample(list(terminal_state_data), special_num),
                 random.sample(self.memory, normal_num))
                 )

            #print('train_data.')

            # use sample will shuffle the data
            # if use np.random.shuffle, it does inplace shuffle and return None
            return random.sample(list(train_data), len(train_data))
        else:
            return random.sample(self.memory, n)

    def get_memory_size(self):
        return len(self.memory)
