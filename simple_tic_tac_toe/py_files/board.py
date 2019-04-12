import numpy as np


class Board():

    def __init__(self, win_reward, lose_reward, draw_reward):
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.draw_reward = draw_reward
        self.reset()

    def reset(self):
        self.state = np.zeros(9)
        self.observation = np.zeros(9)
        self.active_player_id = 1

    #---------------------update_state starts-------------------------------
    def check_if_terminal_state(self, state=None):
        # if given state, determine whether the given state is a terminal state,
        # else check whether self.state is a terminal state

        if state is None:
            state = self.state

        check_points = [[0,1,2],[3,4,5],[6,7,8],
                        [0,3,6],[1,4,7],[2,5,8],
                        [0,4,8],[2,4,6]]

        # if player 1 or -1 wins
        for check_point in check_points:
            series = state[check_point]
            if abs(series.sum()) == 3:
                return [True, series[0]]

        # if both player do not win and there is empty space
        # is_terminal_state = False
        for cell in state:
            if cell == 0:
                return [False, None]

        # if both player do not win and there is no empty space, draw
        return [True, 0]

    def determine_reward(self, is_terminal_state, winner):
        if winner == 1:
            return [self.win_reward, self.lose_reward]
        elif winner == -1:
            return [self.lose_reward, self.win_reward]
        elif is_terminal_state:
            return [draw_reward, draw_reward]
        else:
            return [0, 0]

    def update_observation_from_state(self):
        # update observation_board as the state_board
        self.observation = self.state
        return True

    def get_observation(self):
        return self.observation

    def update_state(self, active_player_id, action):
        """given a valid action and the current active player id
        1. update the internal state
        2. return new_observation, reward, is_terminal_state
        """
        assert(self.state[action] == 0)

        # update the state
        self.state[action] = active_player_id

        # check winning condition
        # winner = [1, -1, 0, None], 0 means draw, None means game not yet finished
        is_terminal_state, winner = self.check_if_terminal_state()

        # calculate reward
        reward = self.determine_reward(is_terminal_state, winner)

        # update observation(views to the player), in this case state and
        # observations will be the same
        self.update_observation_from_state()

        new_observation = self.get_observation()

        return new_observation, reward, is_terminal_state

    #------------------update_state ends----------------------------------

    #------------------step starts----------------------------------

    def get_active_player_id(self):
        """ return the id of the active player (1 or -1)
        """
        return self.active_player_id

    def swap_active_player_id(self):
        """ swap the id of the active player (1 or -1)
        """
        self.active_player_id *= -1
        return True

    def is_valid_action(self, action):
        # if that cell is empty, return True, else return False
        return self.state[action] == 0

    def step(self, action):
        """
        1. check if the action is valid, if False,return
        2. assume the action is valid, update the state according to the action
        3. swap the active player id
        4. return the new_observation, reward, is_terminal_state
        """
        active_player_id = self.get_active_player_id()

        if not self.is_valid_action(action):
            print('Error in step')
            return False, False, False

        new_observation, reward, is_terminal_state = self.update_state(active_player_id, action)

        self.swap_active_player_id()

        return new_observation, reward, is_terminal_state

    #------------------step ends----------------------------------

    def print_board(self):
        board = self.state

        symbol = ['O' if board[i] == 1 else ('X' if board[i] == -1 else ' ')
                  for i in range(9)]

        print('Board:\n')
        print('%s | %s | %s' % (symbol[0], symbol[1], symbol[2]))
        print('---------')
        print('%s | %s | %s' % (symbol[3], symbol[4], symbol[5]))
        print('---------')
        print('%s | %s | %s' % (symbol[6], symbol[7], symbol[8]))

        return True
