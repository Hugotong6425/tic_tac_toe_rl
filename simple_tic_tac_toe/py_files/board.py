from copy import deepcopy
import numpy as np

from .player import Player, Human, Random_player, Q_player


class Board():
    """
    Attribute:
        - self.p1, Class Player
        - self.p2, Class Player
        - self.win_reward, float
        - self.lose_reward, float
        - self.draw_reward, float
        - self.state = np array, size [9]
        - self.observation = np array, size [9] (same as state now)
        - self.active_player_id, int, 1 or -1
        - self.is_terminal_state, bool
    """
    def __init__(self, p1_config=None, p2_config=None, win_reward=1,
                 lose_reward=-1, draw_reward=0):
        self.p1 = self.set_up_player(p1_config)
        self.p2 = self.set_up_player(p2_config)
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.draw_reward = draw_reward
        self.winner = 0

        # assign player id to both players
        # only player 1 could be the agent that accept training
        self.p1.set_player_id(player_id=1)
        self.p2.set_player_id(player_id=-1)

        self.reset()

    def set_up_player(self, player_config):
        """
        set up self.p1
        """
        if player_config is None:
            return None
        player_type = player_config['player_type']

        if player_type == 'human':
            player_name = player_config.get('player_name', 'human player')
            return Human(player_name)
        elif player_type == 'random':
            player_name = player_config.get('player_name', 'random player')
            return Random_player(player_name)
        elif player_type == 'q_player':
            # load in all config and set the default value if not exist
            player_name = player_config.get('player_name', 'q player')
            hidden_layers_size = player_config.get('hidden_layers_size', [20,10])
            batch_size = player_config.get('batch_size', 64)
            learning_rate = player_config.get('learning_rate', 0.001)
            optimizer = player_config.get('optimizer', 'adam')
            loss = player_config.get('loss', 'mse')
            load_trained_model_path = player_config.get('load_trained_model_path', None)
            is_train = player_config.get('is_train', True)
            ini_epsilon = player_config.get('ini_epsilon', 1.0)
            is_double_dqns = player_config.get('is_double_dqns', True)
            epsilon_decay = player_config.get('epsilon_decay', 0.9995)
            epsilon_min = player_config.get('epsilon_min', 0.01)
            gamma = player_config.get('gamma', 0.99)

            return Q_player(hidden_layers_size=hidden_layers_size,
                            batch_size=batch_size, learning_rate=learning_rate,
                            load_trained_model_path=load_trained_model_path,
                            is_train=is_train, is_double_dqns=is_double_dqns,
                            epsilon_decay=epsilon_decay, loss=loss,
                            player_name=player_name, ini_epsilon=ini_epsilon,
                            epsilon_min=epsilon_min, gamma=gamma)

    def reset(self):
        """ reset the board
        should run this when a new episode starts,
        the starting player is default to be player id 1
        """
        self.state = np.zeros(9)
        self.observation = np.zeros(9)
        self.active_player_id = 1
        self.is_terminal_state = False

    def pick_start_player_id(self, who_first=''):
        """  choose the starting player
        run this after def reset()
        """
        if who_first == '':
            self.active_player_id = np.random.choice([1, -1])
        else:
            self.active_player_id = who_first

    #---------------------update_state starts-------------------------------
    def check_if_terminal_state(self, state=None):
        """ check if terminal state is reached and return the winner
        if state is not None, determine whether the given state is a terminal state,
        else check whether self.state is a terminal state

        return [bool, winner]
        """
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
        """ determine both players' reward
        given is_terminal_state (bool) and winner ([1, -1]),
        return [p1_reward, p2_reward]
        """
        if winner == 1:
            return np.asarray([self.win_reward, self.lose_reward])
        elif winner == -1:
            return np.asarray([self.lose_reward, self.win_reward])
        elif is_terminal_state:
            return np.asarray([self.draw_reward, self.draw_reward])
        else:
            return np.asarray([0.0, 0.0])

    def update_observation_from_state(self):
        # update observation_board as the state_board
        self.observation = self.state
        return True

    def get_observation(self):
        return self.observation

    def update_state(self, active_player_id, action):
        """ given a valid action and the current active player id
        1. update the internal state
        2. return new_observation, reward, is_terminal_state
        """
        assert(self.state[action] == 0)

        # update self.state
        self.state[action] = active_player_id

        # check winning condition
        # winner = [1, -1, 0, None], 0 means draw, None means game not yet finished
        is_terminal_state, winner = self.check_if_terminal_state()

        self.winner = winner

        # update self.is_terminal_state
        self.is_terminal_state = is_terminal_state

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
        """ print out the board
        """
        board = self.observation

        symbol = ['O' if board[i] == 1 else ('X' if board[i] == -1 else ' ')
                  for i in range(9)]

        print('\n--------------------------------------------------\n')

        # "current player" is actually the last active player
        print('current player id: %s \n'  % (self.get_active_player_id()*-1))
        print('current player name: %s \n'  % (self.get_active_player().get_player_name()))

        print('Board:\n')
        print('%s | %s | %s' % (symbol[0], symbol[1], symbol[2]))
        print('---------')
        print('%s | %s | %s' % (symbol[3], symbol[4], symbol[5]))
        print('---------')
        print('%s | %s | %s\n' % (symbol[6], symbol[7], symbol[8]))

        print('is_terminal_state: \n%s' % self.is_terminal_state)

        if self.is_terminal_state:
            print('\nGame Over!')

        return True

    def get_action_availability(self):
        """ return the availability of each action
        return a np array with size [9],
        1 means that action is available, 0 means the opposite
        """
        return np.asarray([1 if cell == 0 else 0 for cell in self.state])

    def get_player(self, player_id):
        if player_id == 1:
            return self.p1
        elif player_id == -1:
            return self.p2

    def get_active_player(self):
        """ return active player (not its id)
        """
        active_id = self.active_player_id
        if active_id == 1:
            return self.p1
        elif active_id == -1:
            return self.p2
        else:
            print('ERROR, self.active_player_id = %s.\n' % self.active_id)
            return None

    def report_stat(self, epi, loss, win_cnt, draw_cnt, lose_cnt, games_played):
        if epi is not None:
            print('Current episode: %s' % epi)
        if loss is not None:
            print('Loss: %s' % loss)
        print('p1 win rate: %s' % (win_cnt / games_played))
        print('p1 draw rate: %s' % (draw_cnt / games_played))
        print('p1 lose rate: %s' % (lose_cnt / games_played))
        print()

    def save_stat_history(self, loss, win_cnt, draw_cnt, lose_cnt, games_played):
        self.win_rate_list.append((win_cnt / games_played))
        self.draw_rate_list.append(draw_cnt / games_played)
        self.lose_rate_list.append(lose_cnt / games_played)
        self.loss_list.append(loss)

    def train(self, episode, memory_size, episode_switch_q_target,
              is_special_sample, save_model_path):
        """ training mode of the Q players
        """
        # only player 1 could be the agent that accept training
        self.p1.build_memory(memory_size)
        self.p1.change_if_special_sample(is_special_sample)

        tune_view = Player.tune_observation_view

        # set up default value of variables that will be used
        active_id = None
        loss = 0
        win_cnt = 0
        draw_cnt = 0
        lose_cnt = 0
        who_first = 1
        initial_record = {'observation':None, 'action':None,
                          'reward':np.asarray([0,0]),
                          'next_observation':None, 'done':None}

        self.win_rate_list = []
        self.draw_rate_list = []
        self.lose_rate_list = []
        self.loss_list = []

        for epi in range(episode):
            if (epi+1) % 1000 == 0:
                self.report_stat(epi+1, loss, win_cnt, draw_cnt, lose_cnt, games_played=1000)
                self.save_stat_history(loss, win_cnt, draw_cnt, lose_cnt, games_played=1000)
                win_cnt = draw_cnt = lose_cnt = 0

            # reset the board, switch starting player
            self.reset()
            self.pick_start_player_id(who_first)
            who_first *= -1

            state_record = {1: deepcopy(initial_record),
                            -1: deepcopy(initial_record)}

            while not self.is_terminal_state:
                # get a np array to indicate whether each acion is available
                # ask the current active player to pick a action
                action = self.get_active_player().pick_action(
                    is_action_available=self.get_action_availability(),
                    observation=self.observation
                )

                # record the state change for putting into replay memory
                active_id = self.get_active_player_id()
                inactive_id = active_id * -1
                tunned_observation = tune_view(self.observation.copy(), active_id)
                cat_observation = Player.observation_to_catagorical(tunned_observation)
                state_record[active_id]['observation'] = cat_observation
                state_record[active_id]['action'] = action

                # given the action, update the board
                observation, reward, is_terminal_state = self.step(action)

                # update state record, note that the self.active_id is changed in self.step()
                # but the active_id and inactive_id here do not change
                state_record[active_id]['reward'] = tune_view(reward, active_id)
                tunned_observation = tune_view(observation.copy(), inactive_id)
                cat_observation = Player.observation_to_catagorical(tunned_observation)
                state_record[inactive_id]['next_observation'] = cat_observation
                state_record[inactive_id]['done'] = is_terminal_state
                state_record[inactive_id]['reward'] = state_record[inactive_id]['reward'] + tune_view(reward, inactive_id)
                # only record the player own reward, do not record the opponent reward
                state_record[inactive_id]['reward'] = state_record[inactive_id]['reward'][0]

                # add the state record to the memory
                if state_record[inactive_id]['observation'] is not None:
                    self.p1.memorize(deepcopy(state_record[inactive_id]))

            if self.winner == 1:
                win_cnt += 1
            elif self.winner == 0:
                draw_cnt += 1
            elif self.winner == -1:
                lose_cnt += 1

            # add last state record to the memory
            tunned_observation = tune_view(self.observation.copy(), inactive_id)
            cat_observation = Player.observation_to_catagorical(tunned_observation)
            state_record[active_id]['next_observation'] = cat_observation
            state_record[active_id]['reward'] = state_record[active_id]['reward'][0]
            state_record[active_id]['done'] = is_terminal_state

            self.p1.memorize(deepcopy(state_record[active_id]))

            # train the q_nn after every eposide
            loss = self.p1.update_qnn()

            # copy the q_nn to the q_target every episode_switch_q_target eposide
            if epi % episode_switch_q_target == 0:
                self.p1.update_qtarget()

        self.p1.save_model(save_model_path)
        return self.win_rate_list, self.draw_rate_list, self.lose_rate_list, self.loss_list

    def play(self, who_first='', episode=1, is_print_board=True):
        """ testing mode/ playing mode
        """
        win_cnt = 0
        draw_cnt = 0
        lose_cnt = 0
        self.p1.change_mode(is_train=False)
        self.p2.change_mode(is_train=False)

        # reset the board and randomly pick the starting player
        for i in range(episode):
            self.reset()
            self.pick_start_player_id(who_first)

            while not self.is_terminal_state:
                # get a np array to indicate whether each acion is available
                # ask the current active player to pick a action
                action = self.get_active_player().pick_action(
                    is_action_available=self.get_action_availability(),
                    observation=self.observation
                )

                # given the action, update the board
                self.step(action)

                if is_print_board:
                    self.print_board()

            if self.winner == 1:
                win_cnt += 1
            elif self.winner == 0:
                draw_cnt += 1
            elif self.winner == -1:
                lose_cnt += 1

        self.report_stat(None, None, win_cnt, draw_cnt, lose_cnt, episode)
