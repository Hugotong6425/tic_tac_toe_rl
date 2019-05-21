import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam

from .memory import Memory


class Player():
    """
    Abstract player class
    """
    def __init__(self, player_name=None, player_id=None):
        self.memory = None
        self.player_name = player_name
        self.player_id = player_id

    def build_memory(self, memory_size):
        self.memory = Memory(memory_size)

    def set_player_id(self, player_id):
        '''set player id
        '''
        self.player_id = player_id

    def change_mode(self, is_train):
        self.is_train = is_train

    @staticmethod
    def tune_observation_view(observation, player_id):
        ''' if player_id is -1, swap the 1 and -1 of the observation
        player_id either 1 or -1. Swap the observation such that 1 means self
        and -1 means the opponent

        Args:
            - observation: np array
            - player_id: int, 1 or -1

        e.g.
        if player_id = 1, no need to swap the view,
        input = array([ 1, -1, 0,
                        0,  0, 0,
                       -1, -1, 1 ])

        output = array([ 1, -1, 0,
                         0,  0, 0,
                        -1, -1, 1 ])

        if player_id = -1, need to swap the view,
        input = array([ 1, -1, 0,
                        0,  0, 0,
                       -1, -1, 1 ])

        output = array([-1, 1, 0,
                         0, 0, 0,
                         1, 1, -1 ])

        '''
        return observation * player_id

    @staticmethod
    def observation_to_catagorical(observation):
        '''given a observation of np array [9], convert it to catagorical form
        i.e. np array of size [27]

        input = array([ 1, -1, 0,
                        0,  0, 0,
                       -1, -1, 1 ])

        output = array([ 0, 1, 0,
                         0, 0, 1,
                         1, 0, 0,
                         1, 0, 0,
                         1, 0, 0,
                         1, 0, 0,
                         0, 0, 1,
                         0, 0, 1,
                         0, 1, 1])
        '''
        cat_observation = []
        for cell in range(9):
            if observation[cell] == 0:
                cat_observation.extend([1,0,0])
            elif observation[cell] == 1:
                cat_observation.extend([0,1,0])
            elif observation[cell] == -1:
                cat_observation.extend([0,0,1])
            else:
                print('ERROR in observation to catagorical')
        return np.array(cat_observation)

    def pick_action(self, **kwargs):
        '''different players have different way to pick an action
        '''
        pass

    def memorize(self, record):
        '''some players will jot notes, some will not
        '''
        self.memory.add(record)

    def learn(self, board, **kwargs):
        '''some players will study, some will not
        '''
        pass

    def update_qnn(self):
        '''given a batch of data (all observations inside should be tuned)
        '''
        pass

    def update_qtarget(self):
        pass

    def get_player_name(self):
        return self.player_name

    def get_player_id(self):
        return self.player_id


class Human(Player):
    '''
    choose this player if you want to play the game
    '''
    def __init__(self, player_name='human player'):
        super(Human, self).__init__()
        self.player_name = player_name

    def pick_action(self, **kwargs):
        cell = input('Pick a cell (top left is 0 and bottom right is 8): \n')
        return int(cell)


class Random_player(Player):
    """
    this player will pick random acion for all situation
    """
    def __init__(self, player_name='random player', load_trained_model_path=None):
        super(Random_player, self).__init__()
        self.player_name = player_name

    def pick_action(self, **kwargs):
        possible_action_list = np.argwhere(kwargs['is_action_available'] == 1).reshape([-1])
        return np.random.choice(possible_action_list, 1)[0]


class Q_player(Player):
    def __init__(self, hidden_layers_size, batch_size, learning_rate,
                 ini_epsilon, epsilon_min, epsilon_decay, is_double_dqns, gamma,
                 loss, load_trained_model_path, player_name, is_train):
        super(Q_player, self).__init__()
        self.hidden_layers_size = hidden_layers_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = ini_epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.is_double_dqns = is_double_dqns
        self.gamma = gamma
        self.loss = loss
        self.player_name = player_name
        self.qnn = self.create_neural_network(load_trained_model_path)
        self.is_train = is_train

        self.q_target_network = self.create_neural_network(load_trained_model_path)

    def create_neural_network(self, load_trained_model_path=None):
        '''
        if load_trained_model_path is not None, load the model.
        if load_trained_model_path is None, initialize the model
        '''
        if load_trained_model_path is None:
            x = Input(shape=(27,))

            hidden_result = Dense(self.hidden_layers_size[0], activation='relu',
                                  kernel_initializer='he_normal')(x)
            hidden_result = Dropout(0.5)(hidden_result)

            if len(self.hidden_layers_size) > 1:
                for num_hidden_neurons in self.hidden_layers_size[1:]:
                    hidden_result = Dense(num_hidden_neurons, activation='relu',
                                          kernel_initializer='he_normal')(hidden_result)
                    hidden_result = Dropout(0.5)(hidden_result)

            y = Dense(9, activation=None)(hidden_result)

            model = Model(inputs=x, outputs=y)

            model.compile(optimizer=Adam(lr=self.learning_rate), loss=self.loss)
            return model
        else:
            print('Load in model: %s' % load_trained_model_path)
            return load_model(load_trained_model_path)

    def save_model(self, save_model_path):
        self.qnn.save(save_model_path)

    def get_current_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            return max(self.epsilon_min, self.epsilon)
        else:
            self.epsilon = self.epsilon_min
            return self.epsilon_min

    def change_if_special_sample(self, is_special_sample):
        self.is_special_sample = is_special_sample

    def convert_memory_to_train_data(self, batch_memory_data):
        '''
        Args:
            - batch_memory_data: list of memory data points

        observation:
            - changed to catagorical
            - tunned view

        memory data points:
            [observation: np.array(27), action: int, reward: int,
             next_observation: np.array(27), done: bool]

        Returns:
            - batch_observation: np array of int [batch_size, 27]
            - batch_action: np array of int [batch_size]
            - batch_reward: np array of int [batch_size]
            - batch_next_observation: np array of int [batch_size, 27]
            - batch_done: np array of bool [batch_size]
        '''
        stack_memory = np.stack(batch_memory_data, axis=1)

        return np.stack(stack_memory[0]), stack_memory[1], stack_memory[2], \
            np.stack(stack_memory[3]), stack_memory[4]

    def update_qnn(self):
        '''given a batch of data (all observations inside should be tuned)
        '''
        if self.memory.get_memory_size() < self.batch_size:
            return

        #print('self.memory.get_memory_size(): ', self.memory.get_memory_size())

        # get memory data and convert the data format to feed the network
        batch_memory_data = self.memory.sample(self.batch_size, self.is_special_sample)

        #print('batch_memory_data: ', batch_memory_data)

        batch_observation, batch_action, batch_reward, batch_next_observation, \
            batch_done = self.convert_memory_to_train_data(batch_memory_data)

        if self.is_double_dqns:
            qnn_next_obser_q = self.qnn.predict(batch_next_observation) # [batch, 9]
            next_obser_action = np.argmax(qnn_next_obser_q, axis=1) # [batch]

            qtarget_next_obser_q = self.q_target_network.predict(batch_next_observation) # [batch, 9]
            best_q_next_obser = np.array([qtarget_next_obser_q[i, next_obser_action[i]]
                                          for i in range(self.batch_size)])
        else:
            # create q target for the network to learn
            q_next_state = self.q_target_network.predict(batch_next_observation)
            best_q_next_obser = np.max(q_next_state, axis=1)

        # [batch_size], target q value of the selected action
        # if done, then q_target = batch_reward, else need to add up the term behind
        q_target = batch_reward + self.gamma * best_q_next_obser * (batch_done * (-1) + 1)

        # [batch_size, 9], target value for the network to train
        y_target = self.qnn.predict(batch_observation)

        for i in range(self.batch_size):
            y_target[i, batch_action[i]] = q_target[i]

        history = self.qnn.fit(batch_observation, y_target, epochs=1, verbose=0)
        return history.history['loss'][0]

    def update_qtarget(self):
        print('current epsilon: ', self.epsilon)
        self.q_target_network.set_weights(self.qnn.get_weights())

    def pick_action(self, **kwargs):
        '''given untuned observation and is_action_available, predict the best action

        Args:
            - kwargs['observation']: np array with size [9], untunned, not catagorical
            - kwargs['is_action_available']: np array with size [9]

        Returns:
            - picked_cell: int, 0-8

        Note: can only predict 1 observation at a time now
        '''
        observation = kwargs['observation']
        is_action_available = kwargs['is_action_available']

        if (self.is_train) and (np.random.rand() < self.get_current_epsilon()):
            action_space = [i for i in range(9) if is_action_available[i]]
            return np.random.choice(action_space)

        tunned_observation = self.tune_observation_view(observation, self.player_id)
        cat_observation = self.observation_to_catagorical(tunned_observation)

        # pick the best action within all possible action given by the board
        q_pred = self.qnn.predict(x=cat_observation.reshape([1, 27])).reshape([-1])
        mask = (is_action_available == 1)
        subset_idx = np.argmax(q_pred[mask])
        picked_cell = np.arange(q_pred.shape[0])[mask][subset_idx]

        return picked_cell
