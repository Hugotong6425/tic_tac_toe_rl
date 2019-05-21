import numpy as np

from py_files.board import Board
from py_files.player import Q_player, Random_player, Human

#---------------------- config for set up QPlayer-----------------------
# feel free to make a cool name :)
player_name = 'q player'

# training config
batch_size = 64
learning_rate = 0.001
ini_epsilon = 0.9
epsilon_decay = 0.9995
epsilon_min = 0.01
gamma = 0.99

# only relevant if load_trained_model_path is None
hidden_layers_size = [20,10]

# whether to use double dqn
is_double_dqns = True

# optimizer and loss can be anything that is a Keras optimizer/loss
# e.g. keras.optimizers.Adam(lr=0.0001)
optimizer = 'adam'
loss = 'mse'

# if it is a model path (str), load the model according to the path
# if None, initialize new model weights
load_trained_model_path = None

# set it to be true when training
is_train = True

# player 2 setting
# p2 will not learn, it can be either 'random'(random player) or 'q_player'
# if it is 'q_player', need to provide the model path to load
p2_player_type = 'random'
p2_load_trained_model_path = None
#---------------------- config for set up QPlayer-----------------------

#---------------------- config for board.train()-----------------------
# training config
episode = 3000
memory_size = 1000
episode_switch_q_target = 500
is_special_sample = True
save_model_path = 'model.h5'

win_reward = 1
lose_reward = -1
draw_reward = 0

#---------------------- config for board.train()------------------------

p1_config = {'player_type':'q_player', 'hidden_layers_size':hidden_layers_size,
             'player_name':player_name, 'batch_size':batch_size,
             'learning_rate':learning_rate, 'ini_epsilon':ini_epsilon,
             'epsilon_decay':epsilon_decay, 'epsilon_min':epsilon_min,
             'gamma':gamma, 'is_double_dqns':is_double_dqns, 'optimizer':optimizer,
             'loss':loss, 'load_trained_model_path':load_trained_model_path,
             'is_train':is_train}

p2_config = {'player_type':'random', 'load_trained_model_path':p2_load_trained_model_path}

test_board = Board(p1_config=p1_config, p2_config=p2_config,
                   win_reward=win_reward, lose_reward=lose_reward,
                   draw_reward=draw_reward)

win_rate_list, draw_rate_list, lose_rate_list, loss_list = \
    test_board.train(episode, memory_size=memory_size,
                     episode_switch_q_target=episode_switch_q_target,
                     is_special_sample=is_special_sample,
                     save_model_path=save_model_path)
