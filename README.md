# Tic Tac Toe with Deep Q Learning

Tic-tac-toe (American English), noughts and crosses (British English), or Xs and Os is a paper-and-pencil game for two players, X and O, who take turns marking the spaces in a 3×3 grid. The player who succeeds in placing three of their marks in a horizontal, vertical, or diagonal row wins the game. (Wiki)

In practice, tabular Q learning is a better method to train a tic-tac-toe agent due to the limited number of states and the stablity of tabular Q learning. However, this repo aims to build a deep Q learning agent for this game.

![image](https://upload.wikimedia.org/wikipedia/commons/3/32/Tic_tac_toe.svg)

## Train model

### Requirement

Required package:
- numpy
- keras

You may also install them by `python install -r requirements.txt`

Before training the model, you may open ```train_rl.py``` to change model config. Then simply ```python train_rl.py``` to start the training.

### Config
- `player_name`: str, feel free to make a cool name :)
- `batch_size`: int, batch size
- `learning_rate`: float, learning rate
- `ini_epsilon`: float, initial epsilon
- `epsilon_decay`: float, every episode the current epsilon will time this factor
- `epsilon_min`: float, minimum epsilon
- `gamma`: float, reward discount
- `hidden_layers_size`: list of int, only relevent if `load_trained_model_path` is None
- `is_double_dqns`: bool, whether to use double dqn
- `optimizer`: anything that is a Keras optimizer e.g. keras.optimizers.Adam(lr=0.0001)
- `loss`: anything that is a Keras loss
- `load_trained_model_path`: str or None
- `is_train`: set it to be true when training
- `p2_player_type`: str, 'random' or 'q_player'
- `p2_load_trained_model_path`: If `p2_player_type = 'random`, set it to None, else set it a str (saved model path)
- `episode`: int, episode
- `memory_size`: int, replay memory size
- `episode_switch_q_target`: int, every a number of episode, copy q_value model parameters to q_target
- `is_special_sample`: bool, if True, focus on sampling terminal state when deciding training batch
- `save_model_path`: str, path to save the final trained model
- `win_reward`: float
- `lose_reward`: float
- `draw_reward`: float
