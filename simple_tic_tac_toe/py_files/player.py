# work for random player and human first

class Player():
    """
    Abstract player class
    """
    def __init__(self):
        self.player_id = None
        self.observation = None

    def reset(self, player_id):
        '''reset to the initial state
        '''
        self.player_id = player_id
        self.observation = None

    def observe(self, observation):
        '''
        receive raw observation from env and tune it if needed
        '''
        self.observation = tune_observation_view(observation, self.player_id)

    @staticmethod
    def tune_observation_view(observation, player_id):
        '''
        player_id either 1 or -1. Swap the observation such that 1 means self and -1 means the opponent
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

    def pick_action(self, **kwargs):
        '''different players have different way to pick an action
        '''
        pass

    def memorize(self, add_this):
        '''some players will jot notes, some will not
        '''
        pass

    def learn(self, board, **kwargs):
        '''some players will study, some will not
        '''
        pass


class Human(Player):
    '''
    choose this player if you want to play the game
    '''
    def __init__(self):
        super(Human, self).__init__()

    def pick_action(self, **kwargs):
        cell = input('Pick a cell (top left is 0 and bottom right is 8): \n')
        return int(cell)


class Random_player(Player):
    """
    this player will pick random acion for all situation
    """
    def __init__(self):
        super(Random_player, self).__init__()

    def pick_action(self, **kwargs):
        possible_action_list = np.argwhere(kwargs['is_action_available'] == 1).reshape([-1])
        return np.random.choice(possible_action_list, 1)[0]
