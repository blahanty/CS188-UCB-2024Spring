import nn


class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """

    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        self.learning_rate = 0.5
        self.numTrainingGames = 3000

        self.hidden_layer_size = 128
        self.w1 = nn.Parameter(state_dim, self.hidden_layer_size)
        self.b1 = nn.Parameter(1, self.hidden_layer_size)
        self.w2 = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)
        self.b2 = nn.Parameter(1, self.hidden_layer_size)
        self.w3 = nn.Parameter(self.hidden_layer_size, action_dim)
        self.b3 = nn.Parameter(1, action_dim)
        self.parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

        self.batch_size = 128

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        return nn.SquareLoss(self.run(states), Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        return nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(nn.ReLU(
            nn.AddBias(nn.Linear(states, self.w1), self.b1)), self.w2), self.b2)), self.w3), self.b3)

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        loss = self.get_loss(states, Q_target)
        grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2, grad_wrt_w3, grad_wrt_b3 = nn.gradients(
            loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
        self.w1.update(grad_wrt_w1, -self.learning_rate)
        self.b1.update(grad_wrt_b1, -self.learning_rate)
        self.w2.update(grad_wrt_w2, -self.learning_rate)
        self.b2.update(grad_wrt_b2, -self.learning_rate)
        self.w3.update(grad_wrt_w3, -self.learning_rate)
        self.b3.update(grad_wrt_b3, -self.learning_rate)
