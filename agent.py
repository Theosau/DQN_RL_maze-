############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import collections

class Agent:

    # Function to initialise the agent and the implementation parameters
    def __init__(self):
        '''This function serves to initialise the class attributes
        necessary for the DQN learning algortihm. It creates an instance
        of the DQN and Experience Replay Buffer to be used throughout the
        learning process.'''
        ## Parameters configuration
        #Steps
        self.episode_length = 5000 #initial episode length
        self.old_steps = 0 #number of steps after Nobservations
        self.counter_episode = 0 #number of episodes
        self.num_steps_taken = 0 #total steps taken

        #Neural Network
        self.learning_rate = 0.001 #for the neural network
        self.mini_batch = 128 #for training step of the network
        self.dqn = DQN(self.learning_rate)  #deep Q network instance
        self.traget_frequency = 50 #target network frequency update
        self.discount_factor = 0.9 #discount factor

        #Epsilon greedy policy
        self.delta = 2.5e-5 #decay rate of the exploration
        self.delta_2 = 1e-4
        self.epsilon = 1 #epsilon greedy initial value. Initially fully random

        #Experience Replay Buffer
        self.epsilon_probabilities = 1*1e-6 #minimum pb to take each sample in the batch
        self.buffer_size = int(1e5) #replay buffer maximum size
        self.alpha_weights = 0.7 #piroritising constant
        self.replaybuffer = ReplayBuffer(self.dqn, self.epsilon_probabilities,
                                         self.mini_batch, self.buffer_size,
                                         self.alpha_weights)

        #Transitions
        self.distance_to_goal = 10 #distance to goal, initialised very high
        self.state = None #stores the latest state of the agent in the environment
        self.next_state = None #stores the next state of the agent in the environment
        self.action = None # stores the latest action which the agent has applied to the environment


    # Function to check whether the agent has reached the end of an episode
    # Also serves to updates the next episode length
    def has_finished_episode(self):
        episode_steps = self.num_steps_taken - self.old_steps
        #Check if the episode length has been reached or if the agent reached th goal
        if (episode_steps % self.episode_length==0) or (self.distance_to_goal < 0.03):
            self.counter_episode += 1
            self.old_steps = self.num_steps_taken
            #Linearly decreasing the episode length
            if self.num_steps_taken >= 25000:
                if self.episode_length > 500:
                    self.episode_length -= 500
                #Limiting the smallest episode to 500 steps
                else:
                    self.episode_length = 500
            return True
        else:
            return False

    # Function to get the next action
    def get_next_action(self, state):
        # Store the state; this will be used later, when storing the transition
        self.state = state

        #Update the epsilon value for epsilon greedy policy
        self.update_epsilon()

        #Getting the best future action from the Q network
        network_prediction_action = self.dqn.q_network.forward(torch.tensor(self.state))
        index_max_network_prediction_action = network_prediction_action.max(0)[1].item()

        #Implementing the epsilon greedy policy
        actions_index = np.arange(3)
        probabilities = (self.epsilon/3)*np.ones((3))
        probabilities[index_max_network_prediction_action] += 1 - self.epsilon
        self.discrete_action = np.random.choice(actions_index, p = probabilities)

        # Convert the discrete action into a continuous action.
        action = self._discrete_action_to_continuous()
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return action

    #This function update the epsilon value based on the overall number of steps
    #taken.
    def update_epsilon(self):
        #Start the greedy policy after 5000 steps, and ensure that epsilon
        #remains higher or equal to 0.005 over all steps
        if 5000 < self.num_steps_taken < 30000:
            self.epsilon = 1 - self.delta*(self.num_steps_taken-5000)
        elif self.num_steps_taken >= 30000:
            self.epsilon = max(0.005, 1 - self.delta_2*(self.num_steps_taken-5000))

    #This function serves to convert the discrete action to a continuous, and
    #returns the latter
    def _discrete_action_to_continuous(self):
        step_size = 0.02

        if self.discrete_action == 0:  # Move North/Top
            continuous_action = np.array([0, step_size], dtype=np.float32)
        if self.discrete_action == 1:  # Move East/Right
            continuous_action = np.array([step_size, 0], dtype=np.float32)
        if self.discrete_action == 2:  # Move South/Bottom
            continuous_action = np.array([0, -step_size], dtype=np.float32)
        # if self.discrete_action == 3:  # Move West/Left
        #     continuous_action = np.array([-step_size, 0], dtype=np.float32)

        return continuous_action


    # Function to set the next state and distance, which resulted from applying
    #action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):

        self.distance_to_goal = distance_to_goal

        ## Convert the distance to a reward
        #Reward finding the goal
        if self.distance_to_goal < 0.05:
            reward = 1 - self.distance_to_goal
        #Punish hitting a wall/staying at the same state
        elif self.state is next_state:
            reward = -0.1*self.distance_to_goal
        #Reward going right
        elif self.state[0] < next_state[0]:
            reward = (1 - self.distance_to_goal)
        #Punish going left
        elif self.state[0] > next_state[0]:
            reward = - self.distance_to_goal
        else:
            reward = 0

        # Create a transition
        transition = (self.state, self.discrete_action, reward, next_state)
        # Update the state
        self.state = next_state
        #Store the tranisition in the Buffer
        self.replaybuffer.append_transition(transition)
        #Train the network and get the over and sample specific losses!
        loss, losses = self.update_network()

        #Update the network with the chosen frequency
        if ((self.num_steps_taken - self.mini_batch) % self.traget_frequency) == 0 and (self.num_steps_taken - self.mini_batch > 0):
            self.dqn.update_target()

        #Update the losses every iteration after we have at least one minibatch
        if self.num_steps_taken >= self.mini_batch:
            self.replaybuffer.update_weights(losses)

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # returns the action with the highest Q-value
        network_prediction_action = self.dqn.q_network.forward(torch.tensor(state))
        self.discrete_action = network_prediction_action.max(0)[1].item()
        action = self._discrete_action_to_continuous()
        return action

    #Update the Q network once it went through a number of steps greater than
    #the mini-batch
    def update_network(self):
        if self.num_steps_taken >= self.mini_batch:
            #get all the transitions in the batch under the form of an array
            batch_transitions = self.replaybuffer.sample_batch()#.astype(np.float32)
            #Trains the Q network using the batch inputs
            #Compute the value of the losses for each batch input, after it has been trained
            loss, losses = self.dqn.train_q_network(batch_transitions, self.discount_factor)
            return loss, losses
        else:
            return [], []


# The Network class inherits the torch.nn.Module class, which represents a neural network.
##Note: this has been taken from the Coursework provided code ref:
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

#This class serves to build the DQN
class DQN:

    # The class initialisation function.
    def __init__(self, learning_rate):
        #Create a target network to stabilise the learning
        self.q_target_network = Network(input_dimension=2, output_dimension=3)

        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=3)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        lr = learning_rate
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr) #

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition, discount_factor):
        #Get the disocunt factor
        self.discount_factor = discount_factor
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss, losses = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar and a vector
        return loss.item(), losses

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):
        #Split the transition array (became an array through the sampling in replaybuffer)
        state = transition[:,0:2].astype(np.float32)
        action = transition[:,2].astype(int)
        reward = transition[:,3]
        next_state = transition[:,4:6].astype(np.float32)
        index = 1

        #Predicted values by the Q network now
        network_predictions= self.q_network.forward(torch.tensor(state))
        network_prediction = network_predictions.gather(index, torch.tensor(action).unsqueeze(index))

        #Getting the predictions from the target network
        network_predictions_next_state = self.q_target_network.forward(torch.tensor(next_state))
        network_prediction_next_state = network_predictions_next_state.max(index)[0].unsqueeze(index)

        #Bellman Estimation
        batch_labels_tensor = (torch.tensor(reward).unsqueeze(index) + self.discount_factor * network_prediction_next_state)

        #Compute the losses for each of the mini-batch sample
        losses = abs(network_prediction - batch_labels_tensor)
        #Compute the loss of the DQN for parameters update
        loss = torch.nn.MSELoss()(network_prediction, batch_labels_tensor)
        return loss, losses

    #Update the target network
    def update_target(self):
        parameters_q = torch.nn.Module.state_dict(self.q_network)
        torch.nn.Module.load_state_dict(self.q_target_network, parameters_q)

#This class consturcts the replaybuffer, updates the weights of each sample
#and samples from buffer to create the batch
class ReplayBuffer():

    def __init__(self, dqn, epsilon_probabilities, batch_size_input, buffer_size, alpha):
        self.max_len_buffer = buffer_size
        self.buffer = collections.deque([], self.max_len_buffer) #max size
        self.counter = 0 #to do buffer updates when there is overflow
        self.weights = np.zeros(self.max_len_buffer) #weights for each transitino to update the probabilities
        self.epsilon_probabilities = epsilon_probabilities #min probability for each sample to be taken for the batch
        self.batch_size = batch_size_input #batch size
        self.counter = 0
        self.alpha = alpha

    #Adds each transition to the buffer
    def append_transition(self, transition):
        if len(self.buffer) < self.max_len_buffer:
            self.buffer.append(transition)
            #Initialiase all the weights to be equal for the first batch
            if len(self.buffer) <= self.batch_size:
                self.weights[self.counter] = 1/self.batch_size
            #Initialise the new weights to the max of exisiting buffer
            else:
                self.weights[self.counter] = np.max(self.weights)
        # Overwriting the oldest transition stored
        else:
            if self.counter >= self.max_len_buffer:
                self.counter = 0
            self.buffer[self.counter] = transition
            self.weights[self.counter] = np.max(self.weights)

        self.counter += 1

    #Sample the batch according to probabilities based on the weigths
    #which are based on the losses
    def sample_batch(self):
        self.weights[:len(self.buffer)] = self.weights[:len(self.buffer)] + self.epsilon_probabilities
        #getting probabilities based on weights
        probabilities = self.weights**self.alpha/np.sum(self.weights**self.alpha)
        #sampling according to the probabilities
        self.batch_indices = np.random.choice(range(len(self.buffer)), self.batch_size, True, probabilities[:len(self.buffer)])
        batch_arr = np.zeros((self.batch_size, 6))

        #Creat an array of allt he transitions
        for i in range(self.batch_size):
            batch_arr[i,0] =  self.buffer[self.batch_indices[i]][0][0]
            batch_arr[i,1] =  self.buffer[self.batch_indices[i]][0][1]
            batch_arr[i,2] =  self.buffer[self.batch_indices[i]][1]
            batch_arr[i,3] =  self.buffer[self.batch_indices[i]][2]
            batch_arr[i,4] =  self.buffer[self.batch_indices[i]][3][0]
            batch_arr[i,5] =  self.buffer[self.batch_indices[i]][3][1]

        return batch_arr

    #Update the weights based on the losses which have just been updated in the
    #network optimisation
    def update_weights(self, losses):
        self.weights[self.batch_indices] = losses.detach().numpy().reshape(self.batch_indices.size,)
