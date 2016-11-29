import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator



class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## DONE  ##
        ###########
        # Set any additional class parameters as needed

        self.trials = 0
    
    def logistic(self, x, x0=90, L=1, k=0.3):
        """ This is an inverted logistic function.
            x0, is the sigmoid's midpoint
            L, is the sigmoid's max value
            k is the steepness of the transition
        """

        return L / (1 + pow(math.e, -1 * k * (x-x0)))


    
    def logistic_inv(self, x, x0=90, L=1, k=0.3):
        """ This is an inverted logistic function.
            x0, is the sigmoid's midpoint
            L, is the sigmoid's max value
            k is the steepness of the transition
        """

        return 1 - self.logistic(x, x0, L, k)

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)

        self.trials += 1
        
        ########### 
        ## DONE  ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            self.alpha = 0.0
            self.epsilon = 0.0
        else:
            #self.epsilon = self.epsilon * math.cos(0.065*self.trials) # This was the better cosine decay that I found
            #self.epsilon = self.logistic_inv(self.trials, L=0.4999, k=0.5, x0=9) - 0.4999 # This was a sigmoid almost cosine-alike in which I iterated over
            self.epsilon = self.logistic_inv(self.trials, L=1.0, k=0.85, x0=35)
            self.alpha = self.logistic(self.trials, L=0.499, k=20, x0=35) + 0.001

        return None


    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## DONE  ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent
        # When learning, check if the state is in the Q-table
        #   If it is not, create a dictionary in the Q-table for the current 'state'
        #   For each action, set the Q-value for the state-action pair to 0
        
        state = (waypoint, inputs["light"], inputs["oncoming"], inputs['left'])

        
        self.createQ(state)

        return state


    def get_maxQ(self, state):
        """ The get_maxQ function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## DONE  ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state

        maxQ = None

        if state in self.Q:
            for itr_action in self.valid_actions:
                if self.Q[state][itr_action] > maxQ:
                    maxQ = self.Q[state][itr_action]
        else:
            maxQ = 0.0

        return maxQ


    def get_argmaxQ(self, state):
        """ The get_argmaxQ function is called when the agent is asked to find an
            action with the maximum Q-value of all actions based on the 'state' the smartcab is in. """

        argmaxQ = None
        maxQ = self.get_maxQ(state)

        if state in self.Q:
            for itr_action in self.valid_actions:
                if self.Q[state][itr_action] == maxQ:
                    argmaxQ = itr_action
                    break
        
        return argmaxQ


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## DONE  ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        if self.learning:
            if not(state in self.Q):
                # If it is not, create a new dictionary for that state
                #   Then, for each action available, set the initial Q-value to 0.0
                self.Q[state] = dict()
                for itr_action in self.valid_actions:
                    self.Q[state][itr_action] = 0.0

        return 


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None
        _a = None

        ########### 
        ## DONE  ##
        ###########

        if not(self.learning):
            # When not learning, choose a random action
            _a = int(random.uniform(0, len(self.valid_actions)))
            action = self.valid_actions[_a]
        else:
            # When learning, choose a random action with 'epsilon' probability
            #   Otherwise, choose an action with the highest Q-value for the current state
            if random.random() <= self.epsilon:
                _a = int(random.uniform(0, len(self.valid_actions)))
                action = self.valid_actions[_a]
            else:
                action = self.get_argmaxQ(self.state)

        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        state_prime = None

        ########### 
        ## DONE  ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if self.learning:
            state_prime = self.build_state()
            self.Q[state][action] = ((1 - self.alpha) * self.Q[state][action]) + (self.alpha * ( reward +  self.get_maxQ(state_prime)))

        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return


def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose=False)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True, alpha=0.5, epsilon=0.5)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0, log_metrics=True, display=False, optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=50, tolerance=0.00001)


if __name__ == '__main__':
    run()

