import numpy as np

class Hopfield:
    def __init__(self, patterns:list[np.ndarray]):
        self.patterns = patterns # [[1 -1  1  1 ... 1 (25)], ... 4 patterns]
        self.dimension = len(patterns[0]) # 25
        self.weights = np.dot(np.array(patterns).T, np.array(patterns)) / self.dimension  # 25x25
        np.fill_diagonal(self.weights, 0) # set the diagonal to 0
        
    
    def train(self, untrained_pattern, epochs=50):
        states = []
        state_dim = int(np.sqrt(self.dimension))
        untrained_pattern = np.array(untrained_pattern)
        state = untrained_pattern
        previous_state = np.zeros(self.dimension)
        states.append(state.reshape((state_dim, state_dim)))
        energies = []
        # energies.append(self.energy(state))

        i = 0
        # stop when two consecutive states are equals or when the number of epochs is reached
        while i < epochs and not np.array_equal(state, previous_state):
            energies.append(self.energy(state))
            previous_state = state
            state = self.update(state)
            states.append(state.reshape((state_dim, state_dim)))
            i += 1
        
        # if the state is a pattern, return True
        for pattern in self.patterns:
            if np.array_equal(pattern, state):
                return True, state.reshape((state_dim, state_dim)), states, energies, i
        
        return False, state.reshape((state_dim, state_dim)), states, energies, i # spurious state (not a pattern)


    # calculate the energy of the current state
    def energy(self, state:np.ndarray):
        h = 0
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                h += self.weights[i][j] * state[i] * state[j]
        return -h
    

    # update the states of the network
    def update(self, state:np.ndarray):
        return np.sign(np.dot(self.weights, state.T))
    
        
    def check_stability(self, state):
        for pattern in self.patterns:
            if np.isclose(state, pattern).all():
                return True
        return False
