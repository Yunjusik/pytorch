# this class is coded for Prioritized_Experience_Replay (PER)
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        # additive code for PER
        self.prob_alpha = 0.6 ## initial value
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
        

    def push(self, *args):
        """Saves a transition."""
        
        max_prio = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        
             
        
        
        
        self.priorities[self.position] = max_prio
        
        
        self.position = (self.position + 1) % self.capacity
        
        


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
