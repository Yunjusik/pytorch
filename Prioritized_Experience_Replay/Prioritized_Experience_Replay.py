# this class is coded for Prioritized_Experience_Replay (PER)
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        # additive code for PER
        self.prob_alpha = 0.6 ## initial value
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        #make priority value array with [capacity] size
        

    def push(self, *args):
        """Saves a transition."""
        ##
        max_prio = self.priorities.max() if self.memory else 1.0
        #load max priority rank of full memory, initialize first index as 1.0
        ##
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args) ## transition insert
        
        ##        
        self.priorities[self.position] = max_prio ## update priorities array of given transition index to max_prio
        ##
               
        self.position = (self.position + 1) % self.capacity ##count value update
        
        

    def sample(self, batch_size, beta = 0.4):

        if len(self.memory) == self.capacity:  ## if memory is full, load full priorities
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
        probs = prios ** self.prob_alpha  ## recall that prios is priorities array. each square proccess is done for all memory component
        probs /= probs.sum()  ## probability of sampling of transition based on PER paper.
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)  ## [batch] 1d array

        return samples, indices, weights
   


####  priority update. required data shape
#### batch_indices.shape -> 1D(batch size array)
#### batch_priorities -> 2D (batch size x 1) array.

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio  
            
####   update_priorites function interact with optimize_model
   

    def __len__(self):
        return len(self.memory)
