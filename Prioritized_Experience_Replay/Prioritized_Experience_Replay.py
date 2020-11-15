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
        
        

## main contribution of PER is based on important sampling weight method
    def sample(self, batch_size):
        
        if len(self.memory) == self.capacity: ## if memory is full, load full priorities
            prios = self.priorities 
        else:
            prios = self.priorities[:self.position] 
            
        probs = prios ** self.prob_alpha ## recall that prios is priorities array. each square proccess is done for all memory component
        probs /= probs.sum() ## probability of sampling of transition based on PER paper.
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs) # indice size (batch x 1)
        samples = [self.memory[idx] for idx in indices] # importance samples memory according to indices index, which represents prioritized transition.
        #Then, samples consists of consecutive namedtuple list
        #for example, [transition x(state,action,reward,sate'], transition y(state,action,reward,sate'], transition z(state,action,reward,sate'],...]
        #the number of components in samples is equivalent to indices number
        return random.sample(self.memory, batch_size)
   


####  priority update. required data shape
#### batch_indices.shape -> 1D(batch size array)
#### batch_priorities -> 2D (batch size x 1) array.

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio  
            
####   update_priorites function interact with optimize_model
   

    def __len__(self):
        return len(self.memory)
