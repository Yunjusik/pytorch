class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.nstep_buffer = []
        self.nsteps = 3
    def push(self, *args):
        """Saves a transition."""
        self.nstep_buffer.append(Transition(*args))  ### buffer for n-step return
        if(len(self.nstep_buffer)<self.nsteps): 
            return
        if len(self.memory) < self.capacity:
            self.memory.append(None)  ## memory expansion
        R = sum([self.nstep_buffer[i][3]*(0.99**i) for i in range(self.nsteps)]) #multi-step bootstrap return
        S, A, N_S, _ = self.nstep_buffer.pop(0) # get n-step transition from n-step buffer
        self.memory[self.position] = Transition(S, A, N_S, R)
        self.position = (self.position + 1) % self.capacity
    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            ## memory expansion
            if len(self.memory) < self.capacity:
                self.memory.append(None)
                ##
            R = sum([self.nstep_buffer[i][3] * (0.99 ** i) for i in range(len(self.nstep_buffer))])
            S, A, N_S, _ = self.nstep_buffer.pop(0)
            self.memory[self.position] = Transition(S, A, N_S, R)
            self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
