class Dueling_DQN(nn.Module):
    ### Dueling Network enabled DQN
    def __init__(self, h, w,  num_actions):
        super(DQN, self).__init__()
        self.num_actions=num_actions
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64
        self.fc = nn.Linear(linear_input_size, 512) ## ADV
        self.fc2 = nn.Linear(512, self.num_actions)

        self.fc3 = nn.Linear(linear_input_size, 512) ## val
        self.fc4 = nn.Linear(512,1)


        self.relu = nn.ReLU()
    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)


        adv = self.relu(self.fc(x))    
        val = self.relu(self.fc3(x))
        adv = self.fc2(adv)
        val = self.fc4(val).expand(x.size(0), self.num_actions)
        # adv denotes advantage function of dueling networks
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0),self.num_actions)

        return x
