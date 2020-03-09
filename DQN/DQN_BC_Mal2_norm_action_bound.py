# -*- coding: utf-8 -*-
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import openpyxl   ##
import pandas as pd

from BC_env_Mal_new_norm import BCnetenv

############ 합의성공률을 플랏하기위한 버전
###############  ++ action bound를 위해 select action 함수 변경


ENV_NAME = 'Blockchain_Network'
env = BCnetenv()




# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
'''

'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
######################################################################
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
######################################################################
def get_screen(pre_state):
    if pre_state == None:
        return None
    ## pre_state를 받으면 DQN에 인풋으로 들어갈 space는 None
    else: ## R,C, H, e_p는 각 nxn, nxn, nxn, nxn 로 존재해야함
        rr = torch.from_numpy(np.ascontiguousarray(pre_state[0], dtype=np.float32) / np.ndarray.max(pre_state[0])).unsqueeze(0)    # 1xnxn tensor
        cc = torch.from_numpy(np.ascontiguousarray(pre_state[1], dtype=np.float32) / np.ndarray.max(pre_state[1])).unsqueeze(0)     # 1xnxn
        hh = torch.from_numpy(np.ascontiguousarray(pre_state[2], dtype=np.float32) / np.ndarray.max(pre_state[2])).unsqueeze(0)   # 1xnxn
        e_p =torch.from_numpy(np.ascontiguousarray(pre_state[3], dtype=np.float32)).unsqueeze(0)   # 1xnxn 확률은 0~1사이므로, 정규화필요 X
        e_pp = pre_state[3]
        e_pp = e_pp[0,0]

        F_state_mat = torch.cat((rr, cc, hh, e_p), dim=1) # 배열 합
        F_state_mat = F_state_mat.reshape(4,200,200) # re-shape을 통한 차원 분배
        return F_state_mat.unsqueeze(0).to(device), e_pp  ## 2차원 스크린배열에 batch dimension 추가해서 리턴  1x4x200x200
######################################################################
# Training
#

BATCH_SIZE = 256
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 200        ## 정하는 step에 0.363배가 됨, 클수록 늦게 감쇠
TARGET_UPDATE = 10

# Get number of actions from gym action space
n_actions = env.action_space.n

#DQN input. -> data transmission rate of link (R) -> nxn
#              computing capability of node (c)   -> nxn
#              consensus history            (H)   -> nxn
policy_net = DQN(200, 200, n_actions).to(device)
target_net = DQN(200, 200, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(200000)
steps_done = 0

def select_action(state, e_p):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            constraint = (200 * (1 - (3 * e_p)) - 1) / (3 * 200 * e_p + 1) ## const1 기준
            K_prime = math.floor(constraint)
            if K_prime == 0 :
                K_prime = 1

            if K_prime >= 8 :
                K_prime = 8

            duplicate_net = policy_net(state).detach().cpu().numpy()  ## state 입력시 policy_net의 output단을 복제함.
            ### CUDA에들어간 tensor를 조작하려면, 뉴럴넷에서 detach를 하여 추적을  중지시키고
            ###cpu 램에 할당하여 numpy로 다시 반환
            ### 이때 duplicate_net.shape은 1 x n_actions 인 array가 됨.

            #반복문 생성, duplicate_net의 아웃풋단에서, action중 K_prime을 넘는 샤드갯수를 가진 액션에 대해 모두 0값 처리.
            shard_set = [1, 2, 3, 4, 5, 6, 7, 8]
            for i in range(0, K_prime):
                del(shard_set[0])   ## K_Prime만큼 인덱스제거, 이 연산 후 shard_set은 action bound를 넘은 shard_set이됨.
            ## ex, K_prime이 3이면, shard_Set은 [4,5,6,7,8]만 남는다. 이때, 4,5,6,7,8 의 샤드를  갖는 모든 action을 duplicate_net에서 제거해줘야함

            for i in range(0, 512):
                a = i // 128  # block size (0~3)  2, 4, 6, 8
                b = (i - 128 * a) // 16  # # of shard (1~8)  1, 2 ,3 ,4 ,5 ,6, 7, 8
                n_shard=b+1
                if n_shard in shard_set: # 만약 선택된 action의 샤드갯수가, shard_set(bound 넘는) list에 있다면
                    duplicate_net[0][i] = 0  # 해당 값을 0으로 초기화

            best_action = torch.as_tensor(duplicate_net.argmax()).to(device).view(1,1)


            return best_action

    else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
           ## 리턴값이  torch.tensor.
           ## select action을 통해 torch.tensor형으로 감싸진 action 이 나오는데, 이 값이 env.step으로 다시들어감.


Total_reward = []
Total_ratio = []
def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(Total_reward, dtype=torch.float)
    durations_t2 = torch.tensor(Total_ratio, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('TPS')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


######################################################################

# Training loop
# ^^^^^^^^^^^^^
## Finally, the code for training our model.
## Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` Tensor. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes, such as 300+ for meaningful
# duration improvements.
#

num_episodes = 2000
for i_episode in range(num_episodes):
    state, e_pp = get_screen(env.reset())    # state값은 1x4x200x200
  #  print('state는', state)
    print('e_p는 ', e_pp)

    for t in count():
        # Select and perform an action
        action = select_action(state, e_pp)             ## action은 select action으로부터 tesnor형 자료를 받아오는데
        action = torch.as_tensor(action)

        Next_pre_state, reward, done, const,_ = env.step(action.item())  ## .item을 통해 다시 integer가 반환이 된다.
        reward = torch.tensor([reward], device=device)
        # 조건문. env.step_beyond_done 이 None이라면, 해당 env는 초기 랜덤한 인자에 의해 reward가 강제로 0이되므로,
        # env.step beyond를 env로부터 리턴하여, 이 값이 0이 아닌경우에만 reward로 추출
        Total_reward.append(reward)
        Total_ratio.append(const[-1])

        # Observe new state
        if not done:
            next_state, _ = get_screen(Next_pre_state)
        else:
            next_state = None
        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            #plot_durations()
            break

        plot_durations()
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
       ####################
        ABC = torch.tensor(Total_reward, dtype = torch.float) ## [tensor(3), tensor(4), ...]
        ABC2 = torch.tensor(Total_ratio, dtype = torch.float)
        T_r = pd.DataFrame(ABC.numpy())
        T_r.to_excel('TPS_result.xlsx')
        T_ratio = pd.DataFrame(ABC2.numpy())
        T_ratio.to_excel('ratio_result.xlsx')
        print("file created!!")


print('Complete')
plt.ioff()
plt.show()

######################################################################
# Here is the diagram that illustrates the overall resulting data flow.
#
# .. figure:: /_static/img/reinforcement_learning_diagram.jpg
#
# Actions are chosen either randomly or based on a policy, getting the next
# step sample from the gym environment. We record the results in the
# replay memory and also run optimization step on every iteration.
# Optimization picks a random batch from the replay memory to do training of the
# new policy. "Older" target_net is also used in optimization to compute the
# expected Q values; it is updated occasionally to keep it current.
#



## Fig.2 batch size 2배, 256 ,layer 마지막 64

## Fig.3