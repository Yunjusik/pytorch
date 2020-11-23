import gym
import math
from gym import spaces, logger
from gym.utils import seeding
import random
import numpy as np
from action_space_UE_U import ActionSpace_U
from action_space_UE_E import ActionSpace_E
from action_space_M import ActionSpace_M
from action_space import ActionSpace
from ShardDistribution2 import ShardDistribute


class Offenv(gym.Env):
    '''
Actions for UE (eMBB):
        1) local UE frequency (f) : Discrete l-level - f1, f2, ..... (fmax)
                                   [params : min: fmin, madx : fmax]
        2) offloading ratio (o)   : # of RAT : m, total m+1  slot(local(1) + RAT(m)
                                    combination with repetition problem, m+1(H)l = m+l(C)l --> (m+l) combination (l)
                                    if m = 2, l=5, 7 C 5 = 21 space = (6+5+4+3+2+1)
                                    [params: min: 0 , max: 1]
        3) transmit power (p)     : Individual Power allocation, p1,~ pl, l-slot.
                                  : l x l size.

                                total action size --> [l] * [m+1(C)l] * [l^2]
                                if 2RAT and 5 level,  5^3 X 21 = 2625
#
Actions for UE (URLLC):
        1)local UE frequency (f) : l-level
        2)offloading rato (o) : send data to best RAT or locally process, therefore, 1+l C l,
                                if l=5, 6C5 = 6
        3)transmit power (p) : l-level (only consider 1 RAT)
                            total actions size --> l * 1+l * l = 150 size

Actions for MEC server:

        1) frequency slicing indicator fk = (0.1, 0.2, 0.3, ... 0.9)
        2) RB slicing indicator Nmk       = (0.1, 0.2, 0.3, ... 0.9)
        total action --> 9x9 = 81



state space (UE):
    Type:
    Num       state                    Min       Max     format
    0         Task Job                                     5X1   (lambda, b,c,Q1,Q2)
    1      CSI matrix of UE                                1xm
    2            BW                                        1 (will be expanded to 5xm)
    --> total concatenated state -> 5 x m x 3

state space (MEC SERVER)

    Type:
    Num       state                          format
    0       task window               N^max_k * omega * 4
    1   # of offloaded task                     2x1
        --> we use task window only...


:
    Type: Box(2)
    num     observation        min       max
    0        latency           0         48
    1   required shard limit   1          8

    '''

    def __init__(self):
        # 시뮬레이션 파라미터 설정.
        # 엣지서버 한대에서 노드수 Nkmax를 설정

        self.nk_max = 20  ## 노드 최대치
        self.n = 20
        self.m = 2   ## RAT의 숫자
        self.radius = 200 ## gNB 전송반경

        # defince action sapce & abservation space
        self.action_space_UE_U = ActionSpace_U(150)
        self.action_space_UE_E = ActionSpace_E(2625)
        self.action_space_M = ActionSpace_M(81)
        #self.observation_space = spaces.Box(low=np.array([0, 1]), high=np.array([48, 8]), dtype=np.float32)

        self.Service_ratio = 0.5

##initialize all state

        self.seed()
        self.viewer = None


        self.state_u = None
        self.state_e = None
        self.state_m = None
        self.steps_beyond_done = None

        # state 불러오기. ShardDist함수의 return값을 각각 R,c,H,e_prob로 할당하도록함.
        # 여기서 state는 main의 진짜 state에 필요한 구성요소들을 각각 업데이트하는것임.
        self.R_transmission = None
        self.c_computing = None
        self.H_history = None
        self.e_prob = None
        self.reward = 0

        self.BW = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step_u(self, action_u):
        ## 전체 step process 과정 ##
        # 1. 노드 세부사항 설정 (거리)
        # 2. CSI matrix 생성
        # 3. job 생성
        # 4. UE/MEC의 state state 생성. state 생성시, BW값을랜덤으로 할당
        # 5.
        ##전체 프로세스. 알고리즘은 먼저 UE DNN을 학습시키고, 일정에포크 이상 training loss가 줄어들지 않는경우 stop.. stop에 대한 기준이 있나 찾아보기.
        ## stop한후 메모리를 모두 비우고, 남은 잔여에너지를 MEC서버를 학습시킨다.
        assert self.action_space_UE_U.contains(action_u), "%r (%s) invalid" % (action_u, type(action_u))
        state = self.state_u

        J, H, BW  = state


        # URLLC service type index
        N_ku = self.n * self.Service_ratio  ## the number of URLLC User

        # distance --> 각 노드의  거리를 담은 어레이 , gNB 전송반경 200m 가정
        # distance = {}
        # for i in range(N_ku + 1):
        #   distance[i] = random.randint(1, 200)  # distance[i] return distance of ith user
        # user 한명의 입장만 대변
        distance = random.randint(1, 200)
        # define large scale channel gain (LSC)
        path_loss = 35.3 + 37.6 * math.log(
            distance) + 8  ## db scale,, path_loss (35.3+37.6log10(d)) with shadowing (8db)
        LSCG = 10 ** -(path_loss / 10)  ## Large scale channel gain
        # channel gain matrix
        H = np.zeros((1, self.m))
        for i in range(self.m):
            SSCG = np.random.rayleigh()  ## rayleigh fading coefficient (small-scale-channel-gain)
            H[0, i] = SSCG * LSCG  ### channel gain matrix H => array (1,m) size
        # Job generation (five entity)
        # URLLC packet size...
        J = np.zeros((5, 1))  # 5x1 size job state
        lam = 0.01  # packet/slot
        b_n = 32 * 8  # 32 byte. 256 bit
        c_n = 330 * 32  # required Cpu cycle for processing URLLC Packet
        Q1_t = 1  ## URLLC time limit  (1ms)
        Q2_e = 10 ** -7  ## URLLC packet loss probability threshold

        J[0, 0] = lam
        J[1, 0] = b_n
        J[2, 0] = c_n
        J[3, 0] = Q1_t
        J[4, 0] = Q2_e
        # J --> 5 x 1 size numpy array representing Job state



        # for i in range(N_ku + 1):
        #   distance[i] = random.randint(1, 200)  # distance[i] return distance of ith user
        # user 한명의 입장만 대변
        best_RAT = 0
        expansion_level = 10
        distance = random.randint(1, 200)
        # define large scale channel gain (LSC)
        path_loss = 35.3 + 37.6 * math.log10(distance) + 8  ## db scale,, path_loss (35.3+37.6log10(d)) with shadowing (8db)
        LSCG = 10 ** -(path_loss / 10)  ## Large scale channel gain
        # channel gain matrix
        H = np.zeros((1, self.m * expansion_level))
        for i in range(self.m):
            SSCG = np.random.rayleigh()  ## rayleigh fading coefficient (small-scale-channel-gain)

            for j in range(expansion_level):
                H[0, i*expansion_level + j] = SSCG * LSCG  ### channel gain matrix H => array (1,m * expansion) size
                if H[0, best_RAT*expansion_level] <= H[0,i*expansion_level]:
                    best_RAT=i  ## best RAT change according to channel gain

        # Job generation (five entity)
        # URLLC packet size...
        J = np.zeros((5 * expansion_level, 1))  # (5*expansion x 1) size job state
        lam = 0.01  # packet/slot
        b_n = 32 * 8  # 32 byte. 256 bit
        c_n = 330 * 32  # required Cpu cycle for processing URLLC Packet
        Q1_t = 1  ## URLLC time limit  (1ms)
        Q2_e = 10 ** -7  ## URLLC packet loss probability threshold

        stack = [lam, b_n, c_n, Q1_t, Q2_e]

        for i in range(5):
            for j in range(expansion_level):
                    J[i * expansion_level + j,0] = stack[i]

        # J --> 5*expan x 1 size numpy array representing Job state

        BW = np.zeros((1,1))
        BW_slicing = np.random.rand()
        # for LTE, 서브캐리어 스페이스 = 15khz
        # for 5G NR, 서브캐리어 스페이스 = 15, 30, 60, 120khz
        # 1RB에 12 섭캐
        # 구상, 채널 스테이트 인포메이션에서 채널게인이 가장 좋은게 몇번째 RAT인지 계산
        if best_RAT == 0: ## for LTE case, let assume 50 RB exist
            RB_BW = 180 * 10**3 # 180khz
            RB_max = 50  # 50 RB
            RB = math.floor((BW_slicing * RB_max)/N_ku)

        else:
            RB_BW = 180 * (10 ** 3)  # 180khz
            RB_max = 100  # 50 RB
            RB = math.floor((BW_slicing * RB_max) / N_ku)

      ### RB가 0인경우,, 로컬프로세싱을 하게됨.

        BW[0,0] = RB
        # BW state (# of RB) --> 1x1 numpy array

        ## next concatenate overall space
        ## J,H,BW (50x1, 1x20, 1x1)
        ### state space --> 50 x 20 x 3

        ## then, J -> 50x1, H -> 1x 10, BW -> 50x10

        J = J.repeat(20,axis=1) ## 50 x 20
        H = H.repeat(50,axis=0) ## 50x 20
        BW = BW.repeat(50,20) ## 50 x 20

        self.state_u = [J, H, BW]  ## 50x20짜리 state가 3개 있는 list

        return self.state_u




    '''
          SNR 공식 미리 작성
        
        SNR = channel gain * power(23dbm) / noise 
        
        Noise_spectral_density = -174
        
        noise_power = 10 ** (-20.4)  #noise power w
        
        SNR = channel gain * power(23dbm = 0.199W) / noise_power
        
        data rate = BW * log2(1+SNR)
        
        
         예제, RB1개사용, 풀파워 23dbm 사용, 
         만약 이때 채널게인이 8.25510027e-13 인 경우,
         1412466 bps가나오고, 이를 URLLC 32바이트패킷을 보내는데 0.0001812433초가 걸림. -> 0.18ms.
         따라서 transmission delay는 만족
         생각해보니 이때 걸린 큐잉 딜레이를 반환한 후, MEC서버의 러닝때 사용해야하는것같음. 이때가 RB가 50개라그럼..?
         만약 20MHZ라고 가정하면 RB 100개를 나눠주는 상황. 좀더 여유롭게 가능할듯
