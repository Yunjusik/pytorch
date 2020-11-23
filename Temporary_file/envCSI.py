import gym
import math
import random
import numpy as np

class env_CSI(gym.Env):
    def __init__(self):
        # CSI parameters
        self.expansion_level = 10
        self.nk_max = 20
        self.n=20
        self.m =2 # RAT, LTE:0, 5G midband:1
        self.best_RAT = 0
        self.radius = 200
        self.service_ratio = 0.5
        self.N_ku = self.n * self.service_ratio  ## the number of URLLC User
        self.N_ke = self.n - self.N_ku           ## the number of eMBB user

    def CSI_reset(self, UEtype):
        best_RAT = 0
        expansion_level = self.expansion_level
        distance = random.randint(1, 200)
        # define large scale channel gain (LSC)
        path_loss = 35.3 + 37.6 * math.log10(
            distance) + 8  ## db scale,, path_loss (35.3+37.6log10(d)) with shadowing (8db)
        LSCG = 10 ** -(path_loss / 10)  ## Large scale channel gain
        # channel gain matrix
        H = np.zeros((1, self.m * expansion_level))
        for i in range(self.m):
            SSCG = np.random.rayleigh()  ## rayleigh fading coefficient (small-scale-channel-gain)
            for j in range(expansion_level):
                H[0, i * expansion_level + j] = SSCG * LSCG  ### channel gain matrix H => array (1,m * expansion) size
                if H[0, best_RAT * expansion_level] <= H[0, i * expansion_level]:
                    best_RAT = i  ## best RAT change according to channel gain

        # Job generation (five entity). URLLC 랑 eMBB는 여기서 차이점 발생
        if UEtype == 'urllc' :
            # URLLC packet size...
            J = np.zeros((5 * expansion_level, 1))  # (5*expansion x 1) size job state
            lam = 0.01  # packet/slot
            b_n = 32 * 8  # 32 byte. 256 bit
            c_n = 330 * 32  # required Cpu cycle for processing URLLC Packet
            Q1_t = 0.001  ## URLLC time limit  (1ms)
            Q2_e = 10 ** -7  ## URLLC packet loss probability threshold
            stack = [lam, b_n, c_n, Q1_t, Q2_e]
            for i in range(5):
                for j in range(expansion_level):
                    J[i * expansion_level + j, 0] = stack[i]
            # J --> 5*expan x 1 size numpy array representing Job state
            BW = np.zeros((1, 1))
            BW_slicing = np.random.rand()
            # for LTE, 서브캐리어 스페이스 = 15khz
            # for 5G NR, 서브캐리어 스페이스 = 15, 30, 60, 120khz
            # 1RB에 12 섭캐
            # 구상, 채널 스테이트 인포메이션에서 채널게인이 가장 좋은게 몇번째 RAT인지 계산
            if best_RAT == 0:  ## for LTE case, let assume 50 RB exist
               # RB_BW = 180 * 10 ** 3  # 180khz
                RB_max = 50  # 50 RB
                RB = math.floor((BW_slicing * RB_max) / self.N_ku)
            else:
                #RB_BW = 180 * (10 ** 3)  # 180khz
                RB_max = 100  # 50 RB
                RB = math.floor((BW_slicing * RB_max) / self.N_ku)
                ### RB가 0인경우,, 로컬프로세싱을 하게됨.
            BW[0,0] = RB
            BW = np.kron(BW, np.ones((50,20)))  ## 50 x 20

        if UEtype == 'eMBB': # for eMBB case,
            Randn = random.random()
            J = np.zeros((5 * expansion_level, 1))  # (5*expansion x 1) size job state
            lam = 0.01  # packet/slot
            b_n = 50*(1 + Randn) * 8 * 1024  # 50kbyte ~ 100 kbyte (bit scale)
            c_n = 330 * (b_n/8)   # required Cpu cycle for processing full eMBB Packet
            Q1_t = 0.01  ## eMBB time limit  (10ms)
            Q2_e = 1  ## eMBB packet loss probability threshold
            stack = [lam, b_n, c_n, Q1_t, Q2_e]
            for i in range(5):
                for j in range(expansion_level):
                    J[i * expansion_level + j, 0] = stack[i]

            # J --> 5*expan x 1 size numpy array representing Job state
            BW = np.zeros((1, 2))
            BW_slicing_1 = random.random() ## RAT1 slicing
            BW_slicing_2 = random.random() ## RAT2 slicing
            RB_max_1 = 50
            RB_max_2 = 100
            RB_1 = (BW_slicing_1 * RB_max_1 / self.N_ke)
            RB_2 = (BW_slicing_2 * RB_max_2 / self.N_ke)
            BW[0,0] = RB_1
            BW[0,1] = RB_2
            BW = BW.repeat(5,axis=0)
            BW = np.kron(BW, np.ones((expansion_level,expansion_level)))
            ''' BW allocation for eMBB shape
                                    BW matrix
            [RB of RAT1, RB of RAT1, ... ... ...,RB of RAT2,RB of RAT2]
            [RB of RAT1, RB of RAT1, ... ... ...,RB of RAT2,RB of RAT2]
            [RB of RAT1, RB of RAT1, ... ... ...,RB of RAT2,RB of RAT2]
            [RB of RAT1, RB of RAT1, ... ... ...,RB of RAT2,RB of RAT2]
            
            '''
        J = J.repeat(20, axis=1)  ## 50 x 20
        H = H.repeat(50, axis=0)  ## 50x 20
        self.state_u = [J, H, BW]  ## 50x20짜리 state가 3개 있는 list

        return self.state_u, stack, best_RAT
