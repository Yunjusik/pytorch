import gym
import math
import random
import scipy.optimize
import numpy as np
'''
#해당 버전은 Worst case 분석을 위한 별도의 shard분배
Worst case는 모든 malicious shard가 DC에 분포됨.
Success ratio  --> consensus success ratio 
False Consensus Probability (FCP) --> malicious shard clustering. --> 말리셔스 노드가 DC인원의 2/3이 넘는지 판단

FCP는 두가지 조건문으로 계산
1. block producer가 honest -> malicious가 1/3만 있으면 거절가능  +1
2. block producer가 malicious -> malicious majority (2/3) 에 의해 승낙
어떤 경우이던지 malicious shard가 되는경우를 count하여 FCP를 얻음



'''
class ShardDistribute(gym.Env):
    def __init__(self):
        # Simulation parameters
        self.nb_nodes = 200
        self.nb_MalNode = 30 #round(200*random.randrange(1,21)*0.01)
        self.H_history = []
        self.e_prob = []
        self.NodesinShard = []
        self.Success_ratio = 1
        self.FCP = None

    def ShardDist(self, Shard_numb):
        nb_nodes = self.nb_nodes
        nb_MalNode = self.nb_MalNode
        NodeType = {} ## NodeType이 0이면 Malicious Node, 1이면 Honest Node
        Shard = {} ## Node들의 샤드 번호 저장
        ShardCount = {} ## 랜덤분배를 위해 샤드에 들어간 노드 수 카운트
        MaxShardSize = math.ceil( nb_nodes / (Shard_numb + 1) )
        H_history = [] ## 노드들의 commit 의견이 major or miner 인지 저장
        U = 0 ## Entropy

        FCP = 0
     #   r_mal = nb_MalNode/nb_nodes # 실제 mal 비율




        ## Node Type 설정        
        for i in range(nb_MalNode):  # 0부터 mal node까지 0번. (mal shard)
            NodeType[i] = 0
        for i in range(nb_MalNode,nb_nodes): # mal num부터 끝까지 정상노드로 인덱싱
            NodeType[i] = 1


        if (Shard_numb >= 2):

            ## Node들의 샤드 랜덤 배치
            for i in range(Shard_numb + 1):
                ShardCount[i] = 0          # k=2라고하자, 0,1,2 index에 0을 부여. (초기화용인듯)
            for i in range(nb_nodes):
                tmp = random.randint(0,Shard_numb)  # 0,1,2중 하나 뽑고
                while (ShardCount[tmp] >= MaxShardSize):  # shardcount[0 or 1 or 2] >= maxshardsize 인 동안
                    tmp = random.randint(0, Shard_numb)      ## 만약 처음에 tmp가 0이라고하자. while문은 성립 안하므로 shard[1]=0이되고 shardcount[1]=1이됨.
                Shard[i] = tmp                         ## 즉 shard[i]는 노드 i+1이 갖는 샤드번호이고, shardcount[tmp]는 tmp+1번째 샤드의 노드갯수임.
                ShardCount[tmp] += 1
               ### 다 돌고나면 shardcount[i] 는 i번째샤드의 노드수가 dictionary 형태로 저장되어있음.


            ## 각 샤드에 속하는 노드들의 INDEX 저장
            NodesInShard = {}
            for K in range(Shard_numb + 1):
                NodesInShard[K] = []   # k-1번째 샤드에 속하는 노드 index 초기화
            for i in range(nb_nodes):
                NodesInShard[Shard[i]].append(i)          ##

    
            ## 랜덤으로 각 샤드에서 블록 생성자 선출
            BlockProposer = {}
            for i in range(Shard_numb + 1):
                tmp = random.randint(0,len(NodesInShard[i])-1)
                BlockProposer[i] = NodesInShard[i][tmp]
                                    


            DC_num = ShardCount[Shard_numb]
            if (2/3)*DC_num <= self.nb_MalNode: # 악의적인 노드가 2/3이상된 경우 .. -> 무조건 malicious shard
                FCP = 1
            elif (NodeType[BlockProposer[Shard[i]]] == 1) and ((1/3)*DC_num <= self.nb_MalNode) :
                ## block producer는 honest이나, malicious node가 1/3이상인 경우
                FCP = 1
            else:
                FCP = 0
            print(DC_num,FCP) ## DC 숫자와, FCP 추출.


    
            ## 노드들의 Commit Step에서 결과 저장 (0은 반대, 1은 찬성)
            ShardConsensus = {}
            for K in range(Shard_numb + 1):
                ShardConsensus[K] = []   ## 샤드번호 K의 consensus에 대한 list를 새로 작성하고
                ##노드가 속한 Shard의 블록생성자가 Honest (정상 블록 생성)이면,
                # Honest는 찬성 (1), Malicious는 거절 (0)
            ##노드가 속한 Shard의 블록생성자가 Malicious (비정상 블록 생성)이면,
                # Honest는 거절 (0), Malicious는 찬성 (1)
            for i in range(nb_nodes):   #
                if ( NodeType[BlockProposer[Shard[i]]] == 1 ): # 정상블록
                    ShardConsensus[Shard[i]].append(NodeType[i])
                else:
                    ShardConsensus[Shard[i]].append(1-NodeType[i])
    
            ##정상 블록 생성을 성공한 Shard 개수 세기
            Success_numb = 0
            for K in range(Shard_numb + 1):
    
                # 블록생성자가 Honest이면서 & 찬성에 투표한 수가 2/3 이상인 경우에만 정상 블록 생성
                if ( (NodeType[BlockProposer[K]] == 1) & ( (ShardConsensus[K].count(1)) >= (2/3)*len(ShardConsensus[K]) ) ):
                    Success_numb += 1        
            ## 정상블록생성한 Shard개수 / 전체 Shard 개수 비율 계산
            Success_ratio = Success_numb / (Shard_numb + 1)
    
                
    
    
            # (H[i]는 i번째 노드의 commit 의견이 major인지 miner인지 나타낸 값.
            # major의견은 1, minor 의견은 0. -> 이값은 signature에 의해 부인방지가 보장된다고 가정
            # 합의 기록 H는, predefined된 malicious 비율에 따라 샤드분배환경에서의 합의기록
            for i in range(nb_nodes):
                if ((ShardConsensus[Shard[i]].count(NodeType[i])) >= 0.5*len(ShardConsensus[Shard[i]])):
                    H_history.append(1)
                else :
                    H_history.append(0)
    
    
            # H를 통해 e__prob(estimated faulty prob) 계산
    
            for K in range(Shard_numb + 1):
                if ((ShardConsensus[K].count(0)) >= 0.5*len(ShardConsensus[K])):   ## 0이 과반수인경우
                    major_ratio = (ShardConsensus[K].count(0)) / (len(ShardConsensus[K]))
                    miner_ratio = 1 - major_ratio
    
                else:
                    miner_ratio = (ShardConsensus[K].count(0)) / (len(ShardConsensus[K])) # 1이 과반수인 경우
                    major_ratio = 1 - miner_ratio        
                    
                if ((miner_ratio == 1) or (major_ratio == 1)): ########################################## p=0이면 binary entropy는 0
                    U += 0
                else:
                    #print(miner_ratio, major_ratio)
                    U += -miner_ratio*math.log2(miner_ratio) - major_ratio*math.log2(major_ratio)
    
            U = U / (Shard_numb + 1)
    
    
            def myfunc(x):
    
                return x*np.log2(x)+(1-x)*np.log2(1-x) + U
    
            if U >= 0.42:
                e_prob = scipy.optimize.fsolve(myfunc, x0=0.25)
            elif 0.4<= U <= 0.42:  # k= 1~3
                e_prob = 0.08
            elif 0.3<= U < 0.4:   #k= 1~5
                e_prob = 0.05
            elif 0.2 <= U <0.3:   # k= 1~8
                e_prob = 0.03
            else:
                e_prob = 0.025     # k= 1~8
           # print(e_prob)
    
            # ShardDist함수를 호출할때마다 샤드 shuffling 이 진행되고, 새로운 R, C, H, e_prob 을 리턴
            H_history = np.array(H_history)
            H_history = H_history.reshape(200, 1) ## 200x1형 shape으로 만들고
            self.H_history = np.kron(H_history, np.ones((1, 200))) # 200x200으로 확장한뒤 H_his에 대입
            self.e_prob = e_prob*np.ones((200, 200))  # 200x 200 확장
            self.NodesInShard = NodesInShard
            self.Success_ratio = Success_ratio


        else : # Shard_numb = 1 일 때,
            for i in range(nb_nodes):
                H_history.append(NodeType[i]) #malicious node는 minor 의견이므로 0, honest는 1
                
            e_prob = ( nb_MalNode / nb_nodes )
            
            NodesInShard = {}
            NodesInShard[0] = []   # k-1번째 샤드에 속하는 노드 index 초기화
            for i in range(nb_nodes):
                NodesInShard[0].append(i)          ##       
                
            if (  random.random() <= (nb_MalNode / nb_nodes) ):
                Success_ratio = 0
            else:
                Success_ratio = 1
                
            # ShardDist함수를 호출할때마다 샤드 shuffling 이 진행되고, 새로운 R, C, H, e_prob 을 리턴
            H_history = np.array(H_history)
            H_history = H_history.reshape(200, 1) ## 200x1형 shape으로 만들고
            self.H_history = np.kron(H_history, np.ones((1, 200))) # 200x200으로 확장한뒤 H_his에 대입
            self.e_prob = e_prob*np.ones((200, 200))  # 200x 200 확장
            self.NodesInShard = NodesInShard
            self.Success_ratio = Success_ratio


        return self.H_history, self.e_prob, self.NodesInShard, self.Success_ratio, FCP


# 200x200 size의 H,e_prob를 반환.