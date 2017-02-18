#import gym
  # Uncomment if using gym.make
import time
import sys
sys.path.append("../")
from envs.mdp_gridworld import MDPGridworldEnv

# Uncomment if you added MDPGridworld as a new gym environment
#env = gym.make('MDPGridworld-v0')
# You have to import MDPGridworldEnv properly in order for environment to work
env = MDPGridworldEnv()

# prints out that both states and actions are discrete and their valid values
print env.observation_space
print env.action_space

# to access the values
print env.observation_space.n # env.nS
print env.action_space.n # env.nA

# state = 8
# action = 0 # North
# print env.P[state][action]

#added delay here so you can view output above
time.sleep(2)
dp_score= [[0 for i in range(4)] for i in range(3)]
act=[0,1,2,3]
dp_action= [[act for i in range(4)] for i in range(3)]
# print dp_score
# dp_action[1][1]= [1,2]
# print dp_action
dp_score_new= [[0 for i in range(4)] for i in range(3)]



for i_iter in range (100):
    obs=env.reset()
    while True:
        det=0
        for s in range (12):
            i= s/4
            j= s%4
            value= dp_score[i][j]

            size= len(dp_action[i][j])
            dp_score_new[i][j]=0
            for index_action in range(size):
                # print dp_action[i][j][index_action]
                # print env.P[s][dp_action[i][j][index_action]]
                nex= env.P[s][dp_action[i][j][index_action]][0][1]
                r= env.P[s][dp_action[i][j][index_action]][0][2]
                # print nex
                # print nex/4
                # print nex%4
                # print len(dp_score)
                # pr  
                dp_score_new[i][j]=dp_score_new[i][j]+ 1.0/size*(r+ 0.6*dp_score[nex/4][nex%4])     

            det= max(det, abs(dp_score_new[i][j]- value))

       # print det
        dp_score=dp_score_new
       # print dp_score 
       # print det    
       # time.sleep(1)
        if det< 0.1:
            break
    policy_stable = True
    for s in range(12):
        i= s/4
        j= s%4
        temp_list= dp_action[i][j]
       # print temp_list
        size= len(dp_action[i][j])
        maxscore= -100
        temp=[]
        for idx_action in range(size):
            nex= env.P[s][temp_list[idx_action]][0][1]
            r= env.P[s][temp_list[idx_action]][0][2]
            score= 1.0/size*(r+0.6*dp_score[nex/4][nex%4])
          #  print "score"
          #  print score
            if score> maxscore:
                del temp[:]
                temp.append(temp_list[idx_action])
                #print temp
                maxscore= score
            elif score == maxscore:
                temp.append(temp_list[idx_action])

        if temp !=temp_list:
            dp_action[i][j]= temp
            policy_stable= False
   # print dp_action
    if policy_stable == True:
        break
    






print dp_action
print dp_score
print i_iter


# for i_episode in range(20):
#     obs = env.reset()
#     for t in range(100):
#         env.render()
#         # time.sleep(.5) # uncomment to slow down the simulation
#         action = env.action_space.sample() # act randomly
#         obs2, reward, terminal, _ = env.step(action)
#         if terminal:
#             env.render()
#             print("Episode finished after {} timesteps".format(t+1))
#             break
