import numpy as np 
import time
import sys
import random
# sys.path.append("/Users/Juhn/Documents/gym/gym/envs/toy_text")
sys.path.append("../")
from envs.mdp_gridworld import MDPGridworldEnv

def e_policy(Max,e):
	ran = random.uniform(0,1)
	a = random.randint(0,3)
	if ran > 1-e:
		output = a
	else:
		output = Max
	return output
	pass
def optimal_action(state,Q):
	Q_s_max = Q[(state,0)]
	a = 0
	tempQ = 0
	for i in range(4):
		tempQ = Q[(state,i)]
		if tempQ > Q_s_max:
			Q_s_max = tempQ
			a = i
	return a
	pass
def Q_Learning(env,alpha,gama,epsido,countall):
	count = 0
	iteration = 0
	Q = {(s,a):0 for s in range(env.observation_space.n) for a in range(env.action_space.n)}	
	Pi = {s:[0,1,2,3] for s in range(env.observation_space.n)}
	while True:
		Converge = True
		s=env.reset()
		terminal = False
		while True:		
			a = optimal_action(s,Q)
			#e-greedy:
			a = e_policy(a,0.05)
			if Pi[s] != a:
				Converge = False
			Pi[s] = a
			s1, reward, terminal, _ = env.step(a)			
			a1 = optimal_action(s1,Q)
			# Updata Q value
			Q[(s,a)] = Q[(s,a)] + alpha * (reward + gama*Q[(s1,a1)]-Q[(s,a)])
			s=s1
			if terminal is True:
				break
		if Converge == True:
			count = count + 1
		else:
			count = 0
		iteration += 1
		if count > countall:
			break
	return Pi, iteration
	pass
def Readable_Policy(Pi):
	orientation={0:'^',1:'v',2:'<',3:'>'}
	output=[]
	for i in Pi:
		if i==3 or i==5 or i==7:
			Pi[i]="--"	
		else:
			Pi[i]=orientation[Pi[i]]
		output.append(Pi[i])				
	return output
	pass
	
if __name__ == "__main__":
	env = MDPGridworldEnv()
	Pi,iterations=Q_Learning(env,0.1,0.9,0.2,8)
	Pi=Readable_Policy(Pi)
	print("total iterations:",iterations)
	print("=====Finial Policy====")
	print(Pi[0:4])
	print(Pi[4:8])
	print(Pi[8:12])