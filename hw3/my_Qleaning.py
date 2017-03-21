import numpy as np 
import time
import sys
import random
import ipdb
# sys.path.append("/Users/Juhn/Documents/gym/gym/envs/toy_text")
sys.path.append("../")
from envs.mdp_gridworld import MDPGridworldEnv
class my_Qlearning:
	def __init__(self,env):
		self.env      = env #(act_mode='stochastic')
		self.reset_val()
	def reset_val(self):
		self.space_n  = env.observation_space.n
		 = env.action_space.n
		self.Return={(s,a):[] for s in range(self.space_n) for a in range(self.action_n)}
		self.Q_value={(s,a):0 for s in range(self.space_n) for a in range(self.action_n)}
		act =range(0,self.action_n)
		self.policy = [act for j in range(self.space_n)]
	def get_policy(self,state):
		return self.policy[state]
	def set_policy(self,state,actions):
		self.policy[state]=actions
	def get_a_policy_action(self,state):
		act_list=self.policy[state]
		idx=random.randint(0,len(act_list)-1)
		return act_list[idx]
	def get_E_greedy_action(self,sate,ratio):
		if random.random()<1-ratio:
			return self.get_a_policy_action(state)
			# return self.argmax_a_Q(sate)
		else:
			return random.randint(0,self.action_n-1)
	def q_learning(self,aplha,reward,gama):
		self.reset_val()
		while True:
			Converge = True
			state=self.env.reset()
			terminal = False
			while True:		
				act=self.get_E_greedy_action(self,state,0.05)
				# if Pi[s] != a:
				# 	Converge = False
				# Pi[s] = a
				# self.set_policy
				sate1, reward, terminal, _ = env.step(act)			
				act1 = self.argmax_a_Q(s1)
				# Updata Q value
				self.Q_value[(state,a)]+=alpha * (reward + gama*Q_value[(state1,act1)]-Q[(state,act)])
				state=state1
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
	def argmax_a_Q(self,state):
		action_list = []
		mx=-10000
		for act in range(self.action_n):
			mx = self.Q_value[(state,act)] if mx < self.Q_value[(state,act)] else mx
		for act in range(self.action_n):
			if mx == self.Q_value[(state,act)]:
				action_list.append(act)
		
		return [action_list[0]]

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
	ipdb.set_trace()
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