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
		self.action_n = env.action_space.n
		# self.Return={(s,a):[] for s in range(self.space_n) for a in range(self.action_n)}
		self.Q_value={(s,a):0 for s in range(self.space_n) for a in range(self.action_n)}
		self.eligy = {(s,a):0 for s in range(self.space_n) for a in range(self.action_n)}
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

	def get_E_greedy_action(self,state,epsilon): 
		if random.random()<1-epsilon:
			# return self.get_a_policy_action(state)
			return self.argmax_a_Q(state)
		else:
			return random.randint(0,self.action_n-1)

	def Q_learning(self,alpha,gamma,epsilon):
		max_converge_num =20
		count = 0
		iter_count = 0
		self.reset_val()
		while True:
			Converge = True
			state=self.env.reset()
			terminal = False
			while True:		
				act=self.get_E_greedy_action(state,epsilon)
				if self.get_a_policy_action(state) != act:
					Converge = False
				self.set_policy(state,[act])
				state1, reward, terminal, _ = env.step(act)			
				act1 = self.argmax_a_Q(state1)
				# Updata Q value
				self.Q_value[(state,act)]=self.Q_value[(state,act)] \
											+alpha* (reward + gamma*self.Q_value[(state1,act1)]-self.Q_value[(state,act)])
				state=state1
				if terminal is True:
					break
			count = count+1 if Converge else 0
			iter_count += 1
			if count > max_converge_num:
				break
		return iter_count

	def Q_lambda_learning(self,lamda,alpha,gamma,epsilon):
		max_converge_num =20
		count = 0
		iter_count = 0
		self.reset_val()
		while True:
			Converge = True
			state=self.env.reset()
			terminal = False
			act=self.get_a_policy_action(state)
			while True:
				state_p, reward, terminal, _ = env.step(act)
				act_star = self.argmax_a_Q(state_p)
				act_p    = self.get_E_greedy_action(state_p,epsilon)
				if self.get_a_policy_action(state) != act_p:
					Converge = False
				self.set_policy(state,[act_p])
				delta = reward + gamma * self.Q_value[(state_p,act_star)] - self.Q_value[(state,act)]
				self.eligy[(state,act)]+=1
				for s in range(self.space_n):
					for a in range(self.action_n):
						self.Q_value[(s,a)] += alpha*delta*self.eligy[(s,a)]
						self.eligy[(s,a)] = gamma * lamda *self.eligy[(s,a)] if act_p == act_star else 0
				state = state_p
				act   = act_p
				if terminal is True:
					break
				count = count + 1 if Converge else 0
			iter_count += 1
			if count > max_converge_num:
				break
		return iter_count

	def argmax_a_Q(self,state):
		action_list = []
		mx=-10000
		for act in range(self.action_n):
			mx = self.Q_value[(state,act)] if mx < self.Q_value[(state,act)] else mx
		for act in range(self.action_n):
			if mx == self.Q_value[(state,act)]:
				action_list.append(act)
		
		return action_list[0]

	def show_policy(self):
		arraw_dic={0:'^',1:'v',2:'<',3:'>'}
		arrows =[]
		for state_policy in self.policy:
			p_arraw =[]
			for j in range(len(state_policy)):
				p_arraw.append(arraw_dic[state_policy[j]])
			arrows.append(p_arraw)
		arrows[3] = ['T']
		arrows[5] = ['X']
		arrows[7] = ['T']
		print ("--------------Policy------------")
		print (arrows[0:4])
		print (arrows[4:8])
		print (arrows[8:12])
		print (" ")
	def Q_lambda_learning_TEST(self,alpha,gamma,epsilon):
		lamda_list =list(frange(0,1,0.1))
		# print (lamda_list)
		for lamda in lamda_list:
			iter_count=self.Q_lambda_learning(lamda,alpha,gamma,epsilon)
			print ('----- lamda == {}  converge at iteration {}  -----'.format(lamda,iter_count))
			self.show_policy()
def frange(x, y, step):
  while x < y:
    yield x
    x += step
if __name__ == "__main__":
	# print  ("--------------Deterministic action state ------------ ")
	env = MDPGridworldEnv(act_mode='deterministic')
	Q_L =my_Qlearning(env)
	alpha 	=	0.1
	gamma 	=	0.9
	epsilon = 	0.05
	lamda   = 	0.6
	iter_count=Q_L.Q_learning(alpha,gamma,epsilon)
	print('')
	print  ("============   Q learning ================ ")
	print('Converged at iteration {}'.format(iter_count))
	Q_L.show_policy()

	print  ("============  Q lamda learning =========== ")
	Q_L.Q_lambda_learning_TEST(alpha,gamma,epsilon)
	# iter_count=Q_L.Q_lambda_learning(lamda,alpha,gamma,epsilon)
	# print('Converged at iteration {}'.format(iter_count))
	# Q_L.show_policy()