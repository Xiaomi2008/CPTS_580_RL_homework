import numpy as np
import random
import time
import sys
sys.path.append("../")
from envs.mdp_gridworld import MDPGridworldEnv

class mydp:
	def __init__(self,env):
		self.env      = env #(act_mode='stochastic')
		self.reset_val()
	def reset_val(self):
		self.space_n  = env.observation_space.n
		self.action_n = env.action_space.n
		self.Return={(s,a):[] for s in range(self.space_n) for a in range(self.action_n)}
		self.Q_value={(s,a):0 for s in range(self.space_n) for a in range(self.action_n)}
		act =range(0,self.action_n)
		self.policy = [act for j in range(self.space_n)]

	def monte_carlo_ES(self):
		self.reset_val()
		episode_count =0 
		while True:
			#state=self.env.reset()
			state =self.env_reset_random_starting_point()
			#print ('starting state is {}'.format(state))
			terminal=False
			episode_memory =[]
			step_count =0
			# (a)========== Generate an episode using exploring starts and Pi ============= 
			while not terminal:
				act=self.get_a_policy_action(state)
				next_state, reward, terminal, _ = self.env.step(act)
				episode_memory.append((state,reward,act))
				state=next_state
				if step_count > 1000:
					break
				step_count+=1

			# (b)========== For each pair s,a appearing in the episode ============= 
			episode_array =np.asarray(episode_memory)
			visited_states =np.zeros((self.space_n,self.action_n))
			for i in range(len(episode_memory)):
				s,r,a= episode_memory[i]
				
				if visited_states[s,a]==0:
					R=np.mean(episode_array[i:,1])
					#R = sum(r1*pow(0.9,i2) for i2,(_1,r1,_2) in enumerate(episode_memory[i:]))
					#R /=len(episode_memory[i:])
					#if s ==2 and a  ==3:
					#	print (R)
					#R=np.mean(episode_array[i:,1])
					self.Return[(s,a)].append(R)
					self.Q_value[(s,a)]=np.mean(self.Return[(s,a)])
					visited_states[s,a] =1
					#if s ==2:
					#	print ( 'U ={} , D ={}, L={}, R={}'.format(self.Q_value[(s,0)],self.Q_value[(s,1)],self.Q_value[(s,2)],self.Q_value[(s,3)]))
			#print(episode_memory)

			# (c)========== For each s in the episode ============= 
			states = np.unique(episode_array[:,0])
			for i in range(states.size):
				c_state =states[i,].astype(int)
				# Pi(s) <--- argmax_a_Q(s,a)
				self.set_policy(c_state,self.argmax_a_Q(c_state))
			if episode_count >3000:
				print ('total {} episodes'.format(episode_count))
				break
			episode_count+=1

	def env_reset_random_starting_point(self):
		# the reset function of environment seem  always  start from state 8
		state=self.env.reset()
		# To get a random starting state, just run random action at random steps
		step_random = random.randint(0,self.action_n*self.space_n-1)
		for i in range(step_random):
			act=random.randint(0,self.action_n-1)
			state, _1, _2, _3 = self.env.step(act)
		return state


	def get_a_policy_action(self,state):
		act_list=self.policy[state]
		idx=random.randint(0,len(act_list)-1)
		return act_list[idx]

	#def get_policy(self,state):
	#	return self.policy[state]

	def set_policy(self,state,actions):
		self.policy[state]=actions
		#if state == 2:
		#	print self.policy[state]

	def argmax_a_Q(self,state):
		action_list = []
		mx=-10000
		for act in range(self.action_n):
			mx = self.Q_value[(state,act)] if mx < self.Q_value[(state,act)] else mx
		for act in range(self.action_n):
			if mx == self.Q_value[(state,act)]:
				action_list.append(act)
		
		return action_list


	def update_policy(self):
		for state in range(self.space_n):
			self.set_policy(state,self.get_max_V_actions(state))


	# def show_result(self):
	# 	print "=============================="
	# 	print "--------- State value --------"
	# 	print self.Vstate[0:4]
	# 	print self.Vstate[4:8]
	# 	print self.Vstate[8:12]
	# 	print "  "

	# 	self.show_policy()

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

if __name__ == "__main__":
	print  ("--------------Deterministic action state ------------ ")
	env = MDPGridworldEnv(act_mode='deterministic')
	A =mydp(env)
	A.monte_carlo_ES()
	A.show_policy()
	print(" ")
	
	print ("--------------Stochastic action state  with  'High' randomness ---------------- ")
	env2 = MDPGridworldEnv(act_mode='stochastic',random_prob ='high')
	B =mydp(env2)
	B.monte_carlo_ES()
	B.show_policy()

	print (" ")
	print ("--------------Stochastic action state  with  'Low' randomness ---------------- ")
	env3 = MDPGridworldEnv(act_mode='stochastic',random_prob ='low')
	C =mydp(env3)
	C.monte_carlo_ES()
	C.show_policy()
