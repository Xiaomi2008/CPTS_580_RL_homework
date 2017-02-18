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
		self.Vstate =np.zeros(self.space_n)
		self.Qstate =np.zeros((self.space_n,self.action_n))
		#self.policy_action =np.zeros(self.space_n)*-1
		act =[0,1,2,3]
		self.policy = [act for j in range(self.space_n)]
		#print self.policy_action
		#dp_score_new= [[0 for i in range(4)] for i in range(3)]

	def value_iteration(self,gama = 0.6,theta =0.1):
		self.reset_val()
		while True:
			delta =0
			#print "==========="
			for state in range(self.space_n):
				max_v = 0
				v = self.Vstate[state]
				for action in range(self.action_n):
					T = self.env.P[state][action]
					V_a =0
					# seems that there is only one return list
					# containing a set of values for only one next state
					for tp,next_state,reward,terminal in T:
						V_a+=tp*(reward+gama*self.Vstate[next_state])
					
					max_v=V_a if max_v < V_a else max_v
				
				self.Vstate[state] = max_v
				#print "state [{}] = {}".format(state,max_v)
				delta = max(delta,abs(v-self.Vstate[state]))
				#time.sleep(.1)
				#print delta
			if delta <theta:
				break
	def policy_iteration(self,gama =0.6, theta =0.1):
		self.reset_val()
		# Policy Evaluation
		cur_Vstate   =np.copy(self.Vstate) #  need deep copy , instead of reference
		while True:
			# policy Evaluation
			while True:
				delta =0
				temp_Vstate =np.copy(cur_Vstate)
				
				for state in range(self.space_n):
					V_a =0
					actions =self.get_policy(state)
					v = cur_Vstate[state]   # use old state value

					# uncomment the following to use single random sigle action
					#action =actions[0]
					#if len(actions)>1:
					#	action=random.randint(0,len(actions)-1)
					#actions =[]
					#actions.append(action)
					for action in actions:
						T = self.env.P[state][action]
						for tp,next_state,reward,terminal in T:
							V_a+=tp*(reward+gama*cur_Vstate[next_state])*(1.0/len(actions))
					temp_Vstate[state]=V_a  # new state value
					delta = max(delta,abs(v-temp_Vstate[state]))
				cur_Vstate = np.copy(temp_Vstate)
				#print "========="
				if delta < theta:
					break
			# policy Improvement
			self.Vstate = cur_Vstate
			policy_stable  = True
			for state in range(self.space_n):
				b = self.get_policy(state)
				self.set_policy(state,self.get_max_V_actions(state)) #self.find_action_with_max_v(state)
				if b!= self.get_policy(state):
					policy_stable = False


			if policy_stable:
				break
			#else:
			#	print "policy is  not stable"

	def get_policy(self,state):
		return self.policy[state]

	def set_policy(self,state,actions):
		self.policy[state]=actions

	def get_max_V_actions(self,state,gama= 0.6):
		action_list = []
		action_values =np.zeros(self.action_n)
		for action in range(self.action_n):
				T = self.env.P[state][action]
				for tp,next_state,reward,terminal in T:
					action_values[action]+=tp*(reward+gama*self.Vstate[next_state])
		for action in range(self.action_n):
			mx =np.max(action_values)
			if  mx==action_values[action]:
				action_list.append(action)
		return action_list


	def update_policy(self):
		for state in range(self.space_n):
			self.set_policy(state,self.get_max_V_actions(state))


def show_result(mdp_obj):
	print "========================"
	print mdp_obj.Vstate[0:4]
	print mdp_obj.Vstate[4:8]
	print mdp_obj.Vstate[8:12]
	print "  "


if __name__ == "__main__":
	print "--------------Stochastic action state ------------ "
	env = MDPGridworldEnv(act_mode='stochastic',random_prob ='low')
	A =mydp(env)
	A.value_iteration()
	print "  "
	print "Value Iteration Results:"
	show_result(A)
	
	A.policy_iteration()
	print "Policy Iteration Results:"
	show_result(A)

	print  "--------------Deterministic action state ------------ "
	env2 = MDPGridworldEnv(act_mode='deterministic')
	B =mydp(env2)

	B.value_iteration()
	print "  "
	print "Value Iteration Results:"
	show_result(B)
	
	B.policy_iteration()
	print "Policy Iteration Results:"
	show_result(B)




