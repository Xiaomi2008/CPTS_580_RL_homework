import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3

MAP = [
    "+----+",
    "|   G|",
    "| | F|",
    "|S   |",
    "+----+"
]

class MDPGridworldEnv(discrete.DiscreteEnv):
    """
    IMPORTANT: This code is based from openAI gym's FrozenLake code.
    This is a 3x4 grid world based from problem for an AI-Class. (https://goo.gl/GqkyzT)
    The surface is described using a grid like the following

    S : starting point, non-terminal state (reward: -3)
      : non-terminal states (reward: -3)
    F : fire, burn to death (reward: -100)
    G : goal, an alternate universe where Trump is not the president (reward: +100)

    The episode ends when you reach the goal or burn in hell.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[3,4],act_mode='deterministic',random_prob ='high'):
        self.shape = shape
        self.desc = np.asarray(MAP,dtype='c')

        self.nrow = nR = 3
        self.ncol = nC = 4
        assert(act_mode in ['deterministic','stochastic'])
        
        #  set action -- prob pairs -----------------------------------
        front_prob = {'high':0.5,'low':0.95}
        left_prob  = {'high':0.2,'low':0.025}
        right_prob = {'high':0.2,'low':0.025}
        back_prob  = {'high':0.1,'low':0.0}

        action_probs ={0:front_prob,1:left_prob,2:right_prob,3:back_prob}
        #----------------------------------------------------------------
       

        nA = 4
        nS = nR * nC

        isd = np.zeros(nS)


        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*self.ncol + col
        def inc(row, col, a):
            t_row, t_col = row, col
            if a==WEST:
                col = max(col-1,0)
            elif a==SOUTH:
                row = min(row+1,self.nrow-1)
            elif a==EAST:
                col = min(col+1,self.ncol-1)
            elif a==NORTH:
                row = max(row-1,0)
            if self.desc[row+1,col+1] == b'|':
                row, col = t_row, t_col
            return (row, col)
        def get_next_status(row, col, a):
            rew = -3
            newrow, newcol = inc(row, col, a)
            newstate = to_s(newrow, newcol)
            newletter = self.desc[newrow+1, newcol+1]
            done = bytes(newletter) in b'GF'
            if bytes(newletter) in b'G':
                rew = 100
            elif bytes(newletter) in b'F':
                rew = -100
            return newstate ,rew, done
        def get_left_right_back_action(a):
            # 0 = North , left = West(2), Right = East(3) , opposite =  1
            # 1 = South,  left = East(3), Right = West(2) , opposite =  0
            # 2 = West,   Left = South(1),Right = North(0), opposite =  3
            # 3 = East,   Left = North(0),Right = South(1), opposite = 2 
            
            #    N 
            #    |
            # W-----E
            #    |
            #    S

            a_list       =  [0,3,1,2] # 
            front_idx    =  a_list.index(a)
            left_idx     =  (front_idx -1) % len(a_list)
            right_idx    =  (front_idx +1) % len(a_list)
            back_idx     =  (front_idx +2) % len(a_list)
            
            front_action    = a_list[front_idx]
            left_action     = a_list[left_idx]
            right_action    = a_list[right_idx]
            back_action     = a_list[back_idx]
            return [front_action,left_action,right_action,back_action]

        isd[to_s(2, 0)] = 1
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = self.desc[row+1, col+1]
                    if letter in b'GF':
                        li.append((1.0, s, 0, True))
                    elif letter == b'|':
                        li.append((0.0, s, 0, False))
                    else:
                        
                        # newrow, newcol = inc(row, col, a)
                        # newstate = to_s(newrow, newcol)
                        # newletter = self.desc[newrow+1, newcol+1]
                        # done = bytes(newletter) in b'GF'
                        # if bytes(newletter) in b'G':
                        #     rew = 100
                        # elif bytes(newletter) in b'F':
                        #     rew = -100
                        if act_mode=='stochastic':
                            assert(random_prob in ['low','high'])
                            act_list   = get_left_right_back_action(a)
                        else:
                            act_list={}
                            act_list[0] =a


                        for act_idx in range(len(act_list)):
                            action = act_list[act_idx]
                            if len(act_list)==1:
                                prob = 1.0
                            else:
                                prob=action_probs[act_idx][random_prob]
                                #print prob
                                #[random_prob]
                                # if act_idx == 0:
                                #     prob = front_prob[random_prob]
                                # elif act_idx == 1 or act_idx ==2:
                                #     prob = left_prob[random_prob]
                                # elif act_idx ==3:
                                #     prob = back_prob[random_prob]
                            #newstate,rew,done = self._get_next_status(row,col,action)
                            newstate,rew,done = get_next_status(row,col,action)
                            li.append((prob, newstate, rew, done))


        isd /= isd.sum()
        super(MDPGridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = (self.s // self.ncol)+1, (self.s % self.ncol)+1
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["North","South","West","East"][self.lastaction]))
        else:
            outfile.write("\n")

        return outfile
    
