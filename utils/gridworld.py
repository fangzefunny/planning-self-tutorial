import time 
import numpy as np 
from IPython.display import clear_output

import matplotlib.pyplot as plt 
import seaborn as sns 

from .viz import viz

# -------------------------------------------------- #
#               Deterministic grid world             #
# -------------------------------------------------- #

def from_id_width(id, width): return (id % width, id // width)

def draw_tile(graph, id, style):
    r = " . "
    if 'number' in style and id in style['number']: r = " %-2d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1: r = " \u2190 "
        if x2 == x1 - 1: r = " \u2192 "
        if y2 == y1 + 1: r = " \u2191 "
        if y2 == y1 - 1: r = " \u2193 "
    if 'to_visit' in style and id in style['to_visit']: r = " ? "
    if 'path' in style and id in style['path']:   r = " @ "
    if 'n_best' in style and id in style['n_best']: 
        i = style['n_best'].index(id)
        r = f" {i+1} "
    if 'start' in style and id == style['start']: r = " S "
    if 'goal' in style and id == style['goal']:   r = " G "
    if id in graph.walls: r = "###"
    return r

def draw_grid(graph, **style):
    print("---" * graph.width)
    for y in range(graph.height):
        for x in range(graph.width):
            print("%s" % draw_tile(graph, (x, y), style), end="")
        print()
    print("---" * graph.width)

def reconstruct_path(came_from, start, goal):
    current, path = goal, []
    if goal not in came_from: return []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) 
    path.reverse() 
    return path

# ------------- grid world -------------- #
def map_to_lst(m):
    lst = [list(l) for l in [l.strip() for l in m.split('\n') if l.strip()]]
    return np.where((np.array(lst).reshape([-1]))=='#')[0].tolist()

L_walls = '''
.....#........
.....#........
.....#........
.....#........
.....#........
.....#######..
..............
''' 

world_configs = {
    
    'default': {
        'width': 30, 
        'height': 15, 
        'walls': [from_id_width(id, width=30) for id in [21,22,51,52,81,82,93,94,111,112,123,124,133,134,141,142,153,154,163,164,171,172,173,174,175,183,184,193,194,201,202,203,204,205,213,214,223,224,243,244,253,254,273,274,283,284,303,304,313,314,333,334,343,344,373,374,403,404,433,434]]
    }, 

    'box': {
        'width': 8, 
        'height': 8,
        'walls': [] 
    }, 

    'river': {
        'width': 8, 
        'height': 8,
        'walls': [from_id_width(id, width=8) for id in [19, 27, 32, 33, 34, 35]] 
    }, 

    'L':{
        'width': 14, 
        'height': 7,
        'walls': [from_id_width(id, width=14) for id in map_to_lst(L_walls)] 
    }

}

class grid_world:
    def __init__(self, config='default'):
        self.width  = world_configs[config]['width']
        self.height = world_configs[config]['height']
        self.walls  = world_configs[config]['walls']
    
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id):
        return id not in self.walls
    
    def neighbors(self, id):
        (x, y) = id
        neighbors = [(x+1, y), (x-1, y), (x, y-1), (x, y+1)] # E W N S
        # see "Ugly paths" section for an explanation:
        if (x + y) % 2 == 0: neighbors.reverse() # S N W E
        results = filter(self.in_bounds, neighbors)
        results = filter(self.passable, results)
        return results
    
    def weight_neighbors(self, id):
        (x, y) = id
        neighbors = [(x+1, y), (x-1, y), (x, y-1), (x, y+1)] # E W N S
        costs = [2, 2, 1, 10]
        # see "Ugly paths" section for an explanation:
        if (x + y) % 2 == 0: neighbors.reverse() # S N W E
        results = {}
        for i, n in enumerate(neighbors):
            if self.in_bounds(n) and self.passable(n):
                results[n] = costs[i]
        return results
    
# -------------------------------------------------- #
#               Stochstic:  frozen lake              #
# -------------------------------------------------- #

layout = [
    "S.......",
    "........",
    "...H..H.",
    ".....H..",
    "...H....",
    ".HH...H.",
    ".H..H.H.",
    "...H...G"
]


class frozen_lake:
    n_row = 8
    n_col = 8

    def __init__(self, layout=layout, eps=.2, seed=1234):

        # get occupancy 
        self.rng = np.random.RandomState(seed)
        self.layout = layout
        self.get_occupancy()
        self.eps = eps 
        # define MDP 
        self._init_S()
        self._init_A()
        self._init_P()
        self._init_R()
        

    def get_occupancy(self):
        # get occupancy, current state and goal 
        map_dict = {
            'H': .7,
            '.': 0,
            'S': 0, 
            'G': 0,
        }
        self.occupancy = np.array([list(map(lambda x: map_dict[x], row)) 
                                   for row in self.layout])
        self.goal = np.hstack(np.where(np.array([list(row) 
                        for row in self.layout])=='G'))
        self.curr_cell = np.hstack(np.where(np.array([list(row) 
                        for row in self.layout])=='S'))
        holes = np.array([list(row) for row in self.layout])=='H'
        self.hole_cells = [h for h in np.vstack(np.where(holes)).T]
        
    def cell2state(self, cell):
        return cell[0]*self.occupancy.shape[1] + cell[1]
    
    def state2cell(self, state):
        n = self.occupancy.shape[1]
        return np.array([state//n, state%n])
        
    # ------------------ Define MDP --------------- #

    def _init_S(self):
        '''Define the state space
        '''
        self.nS = frozen_lake.n_row*frozen_lake.n_col
        self.S  = list(range(self.nS))
        self.goal_state = self.cell2state(self.goal)
        self.state = self.cell2state(self.curr_cell)
        self.hole_states = [self.cell2state(h) for h in self.hole_cells]
        self.s_termination = self.hole_states+[self.goal_state]

    def _init_A(self,):
        '''Define the action space 
        '''
        # init 
        self.directs = [
            np.array([-1, 0]), # up
            np.array([ 1, 0]), # down
            np.array([ 0,-1]), # left
            np.array([ 0, 1]), # right
        ]
        self.nA = len(self.directs)
        self.A  = list((range(self.nA)))

    def _init_P(self):
        '''Define the transition function, P(s'|s,a)

            P(s'|s,a) is a probability distribution
        '''

        def p_s_next(s, a):
            p_next = np.zeros([self.nS])
            cell = self.state2cell(s)
            if s in self.s_termination:
                p_next[s] = 1 
            else:
                for j in self.A:
                    s_next = self.cell2state(
                        np.clip(cell + self.directs[j],
                        0, frozen_lake.n_row-1))
                    
                    # add probability 
                    if j == a:
                        p_next[s_next] += 1-self.eps
                    else:
                        p_next[s_next] += self.eps / (self.nA-1)
                
            return p_next
        
        self.p_s_next = p_s_next

    def _init_R(self):
        '''Define the reward function, R(s,a,s')

        return:
            r: reward
            done: if terminated 
        '''
        def R(s):
            if s == self.goal_state:
                return 1, True
            elif s in self.hole_states:
                return -1, True
            else:
                return 0, False
        self.r = R
        
    # ------------ visualize the environment ----------- #

    def reset(self):
        self.curr_cell = np.hstack(np.where(np.array([list(row) 
                        for row in self.layout])=='S'))
        self.state = self.cell2state(self.curr_cell)
        self.done = False
        self.act = None

        return self.state, None, self.done 

    def render(self, ax):
        '''Visualize the figure
        '''
        occupancy = np.array(self.occupancy)
        sns.heatmap(occupancy, cmap=viz.mixMap, ax=ax,
                    vmin=0, vmax=1, 
                    lw=.5, linecolor=[.9]*3, cbar=False)
        ax.axhline(y=0, color='k',lw=5)
        ax.axhline(y=occupancy.shape[0], color='k',lw=5)
        ax.axvline(x=0, color='k',lw=5)
        ax.axvline(x=occupancy.shape[1], color='k',lw=5)
        ax.text(self.goal[1]+.15, self.goal[0]+.75, 'G', color=viz.Red,
                    fontweight='bold', fontsize=10)
        ax.text(self.curr_cell[1]+.25, self.curr_cell[0]+.75, 'O', color=viz.Red,
                    fontweight='bold', fontsize=10)
        r, _ = self.r(self.state)
        ax.set_title(f'Reward: {r}, done: {self.done}')
        ax.set_axis_off()
        ax.set_box_aspect(1)

    def show_pi(self, ax, pi):
        self.reset()
        self.render(ax)
        for s in self.S:
            if s not in self.s_termination:
                cell = self.state2cell(s)
                a = pi[s].argmax()
                next_cell = self.directs[a]*.25
                ax.arrow(cell[1]+.5, cell[0]+.5, 
                        next_cell[1], next_cell[0],
                        width=.01, color='k')
        ax.set_title('Policy')

    def show_v(self, ax, V):
        v_mat = V.reshape([frozen_lake.n_row, frozen_lake.n_col])
        sns.heatmap(v_mat, cmap=viz.RedsMap, ax=ax,
                    lw=.5, linecolor=[.9]*3, cbar=False)
        ax.axhline(y=0, color='k',lw=5)
        ax.axhline(y=v_mat.shape[0], color='k',lw=5)
        ax.axvline(x=0, color='k',lw=5)
        ax.axvline(x=v_mat.shape[1], color='k',lw=5)
        for s in self.S:
            if s not in self.s_termination:
                    cell = self.state2cell(s)
                    v = V[s].round(2)
                    ax.text(cell[1]+.15, cell[0]+.65,
                            str(v), color='k',
                            fontweight='bold', fontsize=8)
        ax.set_title('Value')
        ax.set_axis_off()
        ax.set_box_aspect(1)
    
    def step(self, act):
        # get the next state 
        p_s_next = self.p_s_next(self.state, act)
        self.state = self.rng.choice(self.S, p=p_s_next)
        self.curr_cell = self.state2cell(self.state)
        rew, self.done = self.r(self.state)
        self.act = None 
        return self.state, rew, self.done


if __name__ == "__main__":

    env = grid_world('river')
    draw_grid(env)

