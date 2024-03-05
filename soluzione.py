import cv2
import time
import keras
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display

import sys
sys.path.insert(2, '/content/aima-python/')

from search import *
from collections import *

def show_solution(node):
  if node is None:
      print("no solution")
  else:
      print("solution: ", node.solution())


class SmartVacuum(Problem):

  def __init__(self, initial, width, height):
    print(initial)

    self.w = width
    self.h = height
    self.initial = initial

    self.Robot_index = len(self.initial)-1

    #print(self.initial)
    #print(self.Robot_index)

#posizione fine

    for x in range(len(initial)):
      if initial[x] == 5:
        self.F_index = x

#definiamo lo stato goal

    pippo = []
    for x in self.initial:

      if x in [2, 3]:
        pippo.append(1)
      else:
        pippo.append(x)

    pippo[len(pippo)-1] = self.F_index

    super().__init__(self.initial, goal=pippo)

  def goal_test(self, state):
    return state == self.goal

  def actions(self, state):

    cursor = state[self.Robot_index]

    possible_actions = ['right', 'up', 'left', 'down', 'clear', 'Sclear']

    # se sono in celle sporche le pulisco per forzaa
    if state[cursor] == 2:
      possible_actions=['clear']
      return possible_actions

    if state[cursor] == 3:
      possible_actions=['Sclear']
      return possible_actions

    if cursor % self.w == self.w - 1 or state[cursor + 1] == 4:
      possible_actions.remove('right')

    if cursor < self.w or state[cursor - self.w] == 4:
      possible_actions.remove('up')

    if cursor % self.w == 0 or state[cursor - 1] == 4:
      possible_actions.remove('left')

    if cursor >= self.w*(self.h - 1) or state[cursor + self.w] == 4:
      possible_actions.remove('down')

    #se è pulita non posso pulirla
    if state[cursor] in [0,1,5]:
      possible_actions.remove('clear')
      possible_actions.remove('Sclear')

    return possible_actions

  def result(self, state, action):

    #index of robot
    cursor = state[self.Robot_index]
    print(action)
    print(state)
    print(cursor)

    #transform tuple in list for coloring or switching position
    new_state = state[:self.Robot_index]


    if(action == 'clear' or action == 'Sclear'):

      new_state[cursor] = 1

    else:

      #dictionary to move and index
      moving = { 'right' : 1, 'up' : -self.w, 'left' : -1, 'down' : self.w }

      #index of T moved
      cursor += moving[action]

    out_state = new_state + [cursor]
    return out_state


    def goal_test(self, state):
      if (state == self.goal):
        return True
      else:
        return False

  def path_cost(self, c, state1, action, state2):
    cleaning = {'clear': 2, 'Sclear': 3}
    #costo unitario, sola distanza
    if action in cleaning:
      cleaning_cost = cleaning[action]
      return c + cleaning_cost
    else:
      return c + 1


W, B, BK, WALL, G, R, R_T, G_T, W_T, B_T, BK_T= np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8), np.zeros((3,3,3), np.uint8)
W[::], B[::], BK[::], WALL[::], G[::], R[::] = (255,255,255),(88,57,38),(0,0,0),(128,0,128),(34,139,34),(255,0,0)
W_T[::], B_T[::], BK_T[::], G_T[::], R_T[::] = (255,255,255),(88,57,38),(0,0,0),(34,139,34),(255,0,0)
W_T[1,1] = B_T[1,1] = BK_T[1,1] = G_T[1,1] = R_T[1,1] = (255, 0, 0)

color = [G, W, B, BK, WALL, R, G_T, W_T, B_T, BK_T, R_T]

def visual(path_solution, move_solution, w, h):

  #path_solution is containo node, we extract the node.state
  state_solution = []

  for i in path_solution:
    state_solution.append(i.state.copy())  #copy only value, not the reference

  for state in range(len(state_solution)):

    #find index T
    T_index = state_solution[state][-1]

    #make index t colored us testina
    state_solution[state][T_index] = state_solution[state][T_index] + 6

    #remove index t in state
    state_solution[state] = state_solution[state][0:-1]

  #widget PART STARTING

  a = widgets.IntSlider(min=0,max=len(state_solution)-1,step=1,value=0,description='Step: ', continuous_update=True)
  ui = widgets.HBox([a])

  def f(a):

    for i in range(len(state_solution[a])):
      plt.subplot(h,w, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.imshow(color[state_solution[a][i]])

    if(a==0):
      plt.suptitle(f"Initial stete, Next Move: {move_solution[a]}")
    elif(a <= len(move_solution)-1):
      plt.suptitle(f"Applied move: {move_solution[a-1]} | Next Move: {move_solution[a]}")
    else:
      plt.suptitle(f"Final stete, Applied move: {move_solution[a-1]}")

    plt.show()

  out = widgets.interactive_output(f, {'a': a})
  display(ui, out)
def breadth_search_graph(problem):
  node = Node(problem.initial)

  if problem.goal_test(node.state):
    return node

  frontier = deque([node])
  explored = set()

  while frontier:
    node = frontier.popleft()
    explored.add(tuple(node.state))
    for child in node.expand(problem):
      if tuple(child.state) not in explored and child not in frontier:
        if problem.goal_test(child.state):
          return child
        frontier.append(child)
  return None

def best_first_search_graph(problem, f, no_memoize = False):
  init = Node(problem.initial)

  if problem.goal_test(init.state):
    return init

  f = memoize(f, 'f')
  frontier = PriorityQueue('min', f)
  frontier.append(init)

  explored = set()

  while frontier:
    node = frontier.pop()
    if problem.goal_test(node.state):
        return node

    explored.add(tuple(node.state))

    for child in node.expand(problem):
      if tuple(child.state) not in explored and child not in frontier:
        frontier.append(child)
      elif child in frontier:
        incumbent = frontier.get_item(child)
        if f(incumbent) > f(child):
          del frontier[incumbent]
          frontier.append(child)
    print(f"heuristic: {f(node)}")
  return None

def a_star_search(problem, h = None):
  h = memoize(h or problem.h, 'h')
  return best_first_search_graph(problem, lambda n : h(n), no_memoize = False)

#conto le celle diverse dallo stato attuale allo stato goal
def h1(problem, node):
  return sum( s != g for (s,g) in zip(node.state, problem.goal) )

