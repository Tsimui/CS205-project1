"""
CS205 Project1, solving 8-puzzle with three different searching algorithm:
uniform cost search, a* search with the misplaced heuristic, a* search with the manhattan distance heuristic
"""

import copy
import matplotlib.pyplot as plt
import numpy as np

# set size to 3 is to solve 8-puzzle, if wanna solve 16-puzzle or 25-puzzle, size can be set to 4 or 5
goal_state = []
size = 3
for i in range(1, size * size):
    goal_state.append(str(i))
goal_state.append('0')

# store the scale of search space
uc_nodes = []
misp_nodes = []
manh_nodes = []

# store the spending time of different algorithm
uc_times = []
misp_times = []
manh_times = []


# this function is used to check whether the current state is the goal state
def check_state(current, goal):
    current_string = ''.join(current)
    goal_string = ''.join(goal)
    if current_string == goal_string:
        return True
    else:
        return False


# this function is used to check whether the current state has appeared in the search space
def check_repeat_state(current, all_possible):
    current_string = ''.join(current)
    if current_string in all_possible:
        return True
    else:
        return False


# this function is used to make movement: put the deeper node into queue
def update_nodes(state, edge, all_possible):
    edge.append(state)
    state_str = ''.join(state)
    all_possible.append(state_str)


def matrix_print(state):
    start = 0
    for _ in range(size):
        print(state[start:start+size])
        start += size


# the uniform cost search. since each movement costs the same, the uniform cost search is the same as BFS
def uniform_cost_search(initial):
    edge_nodes = []
    all_possible_nodes = []
    heuristic = []
    if initial:
        edge_nodes.append(initial)
        initial_string = ''.join(initial)
        all_possible_nodes.append(initial_string)
        heuristic.append(0)
        while True:
            current_state = edge_nodes.pop(0)
            current_depth = heuristic.pop(0)
            if check_state(current_state, goal_state):
                print(f'node expanded: {len(all_possible_nodes)}')
                uc_nodes.append(len(all_possible_nodes))
                return True
            else:
                position = current_state.index('0')
                row = position // size
                col = position % size
                if row != 0:
                    up_state = copy.deepcopy(current_state)
                    up_state[position] = up_state[position - size]
                    up_state[position - size] = '0'
                    if not check_repeat_state(up_state, all_possible_nodes):
                        update_nodes(up_state, edge_nodes, all_possible_nodes)
                        heuristic.append(current_depth + 1)
                if row != size - 1:
                    down_state = copy.deepcopy(current_state)
                    down_state[position] = down_state[position + size]
                    down_state[position + size] = '0'
                    if not check_repeat_state(down_state, all_possible_nodes):
                        update_nodes(down_state, edge_nodes, all_possible_nodes)
                        heuristic.append(current_depth + 1)
                if col != 0:
                    left_state = copy.deepcopy(current_state)
                    left_state[position] = left_state[position - 1]
                    left_state[position - 1] = '0'
                    if not check_repeat_state(left_state, all_possible_nodes):
                        update_nodes(left_state, edge_nodes, all_possible_nodes)
                        heuristic.append(current_depth + 1)
                if col != size - 1:
                    right_state = copy.deepcopy(current_state)
                    right_state[position] = right_state[position + 1]
                    right_state[position + 1] = '0'
                    if not check_repeat_state(right_state, all_possible_nodes):
                        update_nodes(right_state, edge_nodes, all_possible_nodes)
                        heuristic.append(current_depth + 1)
    else:
        return False


# this function is used to count the number of misplaced titles
def check_misplaced(current, goal):
    num = 0
    for j in range(len(current)):
        if current[j] != goal[j] and current[j] != '0':
            num += 1
    return num


# the a* search with the misplaced heuristic search
def a_star_with_misplaced(initial):
    edge_nodes = []
    all_possible_nodes = []
    heuristic = []
    if initial:
        edge_nodes.append(initial)
        initial_string = ''.join(initial)
        all_possible_nodes.append(initial_string)
        misplaced_num = check_misplaced(initial, goal_state)
        heuristic.append([0, misplaced_num])
        while True:
            index = heuristic.index(min(heuristic, key=lambda x: sum(x)))
            current_state = edge_nodes.pop(index)
            current_depth = heuristic.pop(index)[0]
            if check_state(current_state, goal_state):
                print(f'node expanded: {len(all_possible_nodes)}')
                misp_nodes.append(len(all_possible_nodes))
                return True
            else:
                position = current_state.index('0')
                row = position // size
                col = position % size
                if row != 0:
                    up_state = copy.deepcopy(current_state)
                    up_state[position] = up_state[position - size]
                    up_state[position - size] = '0'
                    if not check_repeat_state(up_state, all_possible_nodes):
                        update_nodes(up_state, edge_nodes, all_possible_nodes)
                        misplaced_num = check_misplaced(up_state, goal_state)
                        heuristic.append([current_depth + 1, misplaced_num])
                if row != size - 1:
                    down_state = copy.deepcopy(current_state)
                    down_state[position] = down_state[position + size]
                    down_state[position + size] = '0'
                    if not check_repeat_state(down_state, all_possible_nodes):
                        update_nodes(down_state, edge_nodes, all_possible_nodes)
                        misplaced_num = check_misplaced(down_state, goal_state)
                        heuristic.append([current_depth + 1, misplaced_num])
                if col != 0:
                    left_state = copy.deepcopy(current_state)
                    left_state[position] = left_state[position - 1]
                    left_state[position - 1] = '0'
                    if not check_repeat_state(left_state, all_possible_nodes):
                        update_nodes(left_state, edge_nodes, all_possible_nodes)
                        misplaced_num = check_misplaced(left_state, goal_state)
                        heuristic.append([current_depth + 1, misplaced_num])
                if col != size - 1:
                    right_state = copy.deepcopy(current_state)
                    right_state[position] = right_state[position + 1]
                    right_state[position + 1] = '0'
                    if not check_repeat_state(right_state, all_possible_nodes):
                        update_nodes(right_state, edge_nodes, all_possible_nodes)
                        misplaced_num = check_misplaced(right_state, goal_state)
                        heuristic.append([current_depth + 1, misplaced_num])
    else:
        return False


# this function is used to count the manhattan distance
def check_manhattan(current, goal):
    num = 0
    for j in range(len(current)):
        if current[j] != goal[j] and current[j] != '0':
            num += abs(j // size - (int(current[j]) - 1) // size) + abs(j % size - (int(current[j]) - 1) % size)
    return num


# the a* search with the manhattan distance heuristic
def a_star_with_manhattan(initial):
    edge_nodes = []
    all_possible_nodes = []
    heuristic = []
    if initial:
        edge_nodes.append(initial)
        initial_string = ''.join(initial)
        all_possible_nodes.append(initial_string)
        misplaced_num = check_manhattan(initial, goal_state)
        heuristic.append([0, misplaced_num])
        while True:
            index = heuristic.index(min(heuristic, key=lambda x: sum(x)))
            current_state = edge_nodes.pop(index)
            # current_depth = heuristic.pop(index)[0]
            depth_with_dis = heuristic.pop(index)
            current_depth = depth_with_dis[0]
            dis = depth_with_dis[1]
            # this line is for the traceback
            print("The state to expand with a g(n)={} and h(n)={} is: ".format(current_depth, dis))
            matrix_print(current_state)
            if check_state(current_state, goal_state):
                print(f'node expanded: {len(all_possible_nodes)}')
                manh_nodes.append(len(all_possible_nodes))
                return True
            else:
                position = current_state.index('0')
                row = position // size
                col = position % size
                if row != 0:
                    up_state = copy.deepcopy(current_state)
                    up_state[position] = up_state[position - size]
                    up_state[position - size] = '0'
                    if not check_repeat_state(up_state, all_possible_nodes):
                        update_nodes(up_state, edge_nodes, all_possible_nodes)
                        misplaced_num = check_manhattan(up_state, goal_state)
                        heuristic.append([current_depth + 1, misplaced_num])
                if row != size - 1:
                    down_state = copy.deepcopy(current_state)
                    down_state[position] = down_state[position + size]
                    down_state[position + size] = '0'
                    if not check_repeat_state(down_state, all_possible_nodes):
                        update_nodes(down_state, edge_nodes, all_possible_nodes)
                        misplaced_num = check_manhattan(down_state, goal_state)
                        heuristic.append([current_depth + 1, misplaced_num])
                if col != 0:
                    left_state = copy.deepcopy(current_state)
                    left_state[position] = left_state[position - 1]
                    left_state[position - 1] = '0'
                    if not check_repeat_state(left_state, all_possible_nodes):
                        update_nodes(left_state, edge_nodes, all_possible_nodes)
                        misplaced_num = check_manhattan(left_state, goal_state)
                        heuristic.append([current_depth + 1, misplaced_num])
                if col != size - 1:
                    right_state = copy.deepcopy(current_state)
                    right_state[position] = right_state[position + 1]
                    right_state[position + 1] = '0'
                    if not check_repeat_state(right_state, all_possible_nodes):
                        update_nodes(right_state, edge_nodes, all_possible_nodes)
                        misplaced_num = check_manhattan(right_state, goal_state)
                        heuristic.append([current_depth + 1, misplaced_num])
    else:
        return False


def traceback(initial, difficulty):
    print("This is a traceback of an {} 8-puzzle with manhattan distance heuristic, "
          "the initial state is below with g(n)=0".format(difficulty))
    a_star_with_manhattan(initial)


"""
some test case about 8-puzzle. if wanna test the 16-puzzle or 25-puzzle, the test cases should be modified correctly
'0' represent the blank in the puzzle
"""
depth_range = [0, 4, 8, 12, 16, 20]
test_cases = [['1', '2', '3', '4', '5', '6', '7', '8', '0'],
              ['1', '2', '3', '5', '0', '6', '4', '7', '8'],
              ['1', '3', '6', '5', '0', '2', '4', '7', '8'],
              ['1', '3', '6', '5', '0', '7', '4', '8', '2'],
              ['1', '6', '7', '5', '0', '3', '4', '8', '2'],
              ['7', '1', '2', '4', '8', '5', '6', '3', '0']]

# a figure show the difference among these three search algorithm
for d in depth_range:
    initial_state = test_cases.pop(0)
    uniform_cost_search(initial_state)
    a_star_with_misplaced(initial_state)
    a_star_with_manhattan(initial_state)
plt.plot(depth_range, uc_nodes, label='uniform search')
plt.plot(depth_range, misp_nodes, label='A* search with misplaced')
plt.plot(depth_range, manh_nodes, label='A* search with manhattan')
plt.legend(loc='upper left')
plt.xlabel('Depth')
plt.ylabel('Nodes Expanded')
x_ticks = np.linspace(depth_range[0], depth_range[-1], len(depth_range))
plt.xticks(x_ticks)
y = np.array([1, 10, 100, 1000, 10000, 100000])
plt.yscale('log')
plt.yticks(y, labels=[str(label) for label in y])
plt.show()

# the traceback part
traceback(test_cases[1], 'easy')
traceback(test_cases[3], 'difficult')
