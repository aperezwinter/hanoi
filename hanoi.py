# import libraries
import time, resource, tracemalloc
import numpy as np
from copy import copy, deepcopy
from collections import deque
from itertools import combinations

# DEFINE CLASSES

class Stack(object):
    def __init__(self, id: int, items: list=[]) -> None:
        assert id >= 0, f"Stack's id must be a positive integer. Recieved: {id}."
        # Initialize attributes
        self._id = id
        self._items = deque(items)

    def __str__(self) -> str:
        # By defining the __str__ method in your class, 
        # you can specify how an object should be converted 
        # to a string when print() is called on it.
        s = f"Stack({self._id}): Bottom | "
        s += ' | '.join([str(item) for item in self._items])
        s += " | Top"
        return s
    
    def __repr__(self) -> str:
        # The __repr__ method specifies the "official" 
        # string representation of an object. It's typically 
        # used for debugging and development purposes.
        return f"Stack({list(self._items)})"
    
    def __eq__(self, other: object) -> bool:
        # Two `deque` objects are equal if all their items 
        # are equal (order is preserved).
        return self._items == other._items
    
    def empty(self) -> bool:
        # Return True is the stack is empty and False otherwise.
        return len(self._items) == 0

    def push(self, item) -> None:
        # Add an item to the top of the stack.
        self._items.append(item)

    def pop(self) -> object:
        # Remove and return the top item from the stack.
        if self.empty():
            raise IndexError("Pop from an empty stack")
        else:
            return self._items.pop()

    def peek(self) -> object:
        # Return the top item without removing it.
        if self.empty():
            raise IndexError("Peek from an empty stack")
        else:
            return self._items[-1]

    def size(self) -> int:
        # Return the number of items.
        return len(self._items)
    
    def common_item(self, other: object) -> bool:
        # Return True if both deque objects have an 
        # item in common and False otherwise.
        set_this = set(self._items)
        set_other = set(other._items)
        return len(set_this.intersection(set_other)) > 0
    
    def sorted(self, reverse: bool=False) -> bool:
        # Default order is descending.
        sorted_items = deque(sorted(self._items, reverse=reverse))
        return self._items == sorted_items
    
class State(object):
    def __init__(self, rods: list=[], max_disks: int=0, cost: float=0.0) -> None:
        # Check Hanoi state.
        if len(rods) > 0:
            # Bigger disks must be above smaller disks.
            error_order_disk, n = False, 0
            while (not error_order_disk) and (n < len(rods)):
                if rods[n].size() > 1:
                    error_order_disk = rods[n].sorted()
                n += 1
            if error_order_disk:
                raise ValueError("There is at least one bigger disk above a smaller one.")
            # The disks exist in a unique way. Therefore, the same disk cannot exist on more than one rod.
            error_repeat_disk, n = False, 0
            rods_idx = list(range(len(rods)))
            rod_pairs = list(combinations(rods_idx, 2))
            while (not error_repeat_disk) and (n < len(rod_pairs)):
                (i, j) = rod_pairs[n]
                error_repeat_disk = rods[i].common_item(rods[j])
                n += 1
            if error_repeat_disk:
                raise ValueError("There is at least one disk that exists in more than one rod.")
            # The total amount of disks must be equal to `max_disks`.
            n_disks = sum([rod.size() for rod in rods])
            if n_disks != max_disks:
                raise ValueError(f"The number of disks don't match. Max: {max_disks}. Recieved: {n_disks}.")
        
        # Initialize attributes
        self._rods = rods
        self._n_disks = sum([rod.size() for rod in rods])
        self._n_rods = len(rods)
        self._acummulated_cost = cost

    def __str__(self) -> str:
        # By defining the __str__ method in your class, 
        # you can specify how an object should be converted 
        # to a string when print() is called on it.
        s = f"State --> {self._n_rods} rods | {self._n_disks} disks "
        s += '[' + " ".join([str(rod.size()) for rod in self._rods]) + "]\n"
        for rod in self._rods:
            s += rod.__str__() + '\n'
        return s
    
    def __repr__(self) -> str:
        # The __repr__ method specifies the "official" 
        # string representation of an object. It's typically 
        # used for debugging and development purposes.
        s = ",".join([str(list(rod._items)) for rod in self._rods])
        return f"State({s})"
    
    def __eq__(self, other: object) -> bool:
        # Two `State` objects are equal if all their items 
        # are equal (order is preserved).
        return all([rod_i == rod_j for rod_i, rod_j in zip(self._rods, other._rods)])
    
    def __lt__(self, other: object) -> bool:
        # One `State` is lower than the other according to its cost value.
        return self._acummulated_cost < other._acummulated_cost
    
    def __gt__(self, other: object) -> bool:
        # One `State` is greater than the other according to its cost value.
        return self._acummulated_cost > other._acummulated_cost
    
    def __hash__(self) -> int:
        s = []
        for rod in self._rods:
            if rod.empty():
                s.append('0')
            else:
                s.append( ''.join(str(item) for item in rod._items) )
        return int( ''.join(s) )
    
    def check_move(self, _from:int, _to:int) -> bool:
        # Check movement condition from i-th rod to j-th rod.
        # Each rod is given by its index:
        # i-th rod --> `_from` index
        # j-th rod --> `_to` index
        if (_from < self._n_rods) and (_to < self._n_rods):
            if self._rods[_from].empty():
                return False
            elif (self._rods[_from].empty() and self._rods[_to].empty()):
                return False
            elif self._rods[_to].empty():
                return True
            else:
                if self._rods[_from].peek() < self._rods[_to].peek():
                    return True
        else: return False

    def move(self, _from:int, _to:int, check:bool=False) -> bool:
        # Move the disk from i-th rod to j-th rod.
        # Each rod is given by its index:
        # i-th rod --> `_from` index
        # j-th rod --> `_to` index
        if check:
            if (_from < self._n_rods) and (_to < self._n_rods):
                if self._rods[_from].peek() < self._rods[_to].peek():
                    self._rods[_to].push(self._rods[_from].pop())
                    return True
            return False
        else:
            self._rods[_to].push(self._rods[_from].pop())
            return True
    
    def create_sub_states(self) -> list:
        moves, sub_states = [], []
        for i in range(self._n_rods):
            for j in range(i+1, self._n_rods):
                if self.check_move(i, j):
                    moves.append((i, j))
                elif self.check_move(j, i):
                    moves.append((j, i))
                else: continue

        for (_from, _to) in moves:
            sub_states.append(deepcopy(self))
            sub_states[-1].move(_from, _to)
            sub_states[-1]._acummulated_cost += 1
        return sub_states
    
class TreeNode(object):
    def __init__(self, value, depth: int=0) -> None:
        self._value = value
        self._depth = depth # depth level in the tree
        self._children = [] # child nodes 

    def __str__(self) -> str:
        # By defining the __str__ method in your class, 
        # you can specify how an object should be converted 
        # to a string when print() is called on it.
        s = f"Node: Level {self._depth}\n"
        s += self._value.__repr__() + '\n'
        s += '-' * max(len(f"Node: Level {self._depth}\n"), \
                       len(self._value.__repr__() + '\n'))
        return s
    
    def __repr__(self) -> str:
        # The __repr__ method specifies the "official" 
        # string representation of an object. It's typically 
        # used for debugging and development purposes.
        return self.__str__()
    
    def __eq__(self, other: object) -> bool:
        return self._value == other._value
    
    def add_child(self, node) -> None:
        self._children.append(node)

    def remove_child(self, node) -> bool:
        try:
            self._children.remove(node)
        except ValueError:
            return False
        return True
    
    def num_children(self) -> None:
        return len(self._children)

class Tree(object):
    def __init__(self, value) -> None:
        self._root = TreeNode(value, 0)

    def __str__(self) -> str:
        # By defining the __str__ method in your class, 
        # you can specify how an object should be converted 
        # to a string when print() is called on it.
        depth = self.max_depth()
        n_nodes = self.count()
        s = f"Tree: Max Depth={depth} | #Nodes={n_nodes}"
        return s
    
    def __repr__(self) -> str:
        # The __repr__ method specifies the "official" 
        # string representation of an object. It's typically 
        # used for debugging and development purposes.
        return self.__str__()
    
    def set_root(self, value) -> None:
        self._root = TreeNode(value, 0)

    def reset(self) -> None:
        self._root._children = []

    def insert(self, parent_value, child_value) -> bool:
        # Insert a child node with value `child_value` 
        # to its parent node with value `parent_value`.
        parent_node = self.find(parent_value, self._root)
        if parent_node:
            child_node = TreeNode(child_value, parent_node._depth + 1)
            if child_node in parent_node._children:
                return False
            else:
                parent_node.add_child(child_node)
                return True
        else:
            print(f"Parent node with value {parent_value} not found.")

    def find_bfs(self, value, current_node=None, register: bool=False):
        current_node = self._root if current_node is None else current_node
        if register:
            if current_node is None:
                return None, 0
            visited = set()
            queue = deque([(current_node, current_node._depth)])  # Tuple: (node, level)
            visited.add(current_node._value)
            nodes_visited = 1  # Count the root node
            while queue:
                node, level = queue.popleft()
                if node._value == value:
                    return node, nodes_visited
                for child in node._children:
                    queue.append((child, level + 1))
                    visited.add(child._value)
                    nodes_visited += 1
            return None, nodes_visited
        else:
            if current_node is None:
                return None
            queue = deque([current_node])
            while queue:
                node = queue.popleft()
                if node._value == value:
                    return node
                for child in node._children:
                    queue.append(child)
            return None

    def find_dfs(self, value, current_node=None, register: bool=False):
        current_node = self._root if current_node is None else current_node
        if register:
            if current_node is None:
                return None, 0
            visited = set()
            return self.dfs_helper(current_node, value, visited)
        else:
            if current_node is None:
                return None
            if current_node._value == value:
                return current_node
            for child in current_node._children:
                found_node = self.find_dfs(value, child)
                if found_node:
                    return found_node
            return None
    
    def dfs_helper(self, node, value, visited):
        visited.add(node._value)
        if node._value == value:
            return node, len(visited)
        for child in node._children:
            if child._value not in visited:
                found_node, nodes_visited = self.dfs_helper(child, value, visited)
                if found_node:
                    return found_node, nodes_visited
        return None, len(visited)

    
    def find(self, value, current_node=None, method:str="BFS"):
        # Start with the current node, follow the 
        # tree structure and find the node with the 
        # given value.
        current_node = self._root if current_node is None else current_node
        if method == "BFS":
            if current_node is None:
                return None
            queue = deque([current_node])
            while queue:
                node = queue.popleft()
                if node._value == value:
                    return node
                for child in node._children:
                    queue.append(child)
            return None
        elif method == "DFS":
            if current_node is None:
                return None
            if current_node._value == value:
                return current_node
            for child in current_node._children:
                found_node = self.find(value, child, method)
                if found_node:
                    return found_node
            return None
        else: 
            return None
    
    def remove(self, parent_value, child_value) -> bool:
        # Remove a child node with value `child_value` 
        # to its parent node with value `parent_value`.
        parent_node = self.find(parent_value, self._root)
        if parent_node:
            child_node = TreeNode(child_value, parent_node._depth + 1)
            return parent_node.remove_child(child_node)
        else:
            print(f"Parent node with value {parent_value} not found.")
            return False
        
    def expand(self, current_node=None) -> None:
        # Expand the tree and create one level 
        # of children (create sub states).
        current_node = self._root if current_node is None else current_node
        if len(current_node._children) == 0:
            states = current_node._value.create_sub_states()
            for state in states:
                if self.find(state, self._root) is None:
                    self.insert(current_node._value, state)
                else: continue
        else:
            for child in current_node._children:
                self.expand(child)
        return None
        
    def count(self, current_node=None) -> int:
        # Count the number of nodes strating 
        # from node given by `current_node`.
        current_node = self._root if current_node is None else current_node
        if current_node is None:
            return 0
        count = 1  # Count the current node
        for child in current_node._children:
            count += self.count(child)
        return count
    
    def max_depth(self, current_node=None) -> int:
        # Get the maximun depth in the Tree from 
        # the node given by `current_node`.
        current_node = self._root if current_node is None else current_node
        depth = 0
        if len(current_node._children) > 0:
            for child in current_node._children:
                depth = max(depth, self.max_depth(child))
            return depth + 1
        else:
            return depth
        
    def print(self, current_node=None, depth: int=0) -> None:
        current_node = self._root if current_node is None else current_node
        if current_node is None:
            return None
        print(f"L{depth} " + "  " * depth + str(hash(current_node._value)))
        for child in current_node._children:
            self.print(child, depth + 1)

class Hanoi(object):
    def __init__(self, init_state:list=[[3,2,1],[],[]]) -> None:
        self.n_rods = len(init_state)
        self.n_disks =  sum([len(state) for state in init_state])
        self.init_state = State([Stack(i, rod) for i, rod in enumerate(init_state)], self.n_disks)
        self.target_state = None
        self.solution = None
        self.steps = None
        self.tree = Tree(self.init_state)
        

    def solve_bfs(self, target:list=[[],[],[3,2,1]]):
        target_state = State([Stack(i, rod) for i, rod in enumerate(target)], self.n_disks)
        flag = False
        while not flag:
            self.tree.expand()
            solve_node, count_path = self.tree.find_bfs(target_state, register=True)
            flag = True if solve_node is not None else False
        self.solution = solve_node._value
        self.steps = count_path
    
    def solve_dfs(self, target:list=[[],[],[3,2,1]], depth: int=5):
        target_state = State([Stack(i, rod) for i, rod in enumerate(target)], self.n_disks)
        for _ in range(depth):
            self.tree.expand()
        solve_node, count_path = self.tree.find_dfs(target_state, register="True")
        self.solution = solve_node._value
        self.steps = count_path
        return solve_node, count_path
    
    def solve(self, target:list=[[],[],[3,2,1]], method:str="BFS", depth:int=2):
        target_state = State([Stack(i, rod) for i, rod in enumerate(target)], self.n_disks)

        if method == "BFS":
            flag = False
            while not flag:
                self.tree.expand()
                solve_node = self.tree.find(target_state, method=method)
                flag = True if solve_node is not None else False
            # self.tree.reset() # clean all children except the root
            return solve_node
        elif method == "DFS":
            for _ in range(depth):
                self.tree.expand()
            solve_node = self.tree.find(target_state, method="DFS")
            # self.tree.reset() # clean all children except the root
            return solve_node
        else:
            return None
        
# DEFINE SOME FUNCTIONS
def hanoi_stats_bfs(n_disks: int=4, n_rods: int=3, samples: int=10) -> list:
    enlapsed_time, memory = [], []
    init_state = [[] for _ in range(n_rods)]
    target_state = [[] for _ in range(n_rods)]
    
    init_state[0] = list(np.flip(np.arange(1, n_disks+1)))
    target_state[-1] = list(np.flip(np.arange(1, n_disks+1)))
    my_hanoi = Hanoi(init_state)
    
    for _ in range(samples):
        start_time = time.time()
        tracemalloc.start()
        my_hanoi.solve_bfs(target_state) # solve hanoi problem
        end_time = time.time()
        traced_memory = tracemalloc.get_traced_memory()
        enlapsed_time.append((end_time - start_time)*1000)
        memory.append((traced_memory[1]-traced_memory[0]) / (1024)) # convert to kilobytes
        my_hanoi.tree.reset()
    
    mu_time = np.mean(enlapsed_time)
    std_time = np.std(enlapsed_time)
    mu_memory = np.mean(memory)
    std_memory = np.std(memory)

    return (mu_time, std_time), (mu_memory, std_memory)

def hanoi_stats_dfs(n_disks: int=4, n_rods: int=3, samples: int=10) -> list:
    enlapsed_time, memory = [], []
    init_state = [[] for _ in range(n_rods)]
    target_state = [[] for _ in range(n_rods)]
    
    init_state[0] = list(np.flip(np.arange(1, n_disks+1)))
    target_state[-1] = list(np.flip(np.arange(1, n_disks+1)))
    my_hanoi = Hanoi(init_state)

    depth = 2**n_disks - 1 # define tree depth
    
    for _ in range(samples):
        start_time = time.time()
        tracemalloc.start()
        my_hanoi.solve_dfs(target_state, depth) # solve hanoi problem
        end_time = time.time()
        traced_memory = tracemalloc.get_traced_memory()
        enlapsed_time.append((end_time - start_time)*1000)
        memory.append((traced_memory[1]-traced_memory[0]) / (1024)) # convert to kilobytes
        my_hanoi.tree.reset()
    
    mu_time = np.mean(enlapsed_time)
    std_time = np.std(enlapsed_time)
    mu_memory = np.mean(memory)
    std_memory = np.std(memory)

    return (mu_time, std_time), (mu_memory, std_memory)