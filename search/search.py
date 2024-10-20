# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in search_agents.py).
"""
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in obj-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()


def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# def addSuccessors(problem, addCost=True):

class SearchNode:
    def __init__(self, parent, node_info):
        """
            parent: parent SearchNode.

            node_info: tuple with three elements => (coord, action, cost)

            coord: (x,y) coordinates of the node position

            action: Direction of movement required to reach node from
            parent node. Possible values are defined by class Directions from
            game.py

            cost: cost of reaching this node from the starting node.
        """

        self.__state = node_info[0]
        self.action = node_info[1]
        self.cost = node_info[2] if parent is None else node_info[2] + parent.cost
        self.parent = parent

    # The coordinates of a node cannot be modified, se we just define a getter.
    # This allows the class to be hashable.
    @property
    def state(self):
        return self.__state

    def get_path(self):
        path = []
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.action)
            current_node = current_node.parent
        path.reverse()
        return path
    
    # Consider 2 nodes to be equal if their coordinates are equal (regardless of everything else)
    # def __eq__(self, __o: obj) -> bool:
    #     if (type(__o) is SearchNode):
    #         return self.__state == __o.__state
    #     return False

    # # def __hash__(sjelf) -> int:
    # #     return hash(self.__state)

def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    This search algorithm explores a path as deep as possible before backtracking.
    It uses a stack (LIFO) to track the nodes to visit next and ensures it does
    not revisit nodes already visited.
    """

    # Pila para los nodos que exploraremos. Cada elemento es una tupla (estado, acciones hasta ese estado)
    stack = util.Stack()
    
    # Lista para mantener los nodos ya visitados
    visited = []

    # El estado inicial del problema
    start_state = problem.get_start_state()
    
    # Añadimos el estado inicial a la pila. El segundo elemento es una lista vacía que irá acumulando las acciones.
    stack.push((start_state, []))

    # Mientras la pila no esté vacía
    while not stack.is_empty():
        # Tomamos el nodo más profundo (LIFO)
        current_state, actions = stack.pop()

        # Si hemos llegado al estado objetivo, devolvemos la secuencia de acciones
        if problem.is_goal_state(current_state):
            return actions

        # Si no hemos visitado el estado actual
        if current_state not in visited:
            # Lo marcamos como visitado
            visited.append(current_state)

            # Obtenemos los sucesores del estado actual
            for n in problem.get_successors(current_state):
                # Añadimos el sucesor a la pila junto con la secuencia de acciones que nos ha llevado hasta aquí
                new_actions = actions + [n[1]]
                stack.push((n[0], new_actions))

    # Si la pila se vacía y no hemos encontrado una solución, devolvemos una lista vacía
    return []

        
    util.raise_not_defined()

    
    
def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    
    # Cola para mantener los nodos que vamos a explorar (FIFO).
    queue = util.Queue()
    
    # Conjunto para mantener los nodos ya visitados.
    visited = set()

    # Estado inicial del problema.
    start_state = problem.get_start_state()

    # Añadimos el estado inicial a la cola. El segundo valor es la lista de acciones hasta ese estado.
    queue.push((start_state, []))

    # Mientras la cola no esté vacía, seguimos explorando
    while not queue.is_empty():
        # Sacamos el nodo más superficial (FIFO)
        current_state, actions = queue.pop()

        # Si el estado actual es el estado objetivo, devolvemos la secuencia de acciones.
        if problem.is_goal_state(current_state):
            return actions

        # Si no hemos visitado el estado actual
        if current_state not in visited:
            # Marcamos el estado como visitado
            visited.add(current_state)

            # Obtenemos los sucesores del estado actual
            for successor, action, step_cost in problem.get_successors(current_state):
                # Si el sucesor no ha sido visitado, lo agregamos a la cola
                if successor not in visited:
                    new_actions = actions + [action]
                    queue.push((successor, new_actions))

    # Si la cola se vacía y no encontramos una solución, devolvemos una lista vacía.
    return []

    util.raise_not_defined()

def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue  # Import the priority queue for UCS

    # Create a priority queue to store nodes to explore
    pq = PriorityQueue()
    # Set of visited nodes
    visited = set()

    # Get the starting state and push it onto the priority queue
    start_state = problem.get_start_state()
    pq.push((start_state, []), 0)  # (state, path), with initial cost 0

    while not pq.is_empty():
        # Pop the state with the lowest cost
        current_state, actions = pq.pop()

        # Check if this state has been visited
        if current_state in visited:
            continue
        
        # Mark the current state as visited
        visited.add(current_state)

        # Check if we reached the goal
        if problem.is_goal_state(current_state):
            return actions  # Return the path to the goal

        # Get the successors of the current state
        for successor, action, step_cost in problem.get_successors(current_state):
            if successor not in visited:
                # Calculate the new cost to reach this successor
                new_cost = problem.get_cost_of_actions(actions + [action])
                # Push the successor onto the priority queue with its cost
                pq.push((successor, actions + [action]), new_cost)

    return []  # Return an empty list if no path is found

    util.raise_not_defined()

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
