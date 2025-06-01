import numpy as np
from tqdm.notebook import tqdm
import math
import queue
from MCTSNode import MCTSNode
import random

def tree_policy(node):
    """Tree policy for MCTS - implements the selection and expansion phases"""
    if node.get_score() is not None:
        return node
    elif len(node.children) < len(node.get_possible_songs()):
        return node.expand()
    else:
        return tree_policy(best_child(node))

def best_child(node, c=1):
    """Select the best child node using UCB formula"""
    score = float('-inf')
    best_node = None
    for child in node.children.values():
        if score < child.get_ucb(c):
            score = child.get_ucb(c)
            best_node = child
    return best_node

def backup(node, score):
    """Backpropagate the score up the tree using iteration instead of recursion"""
    cur = node
    while cur is not None:
        cur.visits += 1
        cur.score += score
        cur = cur.parent

def default_policy(node, print_rollout_result=False):
    """Simulate from current node to a terminal state"""
    score = node.get_score()
    if score is not None:
        if print_rollout_result:
            print(node.hash)
            print(score)
        return score
    
    # Get possible successors
    successors = node.get_possible_songs()
    successor_random = random.choice(successors)
    temp_node = MCTSNode(successor_random, None)
    return default_policy(temp_node, print_rollout_result)

def mcts_search(num_iterations=50, print_added_nodes=False, print_rollout_result=False, print_final_tree=False, nodes_to_print=None):
    """Run MCTS search for the specified number of iterations"""
    start_node = MCTSNode()
    
    for _ in tqdm(range(num_iterations)):
        # Selection and expansion
        v = tree_policy(start_node)
        if print_added_nodes:
            print(v)
            print("Adding new node {} with parent {}".format(v.hash, v.parent.hash if v.parent else "None"))
            
        # Simulation
        value = default_policy(v, print_rollout_result)
        
        # Backpropagation
        backup(v, value)
    
    # Get best action
    best_node = best_child(start_node, 0)
    if best_node is None:
        return None
        
    action = best_node.action
    if print_final_tree:
        start_node.print_subtree(nodes_to_print)
    print("Action is :\n {}".format(action))
    return action

# Example usage
if __name__ == "__main__":
    mcts_search(10, print_added_nodes=True, print_final_tree=True, nodes_to_print=float("inf"))
    mcts_search(20, print_added_nodes=True, print_final_tree=True, nodes_to_print=float("inf"))
    mcts_search(20, print_final_tree=True, nodes_to_print=float("inf"))
    mcts_search(20, print_final_tree=True, print_rollout_result=True)
