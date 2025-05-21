import numpy as np
import random
from MCTSNode import MCTSNode

class MCTS:
    """
    Monte Carlo Tree Search class used for the Genre Search
    """

    def __init__(self, exploration_weight=1.0):
        self.exploration_weight = exploration_weight

    def run_search(self, root_node, genre_handler, num_simulations):
        for _ in range(num_simulations):
            node = self.select_next_node(root_node, genre_handler)
            score = genre_handler.try_genre(node)
            self.backpropagate(node, score)
        return genre_handler.pick_best_genre(root_node)

    def select_next_node(self, node, genre_handler):
        if len(node.children) < len(genre_handler.genre_list):
            for genre in genre_handler.genre_list:
                if genre not in node.children:
                    new_features = genre_handler.modify_features(node.state, genre)
                    new_node = MCTSNode(state=new_features, parent=node)
                    node.add_child(genre, new_node)
                    return new_node
        return self.select_with_ucb(node)

    def select_with_ucb(self, node):
        log_visits = np.log(node.visits) if node.visits > 0 else 0
        best_score = float('-inf')
        best_child = None

        for child in node.children.values():
            if child.visits == 0:
                score = float('inf')
            else:
                average = child.score / child.visits
                explore_bonus = self.exploration_weight * np.sqrt(log_visits / child.visits)
                score = average + explore_bonus

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def backpropagate(self, node, score):
        while node is not None:
            node.visits += 1
            node.score += score
            node = node.parent
