import queue
import math

class MCTSNode:
    """
    Node class for Monte Carlo Tree Search
    Represents a musical state in the search tree
    """
    
    def __init__(self, state=None, parent=None, action=None, featuresList=None, songlist=None):
        """
        Initialize a node in the MCTS tree
        
        Args:
            state: The musical sequence at this node
            parent: Parent node in the tree
            action: The note parameters that led to this state
            featuresList: List of musical features
            songlist: List of possible next songs/states
        """
        self.state = state  # The musical sequence
        self.parent = parent  # Parent node
        self.genre = featuresList
        self.action = action  # The note parameters that led to this state
        self.children = {}  # Successors of this node
        self.visits = 0  # Number of times this node has been visited
        self.score = 0.0  # Total score accumulated during backpropagation
        self.features = featuresList  # Features of the musical sequence
        
        # Set hash based on state and parent
        if parent is None:
            self.hash = ""
        else:
            self.hash = self.action.toString() + self.parent.hash
            
        # Store possible successors
        self._possible_songs = None
        if songlist is not None:
            self._possible_songs = songlist

    def get_possible_songs(self):
        """Get possible successor states"""
        if self._possible_songs is None:
            # Call external function to get possible songs
            self._possible_songs = self._get_successor_states()
        return self._possible_songs

    def _get_successor_states(self):
        """Get successor states from external function"""
        # This should be implemented to call your external function
        # that generates possible next musical states
        pass

    def add_child(self, successor):
        """Add a child node to this node"""
        if successor in self.children:
            raise ValueError('Child already exists')
            
        # Create new node with this node as parent
        child = MCTSNode(
            state=successor,
            parent=self,
            action=successor,  # The successor becomes the action that led to it
            featuresList=self.features
        )
        self.children[successor] = child
        return child

    def expand(self):
        """Expand this node by adding a new child"""
        for successor in self.get_possible_songs():
            try:
                return self.add_child(successor)
            except ValueError:
                continue
        return None

    def get_explore_term(self, parent, c=1):
        if self.parent is not None:
            return c * (2 * math.log(parent.count) / self.count) ** (1 / 2)
        else:
            return 0

    def get_exploit_term(self):
        return (0.8 * (self.genre_score) + 0.2 * (get_quality(self.midi, self.features))) / self.count

    def get_ucb(self, c=1, default=6):
        if self.count:
            exploit_term = self.get_exploit_term()
            explore_term = self.get_explore_term(self.parent,c)
            return exploit_term + explore_term # will eventually be normalized
        else:
            return default

    def print_subtree(self, max_nodes=None):
        if max_nodes is None:
            max_nodes = len(self.children) + 1
        print(f"\n\nPrinting the subtree starting from node {self.hash} up to a maximum of {max_nodes} nodes\n\n")
        q = queue.Queue()
        q.put(self)
        node_count = 0
        while not q.empty() and node_count < max_nodes:
            node_count += 1
            n = q.get()
            print(n)
            for key in n.children.keys():
                q.put(n.children[key])
