class MCTSNode:

    def __init__(self, state=None, parent=None):
        # state is the audio features
        self.state = state

        # reference to the parent node for backpropogation
        self.parent = parent

        # child nodes which map genres to their nodes
        self.children = {}

        # stats for this node

        # how many times was this node visited
        self.visits = 0
        # total score of this node
        self.score = 0.0

    def add_child(self, genre, child_node):

        self.children[genre] = child_node
