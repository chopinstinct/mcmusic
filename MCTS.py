
from MCTSnode import MCTSnode
import random

class MCTS:

    def __init__(self, possibleGenres=None, simCount=1000, explorationWeight=1.0):
        """
            possibleGenres = list of possible classifications
            simCount = number of MCTS simulations to run 
            "exploration weight is controlled exploration vs exploitation in UCB formula 
        """
        if possibleGenres is None:
            # if none / if we dont want others to input genres set it as our baseline genres 
            self.possibleGenres = ['blues', 'jazz', 'country', 'pop', 'rock']
        else:
            self.possibleGenres = possibleGenres
        
        self.simCount = simCount
        self.explorationWeight = explorationWeight

    def classify (self, audioFeatures):

        # create a root node with the audio features BUT NO GENRE HYP.
        root = MCTSnode(features=audioFeatures)

        # runs the 4 MCTS phases, select, expand, simulate, backpropagate for specified num of sims
        for i in range(self.simCount):

            # calls select method to traverse down tree based on the UCB scores 
            currNode = self.select(root)

            # checking if unexplored genres 
            if not currNode.allGenresExplored(self.possibleGenres):
                currNode = self.expand(currNode)
            
            # runs a MCTS simulation on the currNode to determine a reqrd value for the genre hyp
            reward = self.simulate(currNode)

            # update curr node features like rewards and visits from curr node to root
            self.backpropagate(currNode, reward)

        # selects best child node from root node based on avg reward
        bestChild = root.bestChild(explorationWeight=0.0)

        # calculates confidence in genre pred
        confidence = bestChild.averageReward()

        # returns predicted child and confidence 
        return bestChild.genre, confidence

    # navigates down the tree using UCB scores until reaching a leaf node with unexplored genres  
    def select(self, currNode):

        # continues traversing down tree as long as not leaf and all genres of node have been explored
        while not currNode.isLeaf() and currNode.allGenresExplored(self.possibleGenres):
            # selects best child node based on UCB scores which balances exploration and exploitation
            currNode = currNode.bestChild(explorationWeight=self.explorationWeight)
        return currNode
    

    # adds a child node with a randomly unexplored genre 
    def expand(self, currNode):

        unexploredGenres = []

        #iterates thru all unexplored genres from curr node
        for genre in self.possibleGenres:
            if genre not in currNode.exploredGenres:
                unexploredGenres.append(genre)
            
        if unexploredGenres:
            # chooses random unexplored genre and adds it to child to be explored
            genre = random.choice(unexploredGenres)
            return currNode.addChildren(genre)
            
        return currNode
    
    ### TO DO ###
    # when Taran and Siri finish extraction, need to update how audioFeatires is processed in sim method 
    # will need to map from raw audio featires to meaningful genre characteristics
    # currently uses random rewards, but will need to be replaes with method that compares audio features to genre prototypes 
    # or uses a learned model to predict genre probabilities
    # could do similarity based or classifier based 
    def simulate(self, currNode):

        # checks if root node is none, which has no genre and if it is give it neutral reward .5
        if currNode.genre is None:
            return 0.5

        # placeholder need to be replaced with real feature based simulation 
        features = currNode.features

        # generates random reward which will need to be replaced 
        reward = random.random()

        return reward
    
    def backpropagate(self, currNode, reward):

        #updates all things realted to currNode
        while currNode is not None:
            currNode.visits += 1
            currNode.totalReward += reward
            currNode = currNode.parent






