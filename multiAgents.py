# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        closestFoodDistance = manhattanDistance(newPos, newFood[0])
        for food in newFood.asList(): #Finds closest food pellet
            closestFoodDistance = min(manhattanDistance(newPos, food), closestFoodDistance)
        if closestFoodDistance > 0: #Increase score if food is close
            score += 1 / closestFoodDistance
        else:
            score += 2 #Greater score increase when Pacman eats pellet

        scaredTimer = 0
        closestGhostDistance = manhattanDistance(newPos, successorGameState.getGhostPositions()[0])
        for ghost in newGhostStates: #Finds scared timer for ghost that is closest to Pacman
            scaredTimer = ghost.scaredTimer
            ghostPos = ghost.getPosition()
            closestGhostDistance = min(manhattanDistance(newPos, ghostPos), closestGhostDistance)
        if scaredTimer == 0:
            if closestGhostDistance == 0: #Minimize score for states that causes Pacman's death
                score = 0
            else: #Lower score if non-scared ghosts are nearby
                score -= 1 / closestGhostDistance
        else:
            score += scaredTimer #Increase score if ghosts are scared
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state, depth, agent):
            maxScore = float('-inf')
            legalActions = state.getLegalActions(agent)
            for action in legalActions: #Find maximal score for each legal action
                successor = state.generateSuccessor(agent, action)
                score = minimax(successor, depth, agent + 1)
                maxScore = max(maxScore, score)
            return maxScore

        def minValue(state, depth, agent):
            minScore = float('inf')
            legalActions = state.getLegalActions(agent)
            for action in legalActions: #Find minimal score for each legal action
                successor = state.generateSuccessor(agent, action)
                if agent < state.getNumAgents() - 1: #Each minimizing agent must get a turn in each depth level
                    score = minimax(successor, depth, agent + 1)
                else: #Maximizing agent's turn after iterating through minimizing agents
                    score = minimax(successor, depth - 1, 0)
                minScore = min(minScore, score)
            return minScore

        def minimax(state, depth, agent):
            if state.isLose() or state.isWin() or depth == 0: #Evaluation of terminal states
                return self.evaluationFunction(state)
            if agent == 0: #Maximizer
                return maxValue(state, depth, agent)
            elif agent < state.getNumAgents(): #Minimizer
                return minValue(state, depth, agent)

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        maxScore = float('-inf')
        for action in legalActions: #Finds maximal score of root node by recursing on children
            successor = gameState.generateSuccessor(0, action)
            score = minimax(successor, self.depth, 1) #Finds minimax score of root's children
            if score > maxScore: #Based on children's scores, we extract optimal solution
                maxScore = score
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state, depth, agent, alpha, beta):
            maxScore = float('-inf')
            legalActions = state.getLegalActions(agent)
            for action in legalActions: #Find maximal score for each legal action
                successor = state.generateSuccessor(agent, action)
                score = minimax(successor, depth, agent + 1, alpha, beta)
                maxScore = max(maxScore, score)

                #Alpha-beta pruning
                alpha = max(alpha, maxScore)
                if maxScore > beta:
                    return maxScore

            return maxScore

        def minValue(state, depth, agent, alpha, beta):
            minScore = float('inf')
            legalActions = state.getLegalActions(agent)
            for action in legalActions: #Find minimal score for each legal action
                successor = state.generateSuccessor(agent, action)
                if agent < state.getNumAgents() - 1: #Each minimizing agent must get a turn in each depth level
                    score = minimax(successor, depth, agent + 1, alpha, beta)
                else: #Maximizing agent's turn after iterating through minimizing agents
                    score = minimax(successor, depth - 1, 0, alpha, beta)
                minScore = min(minScore, score)

                #Alpha-beta pruning
                beta = min(beta, minScore)
                if minScore < alpha:
                    return minScore

            return minScore

        def minimax(state, depth, agent, alpha, beta):
            if state.isLose() or state.isWin() or depth == 0: #Evaluation of terminal states
                return self.evaluationFunction(state)
            if agent == 0: #Maximizer
                return maxValue(state, depth, agent, alpha, beta)
            elif agent < state.getNumAgents(): #Minimizer
                return minValue(state, depth, agent, alpha, beta)

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        maxScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in legalActions: #Finds maximal score of root node by recursing on children
            successor = gameState.generateSuccessor(0, action)
            score = minimax(successor, self.depth, 1, alpha, beta) #Finds minimax score of root's children
            if score > maxScore: #Based on children's scores, we extract optimal solution
                maxScore = score
                bestAction = action

            alpha = max(alpha, maxScore)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def maxValue(state, depth, agent):
            maxScore = float('-inf')
            legalActions = state.getLegalActions(agent)
            for action in legalActions: #Find maximal score for each legal action
                successor = state.generateSuccessor(agent, action)
                score = minimax(successor, depth, agent + 1)
                maxScore = max(maxScore, score)

            return maxScore

        def expValue(state, depth, agent):
            expScore = 0
            legalActions = state.getLegalActions(agent)

            for action in legalActions: #Find expected score for each legal action
                successor = state.generateSuccessor(agent, action)
                if agent < state.getNumAgents() - 1: #Each chance agent must get a turn in each depth level
                    score = minimax(successor, depth, agent + 1)
                else: #Maximizing agent's turn after iterating through chance agents
                    score = minimax(successor, depth - 1, 0)

                expScore += score / len(legalActions)  # Calculate score based on equal probability of legal actions

            return expScore

        def minimax(state, depth, agent):
            if state.isLose() or state.isWin() or depth == 0: #Evaluation of terminal states
                return self.evaluationFunction(state)
            if agent == 0: #Maximizer
                return maxValue(state, depth, agent)
            elif agent < state.getNumAgents(): #Chance agent
                return expValue(state, depth, agent)

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        maxScore = float('-inf')

        for action in legalActions: #Finds maximal score of root node by recursing on children
            successor = gameState.generateSuccessor(0, action) #Finds expected score of root's children
            score = minimax(successor, self.depth, 1)
            if score > maxScore: #Based on children's scores, we extract optimal solution
                maxScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin(): #Return large score if no food remaining
        return 1000
    elif currentGameState.isLose(): #Return negative score upon losing
        return -1000

    pelletCoords = currentGameState.getFood().asList()
    powerPelletCoords = currentGameState.getCapsules()
    pos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    score = currentGameState.getScore()

    closestFoodDistance = manhattanDistance(pos, pelletCoords[0]) #Finds closest food pellet if food is remaining
    for food in pelletCoords:  # Finds closest food pellet
        closestFoodDistance = min(manhattanDistance(pos, food), closestFoodDistance)
    if closestFoodDistance > 0:  # Increase score if food is close
        score += 1 / closestFoodDistance
    else:
        score += 2  # Greater score increase when Pacman eats pellet

    scaredTimer = 0
    closestGhostDistance = manhattanDistance(pos, currentGameState.getGhostPositions()[0])
    for ghost in ghostStates:  # Finds scared timer for ghost that is closest to Pacman
        scaredTimer = ghost.scaredTimer
        ghostPos = ghost.getPosition()
        closestGhostDistance = min(manhattanDistance(pos, ghostPos), closestGhostDistance)
    if scaredTimer == 0:
        if closestGhostDistance == 0:  # Minimize score for states that causes Pacman's death
            score = 0
        else:  # Lower score if non-scared ghosts are nearby
            score -= 1 / closestGhostDistance
    else:
        score += scaredTimer  # Increase score if ghosts are scared
    return score

# Abbreviation
better = betterEvaluationFunction
