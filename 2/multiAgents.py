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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        foodDists = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        closestFoodHeu = 9 / min(foodDists) if foodDists else 0
        ghostDists = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        closestGhostHeu = -11 if ghostDists and min(ghostDists) < 2 else 0
        return successorGameState.getScore() + closestFoodHeu + closestGhostHeu


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        def isEnd(gameState: GameState, depth: int):
            return gameState.isWin() or gameState.isLose() or self.depth == depth

        def maxValue(gameState: GameState, depth: int):
            if isEnd(gameState, depth):
                return self.evaluationFunction(gameState)

            v = -float('inf')
            legalActs = gameState.getLegalActions()
            for act in legalActs:
                v = max(v, minValue(gameState.generateSuccessor(0, act), depth, 1))

            return v

        def minValue(gameState: GameState, depth: int, agentIndex: int):
            if isEnd(gameState, depth):
                return self.evaluationFunction(gameState)

            v = float('inf')
            legalActs = gameState.getLegalActions(agentIndex)
            for act in legalActs:
                successor = gameState.generateSuccessor(agentIndex, act)
                v = min(v, maxValue(successor, depth + 1)) if agentIndex == gameState.getNumAgents() - 1 \
                    else min(v, minValue(successor, depth, agentIndex + 1))

            return v

        return sorted(
            [(action, minValue(gameState.generateSuccessor(0, action), 0, 1)) for action in
             gameState.getLegalActions()], key=lambda x: x[1], reverse=True)[0][0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def isEnd(gameState: GameState, depth: int):
            return gameState.isWin() or gameState.isLose() or self.depth == depth

        def maxValue(gameState: GameState, depth: int, alpha: float, beta: float):
            if isEnd(gameState, depth):
                return self.evaluationFunction(gameState)

            v = -float('inf')
            legalActs = gameState.getLegalActions()
            for act in legalActs:
                v = max(v, minValue(gameState.generateSuccessor(0, act), depth, 1, alpha, beta))
                if v > beta:
                    return v

                alpha = max(alpha, v)

            return v

        def minValue(gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float):
            if isEnd(gameState, depth):
                return self.evaluationFunction(gameState)

            v = float('inf')
            legalActs = gameState.getLegalActions(agentIndex)
            for act in legalActs:
                successor = gameState.generateSuccessor(agentIndex, act)
                v = min(v, maxValue(successor, depth + 1, alpha, beta)) if agentIndex == gameState.getNumAgents() - 1 \
                    else min(v, minValue(successor, depth, agentIndex + 1, alpha, beta))
                if v < alpha:
                    return v

                beta = min(beta, v)

            return v

        alpha, beta = -float('inf'), float('inf')
        maxVal = -float('inf')
        act = None
        for action in gameState.getLegalActions():
            val = minValue(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if val > maxVal:
                maxVal = val
                act = action

            alpha = max(alpha, val)

        return act


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

        def isEnd(gameState: GameState, depth: int):
            return gameState.isWin() or gameState.isLose() or self.depth == depth

        def maxValue(gameState: GameState, depth: int):
            if isEnd(gameState, depth):
                return self.evaluationFunction(gameState)

            v = -float('inf')
            legalActs = gameState.getLegalActions()
            for act in legalActs:
                v = max(v, expValue(gameState.generateSuccessor(0, act), depth, 1))

            return v

        def expValue(gameState: GameState, depth: int, agentIndex: int):
            if isEnd(gameState, depth):
                return self.evaluationFunction(gameState)

            v = 0
            legalActs = gameState.getLegalActions(agentIndex)
            for act in legalActs:
                successor = gameState.generateSuccessor(agentIndex, act)
                v += maxValue(successor, depth + 1) if agentIndex == gameState.getNumAgents() - 1 else expValue(
                    successor, depth, agentIndex + 1)

            return v / len(legalActs) if len(legalActs) else 0

        return sorted(
            [(action, expValue(gameState.generateSuccessor(0, action), 0, 1)) for action in
             gameState.getLegalActions()], key=lambda x: x[1], reverse=True)[0][0]


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    A closer closest food makes food heuristic higher. 
    A closer ghost makes the ghost heuristic lower.
    A closer scared ghost makes the ghost heuristic much higher.
    """
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    foodDists = [util.manhattanDistance(pos, food) for food in foods]
    closestFoodHeu = 9 / min(foodDists) if foodDists else 0
    ghostHeu = 0
    for ghost in ghostStates:
        dist = util.manhattanDistance(pos, ghost.getPosition())
        if dist > 0:
            ghostHeu += 99 / dist if ghost.scaredTimer > 0 else -11 / dist

    return currentGameState.getScore() + closestFoodHeu + ghostHeu


# Abbreviation
better = betterEvaluationFunction
