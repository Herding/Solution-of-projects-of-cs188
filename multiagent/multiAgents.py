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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        avgFoodDisOfSuc = 0
        foods = newFood.asList()
        for food in foods:
            avgFoodDisOfSuc += manhattanDistance(newPos, food)
        if len(foods):
            avgFoodDisOfSuc /= len(foods)

        cnt = 0
        ghostDis = 0
        for ghost in newGhostStates:
            if newScaredTimes[cnt] > 0:
                cnt += 1
                continue
            ghostDis += manhattanDistance(newPos, ghost.getPosition())
            cnt += 1

        # 1 / (0.1 + successorGameState.getNumFood())
        # print newPos

        return successorGameState.getScore() + ghostDis - avgFoodDisOfSuc

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 1)
        util.raiseNotDefined()

    def minValue(self, gameState, depth, ghostIndex):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        minVal = float("inf")
        legalActs = gameState.getLegalActions(ghostIndex)
        sucs =  []
        for action in legalActs:
            sucs.append(gameState.generateSuccessor(ghostIndex, action))

        if ghostIndex == gameState.getNumAgents() - 1:
            if depth < self.depth:
                for suc in sucs:
                    minVal = min(minVal, self.maxValue(suc, depth + 1))
            else :
                for suc in sucs:
                    minVal = min(minVal, self.evaluationFunction(suc))
        else :
            for suc in sucs:
                minVal = min(minVal, self.minValue(suc, depth, ghostIndex + 1))

        return minVal

    def maxValue(self, gameState, depth):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        maxVal = float("-inf")
        chooseAct = Directions.STOP

        for action in gameState.getLegalActions(0):
            suc = gameState.generateSuccessor(0, action)
            val = self.minValue(suc, depth, 1)
            if val > maxVal:
                maxVal = val
                chooseAct = action

        if depth > 1:
            return maxVal
        return chooseAct

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 1, float("-inf"), float("inf"))
        util.raiseNotDefined()

    def maxValue(self, gameState, depth, alpha, beta):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        maxVal = float("-inf")
        chooseAct = Directions.STOP
        for action in gameState.getLegalActions(0):
            suc = gameState.generateSuccessor(0, action)
            val = self.minValue(suc, depth, 1, alpha, beta)
            if val > maxVal:
                maxVal = val
                chooseAct = action

            if maxVal > beta:
                return maxVal
            alpha = max(alpha, maxVal)

        if depth > 1:
            return maxVal
        return chooseAct


    def minValue(self, gameState, depth, ghostIndex, alpha, beta):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        minVal = float("inf")
        for action in gameState.getLegalActions(ghostIndex):
            suc = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex == gameState.getNumAgents() - 1:
                if depth < self.depth:
                    val = self.maxValue(suc, depth + 1, alpha, beta)
                else :
                    val = self.evaluationFunction(suc)
            else :
                val = self.minValue(suc, depth, ghostIndex + 1, alpha, beta)
            minVal = min(minVal, val)

            if minVal < alpha:
                return minVal
            beta = min(beta, minVal)

        return minVal

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 1)
        util.raiseNotDefined()

    def maxValue(self, gameState, depth):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        maxVal = float("-inf")
        chooseAct = Directions.STOP
        for action in gameState.getLegalActions(0):
            suc = gameState.generateSuccessor(0, action)
            val = self.expectimin(suc, depth, 1)
            if val > maxVal:
                maxVal = val
                chooseAct = action

        if depth > 1:
            return maxVal
        return chooseAct


    def expectimin(self, gameState, depth, ghostIndex):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        legalActs = gameState.getLegalActions(ghostIndex)
        sucs = []
        for action in legalActs:
            sucs.append(gameState.generateSuccessor(ghostIndex, action))
        expectVal = 0
        probaOfSuc = 1.0 / len(legalActs)
        if ghostIndex == gameState.getNumAgents() - 1:
            if depth < self.depth:
                for suc in sucs:
                    expectVal += probaOfSuc * self.maxValue(suc, depth + 1)
            else :
                for suc in sucs:
                    expectVal += probaOfSuc * self.evaluationFunction(suc)
        else :
            for suc in sucs:
                expectVal += probaOfSuc * self.expectimin(suc, depth, ghostIndex + 1)

        return expectVal

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    nowPos = currentGameState.getPacmanPosition()
    avgFoodDis = 0
    foods = currentGameState.getFood().asList()
    for food in foods:
        avgFoodDis += manhattanDistance(nowPos, food)
        if len(foods):
            avgFoodDis /= len(foods)

    maxFoodDis = 0
    minFoodDis = float("inf")
    for food in foods:
        dist = manhattanDistance(nowPos, food)
        maxFoodDis = max(maxFoodDis, dist)
        minFoodDis = min(minFoodDis, dist)

    nowGhostStates = currentGameState.getGhostStates()
    nowScaredTimes = [ghostState.scaredTimer for ghostState in nowGhostStates]
    ghostDist = 0
    cnt = 0
    for ghost in nowGhostStates:
        if nowScaredTimes[cnt] > 0:
            cnt += 1
            continue
        ghostDist += manhattanDistance(nowPos, ghost.getPosition())
        cnt += 1

    numOfFood = currentGameState.getNumFood()
    numOfCapsules = len(currentGameState.getCapsules())

    costOfPacman = maxFoodDis + numOfFood + numOfCapsules

    return currentGameState.getScore() + ghostDist / (1 + avgFoodDis + minFoodDis) - (costOfPacman + avgFoodDis)/ (1 + ghostDist) - 64 * numOfCapsules

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

