# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"
from math import sqrt

from game import Directions, Actions
import util


class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats


class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats


def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None


def closestEatable(pos, walls, ghostState):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        for ghost in ghostState:
            # if ghost[0] == 1:
            #     print("scared")
            #     print (pos_x, pos_y)
            #     print (ghost[1])
            if ghost[0] == 1 and (pos_x, pos_y) in Actions.getLegalNeighbors(ghost[1], walls):
                return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None


class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features


class NewExtractor(FeatureExtractor):
    """
    Design you own feature extractor here. You may define other helper functions you find necessary.
    """

    def getFeatures(self, state, action):
        "*** YOUR CODE HERE ***"

        food = state.getFood()
        walls = state.getWalls()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of active ghosts 1-step away
        num_of_ghost = 0
        ghostState = []
        features["strong"] = 0
        min_timer = 9999
        for i in range(1, state.getNumAgents()):
            isScared = (state.getGhostState(i).scaredTimer > 0)
            ghostPosition = state.getGhostPosition(i)
            if isScared:
                min_timer = min(min_timer, state.getGhostState(i).scaredTimer)
                features["strong"] = 1
            ghostState.append((isScared, ghostPosition))
            if (next_x, next_y) in Actions.getLegalNeighbors(ghostPosition, walls) and not isScared:
                num_of_ghost += 1

        if ((next_x, next_y) == ghost[1] for ghost in ghostState):
            features["#-of-ghosts-0-step-away"] = num_of_ghost

        features["#-of-ghosts-1-step-away"] = num_of_ghost

        if features["strong"] == 1:

            features["first_strong_score"] = state.getScore() / 3000
            features["min_timer"] = min_timer
            # if ((next_x, next_y) in Actions.getLegalNeighbors(ghost[1], walls) for ghost in ghostState):
            #     features["eats-food"] = 1.0

            g_dist = closestEatable((next_x, next_y), walls, ghostState)
            if g_dist is not None:
                # make the distance a number less than one otherwise the update will diverge wildly
                features["closest-Eatable-ghost"] = float(g_dist) / (walls.width * walls.height)

        else:
            # if there is no danger of ghosts then add the food feature
            if not features["#-of-ghosts-1-step-away"]:
                if food[next_x][next_y]:
                    features["eats-food"] = 1.0

            dist = closestFood((next_x, next_y), food, walls)
            if dist is not None:
                # make the distance a number less than one otherwise the update will diverge wildly
                features["closest-food"] = float(dist) / (walls.width * walls.height)

            # posx = next_x
            # posy = next_y
            # tri_dist = 0
            # for gx, gy in state.getGhostPositions():
            #     tri_dist += sqrt((posx - gx) ** 2 + (posy - gy) ** 2) / (walls.width * walls.height)
            #     posx = gx
            #     posy = gy
            # tri_dist += sqrt((posx - next_x) ** 2 + (posy - next_y) ** 2) / (walls.width * walls.height)
            # features["pacman-ghost-dist"] = tri_dist / 3

        features.divideAll(10.0)
        return features
