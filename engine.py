import numpy as np
import random
from collections import defaultdict
import nn

class State(object): 

    def __init__(self):
        self.board = np.zeros((3,3), dtype=np.int8)
        self.plays = 0
        self.cloned = False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "%s, %s, %s" % (self.plays, self.board, self.cloned)
    
    def show(self):
        if not debug:
            return
        print "%s | %s | %s" % (self.board[0][0], self.board[0][1], self.board[0][2])
        print "---------"
        print "%s | %s | %s" % (self.board[1][0], self.board[1][1], self.board[1][2])
        print "---------"
        print "%s | %s | %s" % (self.board[2][0], self.board[2][1], self.board[2][2])

    def doMove(self, player, move):
        if self.board[move[0]][move[1]] != 0:
            if debug:
                print "Illegal move"
            return False
        self.board[move[0], move[1]] = player
        self.plays += 1
        return True

    def checkWin(self, player):
        if self.board[0][0] == player and self.board[0][1] == player and self.board[0][2] == player:
            return True
        if self.board[1][0] == player and self.board[1][1] == player and self.board[1][2] == player:
            return True
        if self.board[2][0] == player and self.board[2][1] == player and self.board[2][2] == player:
            return True
        if self.board[0][0] == player and self.board[1][0] == player and self.board[2][0] == player:
            return True
        if self.board[0][1] == player and self.board[1][1] == player and self.board[2][1] == player:
            return True
        if self.board[0][2] == player and self.board[1][2] == player and self.board[2][2] == player:
            return True
        if self.board[0][0] == player and self.board[1][1] == player and self.board[2][2] == player:
            return True
        if self.board[2][0] == player and self.board[1][1] == player and self.board[0][2] == player:
            return True
        return False

    def getLegalMoves(self):
        legal = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    legal += [(i, j)]

        if debug:
            print "legal boards %s" % legal
        return legal

    def clone(self):
        s = State()
        s.plays = self.plays
        s.board = self.board.copy()
        s.cloned = True
        return s

    def serializeBoard(self):
        x = 0
        for i in range(3):
            for j in range(3):
                x *= 10
                x += self.board[i][j]
        return x

class Player(object):

    def __init__(self, playerNum):
        self.num = playerNum

    def getMove(self, state):
        raise

    def reward(self, reward, newState):
        pass

    def resetForNextGame(self):
        pass

class RandPlayer(Player):

    def getMove(self, state):
        return random.choice(state.getLegalMoves())

class BetterRandPlayer(Player):
    def getMove(self, state):
        legal = state.getLegalMoves()
        if (1, 1) in legal:
            return (1, 1)
        for m in legal:
            temp = state.clone()
            temp.doMove(self.num, m)
            if temp.checkWin(self.num):
                return m
        otherPlayer = 1 # hard coded...
        for m in legal:
            temp = state.clone()
            temp.doMove(otherPlayer, m)
            if temp.checkWin(otherPlayer):
                return m 
        return random.choice(legal)

class TableQExperiencePlayer(Player):
    
    def __init__(self, playerNum):
        self.num = playerNum
        self.epsilon = 1.0
        self.gamma = 0.9
        self.alpha = 0.052
        self.table = {}
        self.last = None
        self.experiences = []
        self.max_experiences = 500
        self.experience_idx = 0

    def getMove(self, state):
        if random.random() < self.epsilon:
            m = random.choice(state.getLegalMoves())
        else:
            m, e = self.predict(state)
        self.last = (state.serializeBoard(), m)
        return m

    def predict(self, state):
        serial = state.serializeBoard()
        moves = state.getLegalMoves()
        scoredMoves = []

        for m in moves:
            r = self.table.get((serial, m), 0)
            scoredMoves += [(r, m)]
        scoredMoves = sorted(scoredMoves, reverse=True)
        #unconditionally take a 'random choice' on the top scoring nodes
        # will often be just one
        if len(scoredMoves) > 0:
            i = 0
            while i < len(scoredMoves) and scoredMoves[i][0] == scoredMoves[0][0]:
                i += 1
            r, m = random.choice(scoredMoves[:i])
            return m, r
        return random.choice(moves), 0

    def reward(self, reward, newState):
        if reward != 1: #Terminal
            update = reward
        else:
            dontcare, nextStateMaxQ = self.predict(newState)
            update = reward + (self.gamma * nextStateMaxQ)
        
        oldv = self.table.get(self.last, None)
        if oldv is None:
            self.table[self.last] = update
        else:
            self.table[self.last] = oldv + self.alpha * (update - oldv)
        
        if len(self.experiences) < self.max_experiences:
            self.experiences.append((self.last, reward, newState.clone()))
        else:
            self.experiences[self.experience_idx] = (self.last, reward, newState.clone())
            self.experience_idx = (self.experience_idx + 1) % self.max_experiences
            batch = random.sample(self.experiences, 50)

            for b in batch:
                last, reward, newState = b
                if reward != 1: # Terminal
                    update = reward
                else:
                    dontcare, nextStateMaxQ = self.predict(newState)
                    update = reward + (self.gamma * nextStateMaxQ)

                oldv = self.table.get(last, None)
                if oldv is None:
                    self.table[last] = update
                else:
                    self.table[last] = oldv + self.alpha * (update - oldv)
                
        
    def resetForNextGame(self):
        self.last = None
        if self.epsilon != 0:
            self.epsilon = max(self.epsilon-0.001, 0.1)


class TableQLearnPlayer(Player):
    
    def __init__(self, playerNum):
        self.num = playerNum
        self.epsilon = 1.0
        self.gamma = 0.9
        self.alpha = 0.1
        self.table = {}
        self.last = None

    def getMove(self, state):
        if random.random() < self.epsilon:
            m = random.choice(state.getLegalMoves())
        else:
            m, e = self.predict(state)
        self.last = (state.serializeBoard(), m)
        return m

    def predict(self, state):
        serial = state.serializeBoard()
        moves = state.getLegalMoves()
        scoredMoves = []

        for m in moves:
            r = self.table.get((serial, m), 0)
            scoredMoves += [(r, m)]
        scoredMoves = sorted(scoredMoves, reverse=True)
        #unconditionally take a 'random choice' on the top scoring nodes
        # will often be just one
        if len(scoredMoves) > 0:
            i = 0
            while i < len(scoredMoves) and scoredMoves[i][0] == scoredMoves[0][0]:
                i += 1
            r, m = random.choice(scoredMoves[:i])
            return m, r
        return random.choice(moves), 0

    def reward(self, reward, newState):
        if reward != 1: #Terminal
            update = reward
        else:
            dontcare, nextStateMaxQ = self.predict(newState)
            update = reward + (self.gamma * nextStateMaxQ)
        
        oldv = self.table.get(self.last, None)
        if oldv is None:
            self.table[self.last] = update
        else:
            self.table[self.last] = oldv + self.alpha * (update - oldv)
        
    def resetForNextGame(self):
        self.last = None
        if self.epsilon != 0:
            self.epsilon = max(self.epsilon-0.001, 0.1)


class NNQExperiencePlayer(Player):
    
    def __init__(self, playerNum):
        self.num = playerNum
        self.epsilon = 1.0
        self.gamma = 0.6
        #self.alpha = 0.052 using network defualt
        self.last = None
        self.experiences = []
        self.max_experiences = 500
        self.experience_idx = 0

        self.NN = nn.Network([nn.Layer(18, 150, activation=nn.ReLU()), nn.Layer(150, 9)])

    def getMove(self, state):
        inpt = self.board2input(state.board)
        if random.random() < self.epsilon or len(self.experiences) < self.max_experiences:
            m = random.choice(state.getLegalMoves())
            maxq = -99999.9999 # not identified
        else:
            m, maxq = self.predict(inpt)
        if debug:
            print "move %s has value %f" % (m, maxq)
        self.last = (inpt, m)
        return m

    def board2input(self, board):
        """3 layers of a board, available, player 1, player 2"""
        #TODO maybe should be avail, us, them? might generalize better to both positions
        inpt = np.zeros((3,3,2))
        for i in range(3):
            for j in range(3):
                if board[i][j] != 0:
                    inpt[i][j][board[i][j]-1] = 1
        return np.reshape(inpt, (1, 18))

    def predict(self, inpt):
        """Limits predictions on input to legal moves"""
        inptCube = np.reshape(inpt, (3,3,2))
        out = self.NN.query(inpt)
        out3 = np.reshape(out, (3,3))
        maxval = -1000
        move = None
        for i in range(3):
            for j in range(3):
                if inptCube[i][j][0] == 0 and inptCube[i][j][1] == 0 and out3[i][j] > maxval:
                    move = (i, j)
                    maxval = out3[i][j]
        return move, maxval


    def reward(self, reward, newState):
        last_state, last_move = self.last
        rs = (last_state.copy(), last_move, reward, self.board2input(newState.board))
        if len(self.experiences) < self.max_experiences:
            self.experiences.append(rs)
        else:
            self.experiences[self.experience_idx] = rs
            self.experience_idx = (self.experience_idx + 1) % self.max_experiences
         
         
            num_samples = 50
            batch = random.sample(self.experiences, num_samples)
            batch[0] = rs
            x_s = np.zeros((num_samples, 18))
            y_s = np.zeros((num_samples, 9))

            for i, b in enumerate(batch):

                last_input, last_move, last_move_reward, next_input = b
                if last_move_reward != 1 and last_move_reward != -20: # Terminal
                    update = last_move_reward
                else:
                    unneededMove, nextStateMaxQ = self.predict(next_input)
                    update = last_move_reward + (self.gamma * nextStateMaxQ)
                
                last_state_reshaped = np.reshape(last_input, (3,3,2))
                oldv = self.NN.query(last_input) 
                rewardv = np.reshape(oldv, (3, 3))
                rewardv[last_move[0]][last_move[1]] = update
                
                for i in range(3):
                    for j in range(3):
                        if last_state_reshaped[i][j][0] == 0 and last_state_reshaped[i][j][1] == 0:
                            rewardv[i][j] = -50

                x_s[i] = last_input
                y_s[i] = np.reshape(rewardv, -1)

            self.NN.batchFit(x_s, y_s)
                
        
    def resetForNextGame(self):
        self.last = None
        if self.epsilon != 0:
            self.epsilon = max(self.epsilon-0.0001, 0.1)




def playGame(player1, player2):
    s = State()
    p = [player1, player2]
    idx = 0
    while s.plays != 9:
        m = p[idx].getMove(s)
        valid = s.doMove(idx+1, m)
        if debug:
            print "Player %d moves at %s" % (idx+1, m)
        s.show()
        if not valid:
            if debug:
                print "Player %d played an invalid move and loses" % (idx+1)
            p[idx].reward(-20, s)
            p[(idx+1)%2].reward(5, s) #smaller reward for other player forfeiting
            return 3
        if s.checkWin(idx+1):
            if debug:
                print "Player %d wins" % (idx+1)
            p[idx].reward(100, s)
            p[(idx+1)%2].reward(-10, s)
            return idx+1
        #draw
        if s.plays == 9:
            p[0].reward(0, s) 
            p[1].reward(0, s) 
            if debug:
                print "game ends in a draw"
            return 0 
        #If the other player didn't just win, tell the previous player their reward
        if s.plays > 1:
            p[(idx+1)%2].reward(1, s)
        idx = (idx + 1) % 2
    print "shouldn't get here"
    return -1

debug = False

def main():
    global debug
    p1 = NNQExperiencePlayer(1)
    p2 = NNQExperiencePlayer(2)
    p3 = BetterRandPlayer(2)
    
    d = defaultdict(int)
    d2 = defaultdict(int)
    for i in range(30000):
        if i < 19000:
            d[playGame(p1, p2)]+=1
            p1.resetForNextGame()
            p2.resetForNextGame()

        if i >= 18000:
            d2[playGame(p1, p3)]+=1 
            p1.resetForNextGame()
            p3.resetForNextGame()
        if i % 1000 == 0:
            print d, d2
            d = defaultdict(int)
            d2 = defaultdict(int)

    print "root moves"
    #for s, a in p1.table:
    #    if s == 0:
    #        print s, a, p1.table[(s, a)]

    print "starting learned result test"
    p1.epsilon = 0.0
    #p2.epsilon = 0.0
    d = defaultdict(int)
    for i in range(100):
        d[playGame(p1, p3)] += 1
        p1.resetForNextGame()
        p3.resetForNextGame()

    print d

    debug = True
    playGame(p1, p3)
    x = np.zeros((3,3,2))
    print np.reshape(p1.NN.query(np.reshape(x, -1)), (3,3))


if __name__ == '__main__':
    main()
