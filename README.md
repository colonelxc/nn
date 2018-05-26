# Mini neural net library and tic-tac-toe player

This code was the result of trying to learn more things about neural nets and
Q-Learning.

nn.py Has all the actual network code. With it you can create a multi-layered
network with either ReLU or just linear activation functions.

engine.py implements several tic-tac-toe bots, from random ones to Q-Learning
ones without a neural network (uses an exhaustive table instead), to the end
result, a neural network powered Q-Learning bot with experience replay.

One of the best resources I found for Q-Learning was http://outlace.com/rlpart3.html
I also made a gridworld bot with my NN lib, but I need to separate the copied Gridworld
code from the code I wrote before posting it.
