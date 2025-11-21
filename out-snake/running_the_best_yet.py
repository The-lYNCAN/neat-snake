import pickle

from snake_game import SnakeGame, SimpleNeuralNetwork
import neat
import math
import visualize
import os

def eval(net):
    game = SnakeGame(net, use_neural_network=True, difficulty=15)
    score = game.run()
    return score


file_path = "./best_genome.pkl"
with open(file_path, 'rb') as f:
    data = pickle.load(f)
print(data)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, "./../snake_config.ini")
net = neat.nn.FeedForwardNetwork.create(data, config)

print(eval(net))