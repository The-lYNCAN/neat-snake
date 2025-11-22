import pickle

from snake_game import SnakeGame, SimpleNeuralNetwork
import neat
import math
import visualize
import os


def eval(net):
    game = SnakeGame(net, use_neural_network=True, difficulty=200)
    score = game.run()
    return score



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # print("genome: ", genome)
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval(net)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, "./snake_config.ini")
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5,
           filename_prefix='out-snake/neat-checkpoint-'))

best_genome = p.run(eval_genomes, 100)
with open('out-snake/best_genome.pkl', 'wb') as f:
    pickle.dump(best_genome, f)
net = neat.nn.FeedForwardNetwork.create(best_genome, config)
game = SnakeGame(net, use_neural_network=True, difficulty=10)
score = game.run()
out_dir = "./output_viz"
node_names = {}
visualize.draw_net(config, best_genome, True,
   node_names=node_names, directory=out_dir)
visualize.plot_stats(stats, ylog=False, view=True,
   filename=os.path.join(out_dir, 'avg_fitness.svg'))
visualize.plot_species(stats, view=True,
   filename=os.path.join(out_dir, 'speciation.svg'))
