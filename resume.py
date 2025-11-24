import neat
from snake_game import SnakeGame

def eval(net):
    game = SnakeGame(net, use_neural_network=True, difficulty=60)
    score = game.run()
    return score


input_size = 8
hidden_size = 16
output_size = 4

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # print("genome: ", genome)
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval(net)

def resume_neat(config_path, checkpoint_path, additional_generations=100):
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Restore population from checkpoint
    p = neat.Checkpointer.restore_checkpoint(checkpoint_path)

    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=3, time_interval_seconds=None,
                                     filename_prefix='neat-checkpoint-'))

    # Run for additional generations
    winner = p.run(eval_genomes, additional_generations)

    print(f'\nBest genome:\n{winner}')
    print(f'Fitness: {winner.fitness}')

    # Test winner
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    game = SnakeGame(difficulty=25, neural_network=net)
    fitness = game.run()
    print(f'Winner achieved fitness: {fitness}')

    return winner, stats

resume_neat("snake_config.ini", "./out-snake/neat-checkpoint-14", 2)