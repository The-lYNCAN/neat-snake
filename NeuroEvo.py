import pickle
from neat.parallel import ParallelEvaluator
from snake_game import SnakeGame, SimpleNeuralNetwork
import neat
import math
import visualize
import os
import multiprocessing
from tqdm import tqdm
from functools import partial

# Global variables for progress tracking
current_generation = 0
total_genomes = 0
pbar = None


def eval(net):
    game = SnakeGame(net, use_neural_network=True, difficulty=200)
    score = game.run()
    return score


def eval_genome(genome, config):
    """Evaluate a single genome."""
    genome.fitness = 0.0
    try:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = SnakeGame(net, use_neural_network=True, difficulty=200)
        fitness = game.run()
        genome.fitness = fitness
    except Exception as e:
        print(f"\n[Crash] Genome at {id(genome)} crashed: {e}")
        import traceback
        traceback.print_exc()
        genome.fitness = -999999

    return genome.fitness


class ProgressParallelEvaluator(ParallelEvaluator):
    """Custom ParallelEvaluator with progress bar support."""

    def __init__(self, num_workers, eval_function, timeout=None):
        super().__init__(num_workers, eval_function, timeout)
        self.generation = 0
        self.best_fitness_ever = float('-inf')

    def evaluate(self, genomes, config):
        """Evaluate genomes with a beautiful progress bar."""
        global current_generation, total_genomes, pbar

        self.generation += 1
        current_generation = self.generation
        total_genomes = len(genomes)

        # Create beautiful progress bar
        print(f"\n{'=' * 80}")
        print(f"🐍 Generation {self.generation} | Evaluating {total_genomes} genomes on {self.num_workers} cores")
        print(f"{'=' * 80}")

        pbar = tqdm(
            total=total_genomes,
            desc=f"Gen {self.generation:>3}",
            bar_format='{desc}: {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            colour='green',
            ncols=100,
            unit='genome',
            smoothing=0.1
        )

        # Submit all jobs
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        # Gather results with progress updates
        for job, (genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
            pbar.update(1)  # Update progress bar after each job completes

        pbar.close()

        # Calculate and display statistics
        fitnesses = [g.fitness for _, g in genomes if g.fitness is not None]
        if fitnesses:
            best_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            min_fitness = min(fitnesses)

            if best_fitness > self.best_fitness_ever:
                self.best_fitness_ever = best_fitness
                improvement = "🎉 NEW RECORD! 🎉"
            else:
                improvement = ""

            print(f"\n📊 Generation {self.generation} Statistics:")
            print(f"   Best:    {best_fitness:>10.2f} {improvement}")
            print(f"   Average: {avg_fitness:>10.2f}")
            print(f"   Worst:   {min_fitness:>10.2f}")
            print(f"   Record:  {self.best_fitness_ever:>10.2f}")

        print(f"{'─' * 80}\n")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # Disable pygame audio for parallel processing
    os.environ['SDL_AUDIODRIVER'] = 'dummy'

    print("\n" + "=" * 80)
    print("🎮 NEAT Snake Evolution Training")
    print("=" * 80)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "./snake_config.ini"
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(
        5,
        filename_prefix='out-snake/neat-checkpoint-'
    ))

    # Create output directory
    os.makedirs('out-snake', exist_ok=True)

    # Use custom parallel evaluator with progress bar
    num_cores = multiprocessing.cpu_count() - 1
    print(f"\n🚀 Using {num_cores} CPU cores for parallel evaluation")
    print(f"📈 Training for 500 generations\n")

    pe = ProgressParallelEvaluator(num_cores, eval_genome)

    # Run evolution
    best_genome = p.run(pe.evaluate, 500)

    # Save best genome
    print("\n" + "=" * 80)
    print("🏆 Training Complete!")
    print("=" * 80)
    print(f"💾 Saving best genome (fitness: {best_genome.fitness:.2f})...")

    with open('out-snake/best_genome.pkl', 'wb') as f:
        pickle.dump(best_genome, f)

    print("✅ Best genome saved to 'out-snake/best_genome.pkl'")

    # Test best genome
    print("\n🎮 Testing best genome...")
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    game = SnakeGame(net, use_neural_network=True, difficulty=10)
    score = game.run()
    print(f"🎯 Final test score: {score:.2f}")

    # Generate visualizations
    print("\n📊 Generating visualizations...")
    out_dir = "./output_viz"
    os.makedirs(out_dir, exist_ok=True)

    node_names = {}
    visualize.draw_net(
        config,
        best_genome,
        True,
        node_names=node_names,
        directory=out_dir
    )
    visualize.plot_stats(
        stats,
        ylog=False,
        view=True,
        filename=os.path.join(out_dir, 'avg_fitness.svg')
    )
    visualize.plot_species(
        stats,
        view=True,
        filename=os.path.join(out_dir, 'speciation.svg')
    )

    print("✨ Visualizations saved to './output_viz/'")
    print("\n" + "=" * 80)
    print("🎉 All done! Happy evolving! 🐍")
    print("=" * 80 + "\n")