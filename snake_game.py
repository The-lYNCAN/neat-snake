import os
# os.environ['SDL_AUDIODRIVER'] = 'dummy'
# os.environ['XDG_SESSION_TYPE'] = 'x11'  # Force X11 mode
# os.environ['XDG_CURRENT_DESKTOP'] = ''   # No desktop environment
# os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame
import sys
import time
import random
import numpy as np
from neat.math_util import softmax
import math



class SimpleNeuralNetwork:
    """A simple feedforward neural network for demonstration."""
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """Forward pass through the network."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)  # Activation: tanh
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        output = np.argmax(self.z2, axis=1)
        return output[0]  # Return single action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)


class SnakeGame:
    def __init__(self, brain=None, frame_size_x=720, frame_size_y=480, difficulty=10, use_neural_network=False):
        # Difficulty settings
        self.difficulty = difficulty
        self.snake_matrix = np.zeros((frame_size_x, frame_size_y))
        # Window size
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y
        self.idle = 0
        self.cell_w = frame_size_x // 10  # Number of cells horizontally (72)
        self.cell_h = frame_size_y // 10  # Number of cells vertically (48)
        self.body_set = set()
        self.current_direction = 0

        # Initialize Pygame
        pygame.mixer.quit()
        check_errors = pygame.init()
        if check_errors[1] > 0:
            print(f'[!] Had {check_errors[1]} errors when initializing game, exiting...')
            sys.exit(-1)
        else:
            # print('[+] Game successfully initialized')
            pass

        # Initialize game window
        pygame.display.set_caption('Snake Eater')
        if self.difficulty != 200:

            self.game_window = pygame.display.set_mode((frame_size_x, frame_size_y))

        # Colors (R, G, B)
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)
        self.blue = pygame.Color(0, 0, 255)
        self.border_color = pygame.Color(255, 255, 0)
        self.passInputs = []

        # FPS controller
        self.fps_controller = pygame.time.Clock()

        # Neural network setup
        self.use_neural_network = use_neural_network
        if self.use_neural_network:
            # Define input size based on game state (example: 8 features)
            # Features: distances to food (x, y), distances to walls (x, y), distances to body (up, down, left, right)

            self.neural_network = brain

        # Initialize game variables
        self.reset()

    def reset(self):
        """Reset the game to initial state."""
        self.snake_pos = [50, 100]
        self.snake_body = [[100, 50], [100-10, 50], [100-(2*10), 50]]
        self.food_pos = [random.randrange(1, (self.frame_size_x//10)) * 10,
                         random.randrange(1, (self.frame_size_y//10)) * 10]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.score = 0
        self.snake_pos = self.snake_body[0].copy()  # [x, y]

        # ADD: Initialize body_set (exclude head)
        self.body_set = set()
        for block in self.snake_body[1:]:
            self.body_set.add(tuple(block))
        self.game_over_flag = False
        self.position_history = []  # Store last N positions
        self.max_steps = 1000  # Max steps to prevent infinite loops
        self.steps = 0
        self.fitness = 0  # Custom fitness instead of just score
        self.last_food_distance = None

    def get_game_state(self):
        """Extract game state for neural network input."""
        # Example features: distances to food, walls, and body
        head_x, head_y = self.snake_pos
        food_x, food_y = self.food_pos

        # Distances to food
        food_dx = (food_x) / self.frame_size_x
        food_dy = (food_y) / self.frame_size_y

        # Distances to walls
        wall_left = head_x / self.frame_size_x
        wall_right = (self.frame_size_x - head_x) / self.frame_size_x
        wall_up = head_y / self.frame_size_y
        wall_down = (self.frame_size_y - head_y) / self.frame_size_y

        # Distances to body (check if body is in each direction)
        body_up = 1 if any(block[0] == head_x and block[1] == head_y - 10 for block in self.snake_body[1:]) else 0
        body_down = 1 if any(block[0] == head_x and block[1] == head_y + 10 for block in self.snake_body[1:]) else 0
        body_left = 1 if any(block[0] == head_x - 10 and block[1] == head_y for block in self.snake_body[1:]) else 0
        body_right = 1 if any(block[0] == head_x + 10 and block[1] == head_y for block in self.snake_body[1:]) else 0
        direct = 0
        if self.direction == 'UP':
            direct = 3
        if self.direction == 'DOWN':
            direct = 2
        if self.direction == 'LEFT':
            direct = 1
        if self.direction == 'RIGHT':
            direct = 0

        # Combine features into input vector 5
        state = np.array([[food_dx, food_dy, head_x/self.frame_size_x, head_y/self.frame_size_y, direct, 0, 0, 0]])
        return state

    def show_score(self, choice, color, font, size):
        if self.difficulty != 200:

            """Display the score on the screen."""
            score_font = pygame.font.SysFont(font, size)
            score_surface = score_font.render('Score : ' + str(self.score), True, color)
            score_rect = score_surface.get_rect()
            if choice == 1:
                score_rect.midtop = (self.frame_size_x/10, 15)
            else:
                score_rect.midtop = (self.frame_size_x/2, self.frame_size_y/1.25)
            if self.difficulty != 200:
                self.game_window.blit(score_surface, score_rect)

    def game_over(self):
        """Handle game over state, display score, and terminate."""
        if self.difficulty != 200:
            my_font = pygame.font.SysFont('times new roman', 90)
            game_over_surface = my_font.render('YOU DIED', True, self.red)
            game_over_rect = game_over_surface.get_rect()
            game_over_rect.midtop = (self.frame_size_x/2, self.frame_size_y/4)

            self.game_window.fill(self.black)
            self.game_window.blit(game_over_surface, game_over_rect)
            self.show_score(0, self.red, 'times', 20)
            pygame.display.flip()
        # time.sleep(1)
        final_score = self.score
        self.game_over_flag = True
        pygame.quit()
        return final_score

    def handle_events(self):
        """Handle user input events (used only if neural network is disabled)."""
        if not self.use_neural_network:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP or event.key == ord('w'):
                        self.change_to = 'UP'
                    if event.key == pygame.K_DOWN or event.key == ord('s'):
                        self.change_to = 'DOWN'
                    if event.key == pygame.K_LEFT or event.key == ord('a'):
                        self.change_to = 'LEFT'
                    if event.key == pygame.K_RIGHT or event.key == ord('d'):
                        self.change_to = 'RIGHT'
                    if event.key == pygame.K_ESCAPE:
                        pygame.event.post(pygame.event.Event(pygame.QUIT))


    def update(self):
        """Update the game state."""
        if self.game_over_flag:
            return


        # Check for looping (repeated positions)
        position_counts = {}
        # for pos in self.position_history:
        #     position_counts[pos] = position_counts.get(pos, 0) + 1
        #     if position_counts[pos] > 10:  # Same position visited too often
        #         self.fitness -= 10  # Penalize looping
        #         return self.game_over()


        # Update direction, preventing instant reversal
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        # Move the snake
        if self.direction == 'UP':
            self.snake_pos[1] -= 10
        if self.direction == 'DOWN':
            self.snake_pos[1] += 10
        if self.direction == 'LEFT':
            self.snake_pos[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_pos[0] += 10



        # Snake body growing mechanism
        self.snake_body.insert(0, list(self.snake_pos))
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 100
            self.idle = 0
            self.food_spawn = False
        else:
            self.snake_body.pop()

        # Spawn food
        if not self.food_spawn:
            self.food_pos = [random.randrange(1, (self.frame_size_x//10)) * 10,
                             random.randrange(1, (self.frame_size_y//10)) * 10]
        self.food_spawn = True

        # Check game over conditions
        if (self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x-10 or
                self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y-10):
            return self.game_over()
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                return self.game_over()

    def render(self):
        """Render the game state to the screen."""
        if self.game_over_flag:
            return

        # Clear screen
        self.game_window.fill(self.black)

        # Draw borders
        pygame.draw.rect(self.game_window, self.border_color,
                         pygame.Rect(0, 0, self.frame_size_x, self.frame_size_y), 5)

        # Draw snake body
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, self.green, pygame.Rect(pos[0], pos[1], 10, 10))

        # Draw food
        pygame.draw.rect(self.game_window, self.white, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))

        # Draw score
        self.show_score(1, self.white, 'consolas', 20)

        # Update display
        pygame.display.update()

        # Control frame rate
        self.fps_controller.tick(self.difficulty)


    def softmax(self, x):
        """
        Computes the softmax activation function for a given input array or list.

        Args:
            x (np.array or list): The input values (logits).

        Returns:
            np.array: The softmax output, representing a probability distribution.
        """
        # Subtract the maximum value for numerical stability
        # This prevents overflow when computing exponentials of large positive numbers
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def lineEquation(self):
        pass



    def get_inputs(self):
        head_x, head_y = self.snake_pos
        food_x, food_y = self.food_pos
        body = set(tuple(b) for b in self.snake_body[1:])

        # Helper for checking collisions
        def blocked(x, y):
            return (
                x < 0 or x >= self.frame_size_x or
                y < 0 or y >= self.frame_size_y or
                (x, y) in body
            )

        # Movement vectors for each direction
        dirs = {
            "UP":    (0, -10),
            "DOWN":  (0, 10),
            "LEFT":  (-10, 0),
            "RIGHT": (10, 0)
        }

        left_of = {
            "UP": "LEFT",
            "DOWN": "RIGHT",
            "LEFT": "DOWN",
            "RIGHT": "UP"
        }

        right_of = {
            "UP": "RIGHT",
            "DOWN": "LEFT",
            "LEFT": "UP",
            "RIGHT": "DOWN"
        }

        # Compute absolute moves
        dF = dirs[self.direction]
        dL = dirs[left_of[self.direction]]
        dR = dirs[right_of[self.direction]]

        # --- Obstacle sensors (1 = blocked, 0 = free) ---
        Obs_F = 1 if blocked(head_x + dF[0], head_y + dF[1]) else 0
        Obs_L = 1 if blocked(head_x + dL[0], head_y + dL[1]) else 0
        Obs_R = 1 if blocked(head_x + dR[0], head_y + dR[1]) else 0

        # --- Tail distance sensors (normalized) ---
        def tail_distance(dx, dy):
            step = 1
            x, y = head_x, head_y

            while True:
                x += dx
                y += dy

                if blocked(x, y):
                    # Normalize: max distance ≈ board diagonal
                    dist = math.sqrt((x - head_x)**2 + (y - head_y)**2)
                    diag = math.sqrt(self.frame_size_x**2 + self.frame_size_y**2)
                    return dist / diag

                step += 1

        Tail_F = tail_distance(dF[0], dF[1])
        Tail_L = tail_distance(dL[0], dL[1])
        Tail_R = tail_distance(dR[0], dR[1])

        # --- Food direction sensors (one-hot) ---
        Food_L = 1 if food_x < head_x else 0
        Food_R = 1 if food_x > head_x else 0
        Food_U = 1 if food_y < head_y else 0
        Food_D = 1 if food_y > head_y else 0

        # --- Direction one-hot ---
        Dir_U = 1 if self.direction == "UP" else 0
        Dir_R = 1 if self.direction == "RIGHT" else 0
        Dir_D = 1 if self.direction == "DOWN" else 0
        Dir_L = 1 if self.direction == "LEFT" else 0

        return [
            Obs_F, Obs_L, Obs_R,
            Tail_F, Tail_L, Tail_R,
            Food_L, Food_R, Food_U, Food_D,
            Dir_U, Dir_R, Dir_D, Dir_L
        ]



    def step(self, action=None):
        """Perform one step of the game, optionally taking an action."""
        # action = None
        # print(self.snake_body)
        self.snake_matrix = np.zeros((self.frame_size_x, self.frame_size_y))
        # print(self)
        for i in self.snake_body:
            # print(i)
            self.snake_matrix[i[0], i[1]] = 1


        # print(self.snake_matrix)
        if self.use_neural_network:
            # Use neural network to predict action
            # state = self.get_game_state()
            neuralOut = self.neural_network.activate(self.get_inputs())
            action = np.argmax(neuralOut)  # Just get the index of max value
            relative_turns = {
                'UP': ['UP', 'LEFT', 'RIGHT'],
                'DOWN': ['DOWN', 'RIGHT', 'LEFT'],
                'LEFT': ['LEFT', 'DOWN', 'UP'],
                'RIGHT': ['RIGHT', 'UP', 'DOWN']
            }

            # action = 0 → go straight
            # action = 1 → turn left
            # action = 2 → turn right
            self.change_to = relative_turns[self.direction][action]
            # print(action)
        elif action is None:
            # Use keyboard input if no action provided and neural network is disabled
            self.handle_events()
        else:
            # Use provided action (for external control)
            action_map = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
            self.change_to = action_map.get(action, self.direction)

        result = self.update()
        if self.difficulty != 200:
            self.render()

        # Return game state and score if game is over
        state = {
            'snake_pos': self.snake_pos.copy(),
            'snake_body': self.snake_body.copy(),
            'food_pos': self.food_pos.copy(),
            'score': self.score,
            'game_over': self.game_over_flag
        }
        if self.game_over_flag:
            return state, result
        return state, None

    def run(self):
        """Run the game loop."""
        counter = 0
        while True:
            state, score = self.step()
            self.score += .0125
            # if self.idle > 300:
            #     return self.score
            counter += 1
            self.idle += 1
            if self.game_over_flag:
                return self.score


# Example usage
if __name__ == "__main__":
    # Run with neural network
    game = SnakeGame(difficulty=10)
    final_score = game.run()
    print(f"Final Score: {final_score}")
    # print(SimpleNeuralNetwork(8, 16, 4).forward([1, 2, 3, 4, 54, 6, 7, 8]))

    # Run with keyboard input (for testing)
    # game = SnakeGame(use_neural_network=False)
    # final_score = game.run()
    # print(f"Final Score: {final_score}")