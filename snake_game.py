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

        # Initialize Pygame
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
        my_font = pygame.font.SysFont('times new roman', 90)
        game_over_surface = my_font.render('YOU DIED', True, self.red)
        game_over_rect = game_over_surface.get_rect()
        game_over_rect.midtop = (self.frame_size_x/2, self.frame_size_y/4)
        if self.difficulty != 200:

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
        # else:
        #     net = SimpleNeuralNetwork(8, 17, 4).forward(self.get_game_state())
        #     print(net)
        #     if net == 0:
        #         self.change_to = 'UP'
        #     if net == 1:
        #         self.change_to = 'DOWN'
        #     if net == 2:
        #         self.change_to = 'LEFT'
        #     if net == 3:
        #         self.change_to = 'RIGHT'

    def update(self):
        """Update the game state."""
        if self.game_over_flag:
            return
        self.position_history.append(tuple(self.snake_pos))
        if len(self.position_history) > 50:
            self.position_history.pop(0)

        # Check for looping (repeated positions)
        position_counts = {}
        # for pos in self.position_history:
        #     position_counts[pos] = position_counts.get(pos, 0) + 1
        #     if position_counts[pos] > 10:  # Same position visited too often
        #         self.fitness -= 10  # Penalize looping
        #         return self.game_over()
        if self.snake_body in self.position_history:
            self.score -= 100
            return self.game_over()
        self.position_history.append(self.snake_body[0])

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

        current_distance = abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])
        if self.last_food_distance is not None:
            if current_distance < self.last_food_distance:
                self.fitness += 1  # Reward for moving closer
            else:
                self.fitness -= 2  # Penalty for moving away
        self.last_food_distance = current_distance

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
            self.score -= 125
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

    def get_inputs(self):
        rays = []
        angles = [45, 90, -45, -90, 135, -135, 180]
        direction = {"DOWN": 0, "UP": 1, "RIGHT": 2, "LEFT": 3}
        direction = direction[self.direction]

        # Frame normalization values
        diag = math.sqrt(self.frame_size_x ** 2 + self.frame_size_y ** 2)

        # Corners (unused but kept)
        topLeft = (0, 0)
        topRight = (self.frame_size_x - 1, 0)
        bottomLeft = (0, self.frame_size_y - 1)
        bottomRight = (self.frame_size_x - 1, self.frame_size_y - 1)

        (x, y) = self.snake_pos

        # Normalize head position
        norm_x = x / self.frame_size_x
        norm_y = y / self.frame_size_y

        # ---- Movement direction encoding (unchanged) ----
        if direction == 0:  # DOWN
            rays.append(0)
            rays.append(0)

            downStraight = (self.frame_size_y - y) / self.frame_size_y
            rightStraight = (self.frame_size_x - x) / self.frame_size_x
            leftStraight = x / self.frame_size_x

            for i in angles:
                i = math.radians(i)
                raw = (self.frame_size_y - y) / math.cos(math.radians(i))
                hypo = raw / diag
                rays.append(1 / hypo if hypo != 0 else 0)

            rays.append(1 / leftStraight if leftStraight != 0 else 0)
            rays.append(1 / rightStraight if rightStraight != 0 else 0)
            rays.append(1 / downStraight if downStraight != 0 else 0)

        if direction == 1:  # UP
            rays.append(0)
            rays.append(1)

            upStraight = y / self.frame_size_y
            rightStraight = (self.frame_size_x - x) / self.frame_size_x
            leftStraight = x / self.frame_size_x

            for i in angles:
                i = math.radians(i)
                raw = y / math.cos(math.radians(i))
                hypo = raw / diag
                rays.append(1 / hypo if hypo != 0 else 0)

            rays.append(1 / rightStraight if rightStraight != 0 else 0)
            rays.append(1 / leftStraight if leftStraight != 0 else 0)
            rays.append(1 / upStraight if upStraight != 0 else 0)

        if direction == 2:  # RIGHT
            rays.append(1)
            rays.append(0)

            rightStraight = (self.frame_size_x - x) / self.frame_size_x
            upStraight = y / self.frame_size_y
            downStraight = (self.frame_size_y - y) / self.frame_size_y

            for i in angles:
                i = math.radians(i)
                raw = (self.frame_size_x - x) / math.cos(math.radians(i))
                hypo = raw / diag
                rays.append(1 / hypo if hypo != 0 else 0)

            rays.append(1 / downStraight if downStraight != 0 else 0)
            rays.append(1 / upStraight if upStraight != 0 else 0)
            rays.append(1 / rightStraight if rightStraight != 0 else 0)

        if direction == 3:  # LEFT
            rays.append(1)
            rays.append(1)

            leftStraight = x / self.frame_size_x
            upStraight = y / self.frame_size_y
            downStraight = (self.frame_size_y - y) / self.frame_size_y

            for i in angles:
                i = math.radians(i)
                raw = x / math.cos(math.radians(i))
                hypo = raw / diag
                rays.append(1 / hypo if hypo != 0 else 0)

            rays.append(1 / upStraight if upStraight != 0 else 0)
            rays.append(1 / downStraight if downStraight != 0 else 0)
            rays.append(1 / leftStraight if leftStraight != 0 else 0)

        # ---- Food info ----
        palletX, palletY = self.food_pos

        # Normalize food position
        norm_food_x = palletX / self.frame_size_x
        norm_food_y = palletY / self.frame_size_y

        # Distance to food (normalized)
        disFood = math.sqrt((palletX - x) ** 2 + (palletY - y) ** 2)
        normFoodDist = disFood / diag

        # Angle to food normalized to [0,1]
        base = x - palletX
        angle = math.acos(base / disFood) if disFood != 0 else 0
        normAngle = angle / math.pi

        # Final normalized inputs
        rays.append(norm_y)
        rays.append(norm_x)
        rays.append(1 / normFoodDist if normFoodDist != 0 else 0)
        rays.append(normAngle)
        rays.append(norm_food_x)
        rays.append(norm_food_y)

        return rays

    def step(self, action=None):
        """Perform one step of the game, optionally taking an action."""
        # action = None
        # print(self.snake_body)
        self.snake_matrix = np.zeros((self.frame_size_x, self.frame_size_y))
        # print(self)
        # print(self.get_inputs())
        for i in self.snake_body:
            # print(i)
            self.snake_matrix[i[0], i[1]] = 1


        # print(self.snake_matrix)
        if self.use_neural_network:
            # Use neural network to predict action
            # state = self.get_game_state()
            neuralOut = self.neural_network.activate(self.get_inputs())
            # print(neuralOut)
            action = softmax(neuralOut)
            # print(action)
            action_map = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
            self.change_to = action_map.get(action.index(max(action)), self.direction)
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
            self.score = self.score + 1/40
            if self.idle > 200:
                return self.score + counter/40 - self.last_food_distance/100
            counter += 1
            self.idle += 1
            if self.game_over_flag:
                if counter < 500:
                    return self.score + counter/40 - self.last_food_distance/100 - 100
                else:
                    return self.score + counter/40 - self.last_food_distance/100


# Example usage
if __name__ == "__main__":
    # Run with neural network
    game = SnakeGame()
    final_score = game.run()
    print(f"Final Score: {final_score}")
    # print(SimpleNeuralNetwork(8, 16, 4).forward([1, 2, 3, 4, 54, 6, 7, 8]))

    # Run with keyboard input (for testing)
    # game = SnakeGame(use_neural_network=False)
    # final_score = game.run()
    # print(f"Final Score: {final_score}")