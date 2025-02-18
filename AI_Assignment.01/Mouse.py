import numpy as np
import pygame
import sys

# Define constants
GRID_SIZE = 100
CELL_SIZE = 5
MOUSE_ENERGY = 100
FOOD_REWARD = 10
MOVE_PENALTY = -1
STAY_PENALTY = -2  # Additional penalty for staying in the same place
SENSORY_MATRIX_SIZE = 3
INITIAL_FOOD_COUNT = 50  # Increased food count

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)  # Food color
MOUSE_COLOR = (255, 165, 0)  # New mouse color (orange)
GRID_COLOR = (169, 169, 169)  # New grid color (grey)
ENERGY_BAR_COLOR = (0, 128, 0)
RED = (255, 0, 0)

class Environment:
    def __init__(self, grid_size=GRID_SIZE, initial_food=INITIAL_FOOD_COUNT):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        self.mouse_position = [grid_size // 2, grid_size // 2]
        self.mouse_energy = MOUSE_ENERGY
        self.food_positions = self.generate_food(initial_food)
        self.update_grid()
        self.previous_position = tuple(self.mouse_position)  # Track previous position to detect staying

    def generate_food(self, food_count):
        food_positions = []
        for _ in range(food_count):
            x, y = np.random.randint(0, self.grid_size, size=2)
            food_positions.append((x, y))
        return food_positions

    def update_grid(self):
        self.grid.fill(0)
        self.grid[self.mouse_position[0], self.mouse_position[1]] = -1  # Mouse indicator
        for (x, y) in self.food_positions:
            self.grid[x, y] = 1  # Food indicator

    def get_sensory_input(self):
        x, y = self.mouse_position
        x_start, x_end = max(0, x-1), min(self.grid_size, x+2)
        y_start, y_end = max(0, y-1), min(self.grid_size, y+2)
        sensory_area = self.grid[x_start:x_end, y_start:y_end]
        return np.pad(sensory_area, ((0, SENSORY_MATRIX_SIZE-sensory_area.shape[0]), 
                                     (0, SENSORY_MATRIX_SIZE-sensory_area.shape[1])), 
                                     mode='constant', constant_values=0)

    def move_mouse(self, direction):
        x, y = self.mouse_position
        if direction == 0 and x > 0:  # North
            x -= 1
        elif direction == 1 and x < self.grid_size - 1:  # South
            x += 1
        elif direction == 2 and y < self.grid_size - 1:  # East
            y += 1
        elif direction == 3 and y > 0:  # West
            y -= 1
        self.previous_position = tuple(self.mouse_position)  # Store current position before moving
        self.mouse_position = [x, y]
        self.mouse_energy -= 1
        self.update_grid()

    def check_food(self):
        if tuple(self.mouse_position) in self.food_positions:
            self.food_positions.remove(tuple(self.mouse_position))
            self.mouse_energy = MOUSE_ENERGY  # Replenish energy when food is found
            return FOOD_REWARD
        elif tuple(self.mouse_position) == self.previous_position:
            return STAY_PENALTY  # Penalty for staying in the same place
        return MOVE_PENALTY


class Model:
    def __init__(self, input_size=8, output_size=4):
        self.weights = np.random.rand(input_size, output_size)

    def forward(self, sensory_input):
        flat_input = sensory_input.flatten()
        sensory_vector = np.concatenate([flat_input[:4], flat_input[5:]])
        scores = np.dot(sensory_vector, self.weights)
        probabilities = np.exp(scores) / np.sum(np.exp(scores))
        return probabilities

    def backpropagate(self, direction, reward, sensory_input, learning_rate=0.01):
        flat_input = sensory_input.flatten()
        sensory_vector = np.concatenate([flat_input[:4], flat_input[5:]])
        target = np.zeros(4)
        target[direction] = reward
        scores = np.dot(sensory_vector, self.weights)
        probs = np.exp(scores) / np.sum(np.exp(scores))
        gradient = (probs - target).reshape(-1, 1) * sensory_vector.reshape(1, -1)
        self.weights -= learning_rate * gradient.T


def render(env, message=None):
    screen.fill(WHITE)
    
    # Draw the grid, mouse, and food
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if env.grid[x, y] == -1:
                pygame.draw.rect(screen, MOUSE_COLOR, rect)  # Updated mouse color
            elif env.grid[x, y] == 1:
                pygame.draw.rect(screen, GREEN, rect)  # Food
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)  # Updated grid color

    # Draw the energy bar
    energy_bar_width = int((env.mouse_energy / MOUSE_ENERGY) * (GRID_SIZE * CELL_SIZE))
    energy_bar_rect = pygame.Rect(10, GRID_SIZE * CELL_SIZE + 10, energy_bar_width, 20)
    pygame.draw.rect(screen, ENERGY_BAR_COLOR, energy_bar_rect)
    pygame.draw.rect(screen, BLACK, pygame.Rect(10, GRID_SIZE * CELL_SIZE + 10, GRID_SIZE * CELL_SIZE, 20), 2)  # Border for the energy bar

    # Display any message if provided
    if message:
        font = pygame.font.SysFont(None, 36)
        text = font.render(message, True, RED)
        screen.blit(text, (10, GRID_SIZE * CELL_SIZE + 40))

    pygame.display.flip()


def train_and_visualize(env, model, episodes=100):
    for episode in range(episodes):
        print(f"Starting episode {episode+1}")
        env.__init__()  # Reset environment for each episode
        while env.mouse_energy > 0 and env.food_positions:
            render(env)
            sensory_input = env.get_sensory_input()
            probabilities = model.forward(sensory_input)
            # Randomize actions to encourage exploration
            if np.random.rand() < 0.2:  # 20% random move for exploration
                direction = np.random.choice(4)
            else:
                direction = np.random.choice(4, p=probabilities)
            env.move_mouse(direction)
            reward = env.check_food()
            model.backpropagate(direction, reward, sensory_input)
            pygame.time.delay(50)
            if reward == FOOD_REWARD:
                print(f"Episode {episode+1}: Food found with reward {reward}")
        # Display game over message if mouse dies
        if env.mouse_energy <= 0:
            render(env, message="Game Over - Mouse has died")
            pygame.time.delay(1000)
        print(f"Episode {episode+1} completed.")


# Initialize environment and model
env = Environment()
model = Model()

# Initialize PyGame
pygame.init()
screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE + 70))  # Extra space for energy bar
pygame.display.set_caption("Mouse and Food Simulation with Energy Bar")

try:
    train_and_visualize(env, model)
finally:
    pygame.quit()
    sys.exit()
