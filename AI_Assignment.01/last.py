import numpy as np
import pygame
import sys

# Define constants
WORLD_SIZE = 100
BLOCK_SIZE = 5
MOUSE_LIFE = 100
FOOD_POINTS = 10
STEP_COST = -1
WAIT_COST = -2  # Penalty for staying in the same place
DEATH_PENALTY = -50  # Penalty for death
SENSE_GRID = 3
STARTING_FOOD_COUNT = 100  # Increased food count for more rewards

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID_BACKGROUND = (230, 230, 230)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
LIFE_BAR_COLOR_FULL = (0, 200, 0)
LIFE_BAR_COLOR_LOW = (255, 0, 0)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WORLD_SIZE * BLOCK_SIZE, WORLD_SIZE * BLOCK_SIZE))

class Habitat:
    def __init__(self, world_size=WORLD_SIZE, food_count=STARTING_FOOD_COUNT):
        self.world_size = world_size
        self.map = np.zeros((world_size, world_size))
        self.mouse_location = [world_size // 2, world_size // 2]
        self.mouse_life = MOUSE_LIFE
        self.food_spots = self.spawn_food(food_count)
        self.food_collected = 0  # Track the amount of food collected
        self.refresh_map()
        self.old_location = tuple(self.mouse_location)

    def spawn_food(self, food_count):
        positions = []
        for _ in range(food_count):
            while True:
                x, y = np.random.randint(0, self.world_size, size=2)
                if (x, y) not in positions:  # Avoid placing food on top of each other
                    positions.append((x, y))
                    break
        return positions

    def refresh_map(self):
        self.map.fill(0)
        self.map[self.mouse_location[0], self.mouse_location[1]] = -1  # Mouse indicator
        for (x, y) in self.food_spots:
            self.map[x, y] = 1  # Food indicator

    def capture_senses(self):
        x, y = self.mouse_location
        x_min, x_max = max(0, x - SENSE_GRID // 2), min(self.world_size, x + SENSE_GRID // 2 + 1)
        y_min, y_max = max(0, y - SENSE_GRID // 2), min(self.world_size, y + SENSE_GRID // 2 + 1)
        sensory_zone = self.map[x_min:x_max, y_min:y_max]
        return np.pad(sensory_zone, ((0, SENSE_GRID - sensory_zone.shape[0]), 
                                     (0, SENSE_GRID - sensory_zone.shape[1])), 
                                     mode='constant', constant_values=0)

    def move_mouse(self, direction):
        x, y = self.mouse_location
        new_location = self.mouse_location.copy()

        if direction == 0 and x > 0:  # Up
            new_location[0] -= 1
        elif direction == 1 and x < self.world_size - 1:  # Down
            new_location[0] += 1
        elif direction == 2 and y < self.world_size - 1:  # Right
            new_location[1] += 1
        elif direction == 3 and y > 0:  # Left
            new_location[1] -= 1

        # Apply penalties
        if new_location == self.mouse_location:
            self.mouse_life += WAIT_COST  # Apply wait cost if staying in the same place
        else:
            self.mouse_life += STEP_COST  # Step cost for moving

        self.old_location = tuple(self.mouse_location)
        self.mouse_location = new_location
        self.refresh_map()

    def detect_food(self):
        if tuple(self.mouse_location) in self.food_spots:
            self.food_spots.remove(tuple(self.mouse_location))
            self.food_collected += 1  # Increment food collected
            self.mouse_life = MOUSE_LIFE  # Restore life when food is found
            return FOOD_POINTS
        elif self.mouse_life <= 0:  # Check for death penalty
            return DEATH_PENALTY  # Return death penalty if energy is zero
        return STEP_COST  # Return step cost if no food is found

class Brain:
    def __init__(self, in_size=8, out_size=4):
        self.neural_weights = np.random.rand(in_size, out_size)

    def predict(self, senses):
        flat_senses = senses.flatten()
        sense_vector = np.concatenate([flat_senses[:4], flat_senses[5:]])
        movement_scores = np.dot(sense_vector, self.neural_weights)
        movement_probs = np.exp(movement_scores) / np.sum(np.exp(movement_scores))
        return movement_probs

    def adjust_weights(self, chosen_dir, reward, senses, rate=0.01):
        flat_senses = senses.flatten()
        sense_vector = np.concatenate([flat_senses[:4], flat_senses[5:]])
        target_output = np.zeros(4)
        target_output[chosen_dir] = reward
        movement_scores = np.dot(sense_vector, self.neural_weights)
        movement_probs = np.exp(movement_scores) / np.sum(np.exp(movement_scores))
        weight_adjust = (movement_probs - target_output).reshape(-1, 1) * sense_vector.reshape(1, -1)
        self.neural_weights -= rate * weight_adjust.T

def visualize_habitat(env, info=None):
    screen.fill(WHITE)
    for x in range(env.world_size):
        for y in range(env.world_size):
            rect = pygame.Rect(y * BLOCK_SIZE, x * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            if env.map[x, y] == -1:
                pygame.draw.rect(screen, BLUE, rect)  # Mouse
            elif env.map[x, y] == 1:
                pygame.draw.rect(screen, GREEN, rect)  # Food
            pygame.draw.rect(screen, GRID_BACKGROUND, rect, 1)

    # Update Life Bar with Gradient
    life_percentage = env.mouse_life / MOUSE_LIFE
    bar_color = (int(LIFE_BAR_COLOR_LOW[0] * (1 - life_percentage) + LIFE_BAR_COLOR_FULL[0] * life_percentage),
                  int(LIFE_BAR_COLOR_LOW[1] * (1 - life_percentage) + LIFE_BAR_COLOR_FULL[1] * life_percentage),
                  int(LIFE_BAR_COLOR_LOW[2] * (1 - life_percentage) + LIFE_BAR_COLOR_FULL[2] * life_percentage))
    life_bar_width = int(life_percentage * (WORLD_SIZE * BLOCK_SIZE))
    life_bar_rect = pygame.Rect(10, 10, life_bar_width, 20)
    pygame.draw.rect(screen, bar_color, life_bar_rect)
    pygame.draw.rect(screen, BLACK, pygame.Rect(10, 10, WORLD_SIZE * BLOCK_SIZE, 20), 2)

    # Display Status Info
    font = pygame.font.SysFont(None, 24)
    food_text = font.render(f"Food Remaining: {len(env.food_spots)}", True, BLACK)
    food_collected_text = font.render(f"Food Collected: {env.food_collected}", True, BLACK)
    energy_text = font.render(f"Energy Level: {env.mouse_life}", True, BLACK)
    screen.blit(food_text, (10, 35))
    screen.blit(food_collected_text, (10, 55))
    screen.blit(energy_text, (10, 75))

    if info:
        alert_font = pygame.font.SysFont(None, 36)
        text = alert_font.render(info, True, RED)
        screen.blit(text, (10, 95))

    pygame.display.flip()

def simulate_training(env, model, num_episodes=100):
    for episode in range(num_episodes):
        env.__init__()  # Reset environment for each episode
        total_reward = 0  # Initialize total reward for the episode
        while env.mouse_life > 0 and env.food_spots:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            visualize_habitat(env)
            senses = env.capture_senses()
            movement_probs = model.predict(senses)
            direction = np.random.choice(4) if np.random.rand() < 0.2 else np.random.choice(4, p=movement_probs)
            env.move_mouse(direction)
            points = env.detect_food()
            total_reward += points  # Accumulate total rewards
            model.adjust_weights(direction, points, senses)
            pygame.time.delay(50)

            if points == FOOD_POINTS:
                print(f"Episode {episode}: Food found with reward {points}")
        
        if env.mouse_life <= 0:
            visualize_habitat(env, info="Mouse has perished")
            print(f"Episode {episode} completed with DEATH_PENALTY applied.")
        else:
            print(f"Episode {episode} completed with total reward: {total_reward}")

# Run the simulation
if __name__ == "__main__":
    habitat = Habitat()
    brain = Brain()
   
