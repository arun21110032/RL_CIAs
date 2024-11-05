import numpy as np
import random
import matplotlib.pyplot as plt

# Constants for the grid world
GRID_SIZE = 100
OBSTACLE_PROB = 0.2  # 20% chance of a cell being an obstacle
REWARD_GOAL = 100
REWARD_MOVE = -1
REWARD_OBSTACLE = -10
DISCOUNT_FACTOR = 0.9
THRESHOLD = 0.01  # Convergence threshold for value iteration

# Actions: Up, Down, Left, Right
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class GridWorld:
    def __init__(self, grid_size=GRID_SIZE, obstacle_prob=OBSTACLE_PROB):
        self.grid_size = grid_size
        self.obstacle_prob = obstacle_prob
        self.grid = np.zeros((grid_size, grid_size))
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.init_grid()

    def init_grid(self):
        # Set up obstacles randomly
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if random.random() < self.obstacle_prob and (i, j) not in [self.start, self.goal]:
                    self.grid[i, j] = -1  # Obstacle
        
        # Ensure start and goal points are open
        self.grid[self.start] = 0
        self.grid[self.goal] = 0

    def is_valid(self, state):
        """Check if the state is within bounds and not an obstacle."""
        x, y = state
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[x, y] != -1:
            return True
        return False

    def get_next_state(self, state, action):
        """Return the next state after taking the action."""
        x, y = state
        dx, dy = action
        next_state = (x + dx, y + dy)
        if self.is_valid(next_state):
            return next_state
        return state  # If the move is invalid, stay in the same state

    def get_reward(self, state):
        """Return the reward for the given state."""
        if state == self.goal:
            return REWARD_GOAL
        elif self.grid[state] == -1:
            return REWARD_OBSTACLE
        return REWARD_MOVE


class ValueIterationAgent:
    def __init__(self, grid_world, discount_factor=DISCOUNT_FACTOR, threshold=THRESHOLD):
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = threshold
        self.value_table = np.zeros((grid_world.grid_size, grid_world.grid_size))

    def value_iteration(self):
        while True:
            delta = 0
            new_value_table = np.copy(self.value_table)

            for x in range(self.grid_world.grid_size):
                for y in range(self.grid_world.grid_size):
                    state = (x, y)
                    if state == self.grid_world.goal or self.grid_world.grid[state] == -1:
                        continue  # Skip goal and obstacle cells
                    
                    value_max = float('-inf')
                    for action in ACTIONS:
                        next_state = self.grid_world.get_next_state(state, action)
                        reward = self.grid_world.get_reward(next_state)
                        value = reward + self.discount_factor * self.value_table[next_state]
                        value_max = max(value_max, value)

                    new_value_table[x, y] = value_max
                    delta = max(delta, abs(self.value_table[x, y] - new_value_table[x, y]))

            self.value_table = new_value_table
            if delta < self.threshold:
                break

    def get_policy(self):
        """Extract the optimal policy from the value table."""
        policy = np.zeros((self.grid_world.grid_size, self.grid_world.grid_size), dtype=(int, 2))
        for x in range(self.grid_world.grid_size):
            for y in range(self.grid_world.grid_size):
                state = (x, y)
                if state == self.grid_world.goal or self.grid_world.grid[state] == -1:
                    continue  # Skip goal and obstacle cells

                best_action = None
                best_value = float('-inf')

                for action in ACTIONS:
                    next_state = self.grid_world.get_next_state(state, action)
                    reward = self.grid_world.get_reward(next_state)
                    value = reward + self.discount_factor * self.value_table[next_state]

                    if value > best_value:
                        best_value = value
                        best_action = action

                policy[x, y] = best_action

        return policy


def visualize_grid(grid_world, policy=None):
    """Visualize the grid world and the optimal policy."""
    grid = grid_world.grid
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray')
    start_x, start_y = grid_world.start
    goal_x, goal_y = grid_world.goal
    ax.text(start_y, start_x, 'S', color='green', ha='center', va='center', fontsize=14, fontweight='bold')  # Start point
    ax.text(goal_y, goal_x, 'G', color='blue', ha='center', va='center', fontsize=14, fontweight='bold')  # Goal point
    if policy is not None:
        for x in range(grid_world.grid_size):
            for y in range(grid_world.grid_size):
                if grid_world.grid[x, y] == -1 or (x, y) == grid_world.goal:
                    continue
                dx, dy = policy[x, y]
                ax.arrow(y, x, dy * 0.3, dx * -0.3, head_width=0.3, head_length=0.3, fc='red', ec='red')

    plt.show()


# Create the grid world and the agent
grid_world = GridWorld()
agent = ValueIterationAgent(grid_world)

# Run Value Iteration
agent.value_iteration()

# Extract and visualize the optimal policy
optimal_policy = agent.get_policy()
visualize_grid(grid_world, optimal_policy)
