"""
Constants and configuration for the simulation
"""

# Screen dimensions
WIDTH, HEIGHT = 1920, 1080
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# Simulation parameters
POPULATION_SIZE = 50
EVOLUTION_INTERVAL_SECONDS = 15
NUM_CHECKPOINTS = 10
MIN_CHECKPOINT_DISTANCE = 250

# OSM coordinates (Auckland, New Zealand by default)
DEFAULT_CENTER_LAT = -36.8825825
DEFAULT_CENTER_LON = 174.9143453
DEFAULT_RADIUS = 1000
