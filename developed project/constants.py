"""

 ██████  ██████  ███    ██ ███████ ████████  █████  ███    ██ ████████ ███████    ██████  ██    ██ 
██      ██    ██ ████   ██ ██         ██    ██   ██ ████   ██    ██    ██         ██   ██  ██  ██  
██      ██    ██ ██ ██  ██ ███████    ██    ███████ ██ ██  ██    ██    ███████    ██████    ████   
██      ██    ██ ██  ██ ██      ██    ██    ██   ██ ██  ██ ██    ██         ██    ██         ██    
 ██████  ██████  ██   ████ ███████    ██    ██   ██ ██   ████    ██    ███████ ██ ██         ██    
                                                                                                   
                                                                                                   

Constants and configuration settings for the AI driving simulation.
Contains all global variables, colors, and configuration values.
"""

from collections import deque

# Mode configuration: 0 = Training, 1 = Destination, 2 = Demo
# - Mode 0 (Training): AI cars evolve and learn to drive
# - Mode 1 (Destination): Click two points to see the best AI drive from start to destination
# - Mode 2 (Demo): Continuously cycle through random destinations automatically
MODE = 2

# Population and simulation settings
POPULATION_SIZE = 30

# Window dimensions - can be overridden via command line args
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# Simulation timing
FPS = 60

# Advanced overlay detail level (0=off, 1=basic, 2=extended)
OVERLAY_DETAIL_LEVEL = 0

# Waypoint spacing configuration (can be tuned without editing method bodies)
WAYPOINT_SPACING_DEFAULT = 150  # pixels between resampled checkpoints
WAYPOINT_MIN_SPACING = 100
WAYPOINT_MAX_SPACING = 220
TARGET_WAYPOINT_COUNT = 25  # Adaptive goal for long paths

# Global variables for tracking statistics
fitness_history = deque(maxlen=100)  # Keep last 100 generations
generation_history = deque(maxlen=100)
saved_cars_history = deque(maxlen=100)

# Destination mode variables
destination_mode_start = None
destination_mode_end = None
destination_mode_car = None
destination_mode_state = "waiting"  # "waiting", "selecting_start", "selecting_end", "driving"

# Demo mode variables
demo_mode_timer = 0
demo_mode_next_destination_delay = 180  # frames to wait before next destination

# Raycast configuration - Enhanced setup with more sensors
NUM_RAYCASTS = 8
RAYCAST_SPREAD = 180  # 180 degree spread for comprehensive sensing

# Frame timing globals
frame_time_ms_last = 0.0
frame_time_ms_avg = 0.0