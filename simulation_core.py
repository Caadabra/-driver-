"""

███████ ██ ███    ███ ██    ██ ██       █████  ████████ ██  ██████  ███    ██          ██████  ██████  ██████  ███████    ██████  ██    ██ 
██      ██ ████  ████ ██    ██ ██      ██   ██    ██    ██ ██    ██ ████   ██         ██      ██    ██ ██   ██ ██         ██   ██  ██  ██  
███████ ██ ██ ████ ██ ██    ██ ██      ███████    ██    ██ ██    ██ ██ ██  ██         ██      ██    ██ ██████  █████      ██████    ████   
     ██ ██ ██  ██  ██ ██    ██ ██      ██   ██    ██    ██ ██    ██ ██  ██ ██         ██      ██    ██ ██   ██ ██         ██         ██    
███████ ██ ██      ██  ██████  ███████ ██   ██    ██    ██  ██████  ██   ████ ███████  ██████  ██████  ██   ██ ███████ ██ ██         ██    
                                                                                                                                           
                                                                                                                                           
Core simulation logic and main game loop management.
Contains initialization, mode handling, and the main simulation loop.
"""

import pygame
import math
from collections import deque

from constants import *
from car import Car
from osm_roads import OSMRoadSystem
from camera import Camera
from dijkstra_pathfinding import DijkstraPathfinder


class SimulationCore:
    def __init__(self, width, height, mode=0, auto_mode=False, initial_scale=1.0):
        """Initialize the simulation core with display and game state"""
        self.width = width
        self.height = height
        self.mode = mode
        self.auto_mode = auto_mode
        
        # Initialize pygame display in standalone window (no embedding)
        try:
            pygame.init()
            # Always run as a regular window (no SDL embedding)
            self.screen = pygame.display.set_mode((width, height))
            
            # Neutral window title without AI wording
            pygame.display.set_caption("Driver Simulation")
            self.clock = pygame.time.Clock()
            print("Pygame display initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize pygame display: {e}")
            raise
        
        # Initialize game systems
        self.road_system = OSMRoadSystem()
        self.camera = Camera(0, 0, width, height)
        self.camera.zoom = initial_scale  # Set zoom after initialization
        self.pathfinder = DijkstraPathfinder(self.road_system)

        # Simulation state
        self.running = True
        self.cars = []
        self.generation = 0
        self.evolution_timer = 0
        self.evolution_interval = 40 * FPS  # 40 seconds per generation
        
    # Performance tracking (training disabled) — remove unused histories
        
        # Overlay and debugging
        self.overlay_detail_level = 0
        self.frame_time_ms_last = 0.0
        self.frame_time_ms_avg = 0.0
        
        # Mode-specific state
        self.destination_mode_start = None
        self.destination_mode_end = None
        self.destination_mode_car = None
        self.destination_mode_state = "waiting"  # "waiting", "selecting_start", "selecting_end", "driving"
        

        
        self.initialize_mode()

    def create_destination_mode_car(self, start_pos, end_pos):
        """Create a car for destination mode without AI/evolution dependencies"""
        car = Car(start_pos[0], start_pos[1], RED, use_ai=False, road_system=self.road_system, pathfinder=self.pathfinder)
        # Prepare a route to the destination
        path_nodes = self.pathfinder.find_path(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
        if path_nodes:
            car.current_path = path_nodes
            car.path_waypoints = self.pathfinder.path_to_waypoints(path_nodes)
            if hasattr(car, '_resample_waypoints_evenly'):
                car._resample_waypoints_evenly()
            if hasattr(car, '_prune_backward_waypoints'):
                car._prune_backward_waypoints()
            car.current_waypoint_index = 0
            car.individual_destination = end_pos
            if car.path_waypoints:
                car.target_x, car.target_y = car.path_waypoints[0]
                # Orient towards first waypoint
                try:
                    sx, sy = car.x, car.y
                    wp_index = 0
                    if len(car.path_waypoints) > 1 and math.hypot(car.path_waypoints[0][0]-sx, car.path_waypoints[0][1]-sy) < 5:
                        wp_index = 1
                    if wp_index < len(car.path_waypoints):
                        tx, ty = car.path_waypoints[wp_index]
                        dx = tx - sx; dy = ty - sy
                        raw_angle = math.degrees(math.atan2(dx, -dy))
                        car.angle = raw_angle
                except Exception:
                    pass
        return car

    def initialize_mode(self):
        """Initialize the simulation based on the selected mode"""
        if self.mode == 0:  # Training Mode (disabled)
            print("=== TRAINING MODE DISABLED ===")
            print("Switching to Destination Mode")
            self.mode = 1
            # Fall through to destination mode setup below

        elif self.mode == 1:  # Destination Mode
            print("=== DESTINATION MODE ===")
            print("Click two points on the map: first for start position, then for destination")
            self.cars = []

            # Auto mode: pick random start/end and spawn a car immediately, clamp camera
            if self.auto_mode:
                sx, sy, _ = self.road_system.get_random_spawn_point()
                # Try to find a destination far enough and reachable
                ex, ey = sx + 800, sy + 0
                # Use pathfinder to get an actual destination
                node, pos = self.pathfinder.get_random_destination()
                if pos:
                    ex, ey = pos
                car = self.create_destination_mode_car((sx, sy), (ex, ey))
                self.cars = [car]
                self.destination_mode_car = car
                self.destination_mode_state = "driving"
                # Clamp camera to car follow mode immediately
                self.camera.manual_mode = False
                self.camera.follow_target(car.x, car.y)

        elif self.mode == 2:  # Demo Mode
            print("=== DEMO MODE ===")
            print("Autopilot will automatically cycle through random destinations")
            self.cars = []
            
            # Start immediately with first random destination
            sx, sy, _ = self.road_system.get_random_spawn_point()
            node, pos = self.pathfinder.get_random_destination()
            if pos:
                ex, ey = pos
            else:
                ex, ey = sx + 800, sy + 0
            
            car = self.create_destination_mode_car((sx, sy), (ex, ey))
            self.cars = [car]
            self.destination_mode_car = car
            self.destination_mode_state = "driving"
            self.demo_mode_timer = 0
            
            # Auto-follow camera
            self.camera.manual_mode = False
            self.camera.follow_target(car.x, car.y)

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_k:
                    self.overlay_detail_level = (self.overlay_detail_level + 1) % 4
                    states = ["OFF", "BASIC", "EXTENDED", "THINK"]
                    print(f"Overlay mode: {states[self.overlay_detail_level]}")
            elif event.type == pygame.MOUSEBUTTONDOWN and self.mode == 1:
                # Destination mode mouse handling
                if event.button == 1:  # Left click
                    mouse_pos = pygame.mouse.get_pos()
                    world_pos = self.camera.screen_to_world(mouse_pos[0], mouse_pos[1])
                    
                    if self.destination_mode_state == "waiting":
                        self.destination_mode_start = world_pos
                        self.destination_mode_state = "selecting_start"
                        print(f"Start position selected: {world_pos}")
                    elif self.destination_mode_state == "selecting_start":
                        self.destination_mode_end = world_pos
                        self.destination_mode_state = "selecting_end"
                        print(f"End position selected: {world_pos}")
                        
                        # Create the best car for this route
                        car = self.create_destination_mode_car(self.destination_mode_start, self.destination_mode_end)
                        self.cars = [car]
                        self.destination_mode_car = car
                        self.destination_mode_state = "driving"
                        print("Car created and ready to drive!")
                    elif self.destination_mode_state == "driving":
                        # Reset for new route
                        self.destination_mode_start = world_pos
                        self.destination_mode_end = None
                        self.destination_mode_car = None
                        self.cars = []
                        self.destination_mode_state = "selecting_start"
                        print(f"New start position selected: {world_pos}")

    # Training mode fully removed

    def update_destination_mode(self, keys):
        """Update logic for destination mode"""
        # If we're driving in destination mode, auto-follow the car
        if self.destination_mode_state == "driving" and self.destination_mode_car:
            self.camera.manual_mode = False
            self.camera.follow_target(self.destination_mode_car.x, self.destination_mode_car.y)
        self.camera.update(keys)
        
        # Draw roads
        self.road_system.draw_roads(self.screen, self.camera.x, self.camera.y, self.width, self.height, self.camera.zoom)
        
        # Draw selection indicators
        if self.destination_mode_start:
            start_screen = self.camera.world_to_screen(self.destination_mode_start[0], self.destination_mode_start[1])
            pygame.draw.circle(self.screen, GREEN, start_screen, 10)
            pygame.draw.circle(self.screen, WHITE, start_screen, 10, 2)
        
        if self.destination_mode_end:
            end_screen = self.camera.world_to_screen(self.destination_mode_end[0], self.destination_mode_end[1])
            pygame.draw.circle(self.screen, RED, end_screen, 10)
            pygame.draw.circle(self.screen, WHITE, end_screen, 10, 2)
        
        # Update and draw car if in driving state
        if self.destination_mode_state == "driving" and self.destination_mode_car:
            car = self.destination_mode_car
            car.calculate_predictive_path()
            car.update_path_following_accuracy()
            car.move(keys)
            car.update_raycasts()
            car.draw(self.camera, True)  # Always visualize the destination mode car
            
            # Draw modern HUD for destination mode car
            self.hud.draw_hud(self.screen, car, 0, 0, 0, 1, self.mode, self.road_system, self.camera)

    def update_demo_mode(self, keys):
        """Update logic for demo mode"""
        # Auto-follow the car
        if self.destination_mode_state == "driving" and self.destination_mode_car:
            self.camera.manual_mode = False
            self.camera.follow_target(self.destination_mode_car.x, self.destination_mode_car.y)
        self.camera.update(keys)
        
        # Draw roads
        self.road_system.draw_roads(self.screen, self.camera.x, self.camera.y, self.width, self.height, self.camera.zoom)
        
        # Update and draw car
        if self.destination_mode_state == "driving" and self.destination_mode_car:
            car = self.destination_mode_car
            car.calculate_predictive_path()
            car.update_path_following_accuracy()
            car.move(keys)
            car.update_raycasts()
            car.draw(self.camera, True)
            
            # Check if destination reached
            if car.destination_reached:
                self.demo_mode_timer += 1
                if self.demo_mode_timer >= self.demo_mode_next_destination_delay:
                    # Generate new random destination
                    sx, sy, _ = self.road_system.get_random_spawn_point()
                    node, pos = self.pathfinder.get_random_destination()
                    if pos:
                        ex, ey = pos
                    else:
                        ex, ey = sx + 800, sy + 0
                    
                    # Create new car with new destination
                    new_car = self.create_destination_mode_car((sx, sy), (ex, ey))
                    self.destination_mode_car = new_car
                    self.cars = [new_car]
                    self.demo_mode_timer = 0
                    print(f"Demo: New destination at ({ex:.0f}, {ey:.0f})")
            else:
                self.demo_mode_timer = 0  # Reset timer if not at destination
            
            # Draw modern HUD for demo mode car
            self.hud.draw_hud(self.screen, car, 0, 0, 0, 1, self.mode, self.road_system, self.camera)

    def update(self):
        """Main update loop"""
        self.screen.fill((34, 139, 34))  # Dark green background
        
        keys = pygame.key.get_pressed()
        
        if self.mode == 1:  # Destination mode
            self.update_destination_mode(keys)
        elif self.mode == 2:  # Demo mode
            self.update_demo_mode(keys)


    def run(self):
        """Main simulation loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw_overlays()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        return self.running