"""
3D Visualization of Car AI Simulation using Three.js
This program creates a 3D environment that visualizes the car simulation data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import math
import threading
import time
from typing import List
import websockets
from flask import Flask, render_template_string
import random
from constants import *
from car import Car
from osm_roads import OSMRoadSystem
from dijkstra_pathfinding import DijkstraPathfinder

app = Flask(__name__)

# Global variables to store simulation state
simulation_core = None
simulation_running = False
websocket_clients = set()
emergency_stop_active = False  # Global emergency stop state

class HeadlessSimulationCore:
    """Headless version of simulation core that runs without pygame display"""
    
    def __init__(self, width=1280, height=720, mode=0, auto_mode=False, initial_scale=1.0):
        """Initialize headless simulation without pygame display"""
        self.width = width
        self.height = height
        self.mode = mode
        self.auto_mode = auto_mode
        
        # Initialize game systems without pygame
        self.road_system = OSMRoadSystem()
        
        # Create a simple camera class for headless mode
        class HeadlessCamera:
            def __init__(self, x=0, y=0, width=1280, height=720):
                self.x = x
                self.y = y
                self.width = width
                self.height = height
                self.zoom = 1.0
                self.manual_mode = False
            
            def follow_target(self, x, y):
                self.x = x
                self.y = y
            
            def update(self, keys=None):
                pass
        
        self.camera = HeadlessCamera(0, 0, width, height)
        self.camera.zoom = initial_scale
        self.pathfinder = DijkstraPathfinder(self.road_system)
        
        # Simulation state
        self.running = True
        self.cars = []
        self.generation = 0
        self.evolution_timer = 0
        self.evolution_interval = 40 * 60  # 40 seconds at 60 FPS
        
        # Path validation state
        self.paths_validated = False
        self.valid_paths_count = 0
        self.required_paths_count = 0
        
        # Performance tracking
        from collections import deque
        self.fitness_history = deque(maxlen=100)
        self.generation_history = deque(maxlen=100)
        self.saved_cars_history = deque(maxlen=100)
        
        # Mode-specific state
        self.destination_mode_start = None
        self.destination_mode_end = None
        self.destination_mode_car = None
        self.destination_mode_state = "waiting"
        
        self.initialize_mode()
    
    def initialize_mode(self):
        """Initialize the simulation based on the selected mode (headless version)"""
        self.road_system.load_roads()
        print(f"Road system loaded with {len(self.road_system.road_segments)} segments")
        
        if self.mode == 0:  # Training mode
            self.initialize_training_mode()
        elif self.mode == 1:  # Destination mode
            self.initialize_destination_mode()
            
        # Validate paths after initialization
        if not self.validate_all_paths():
            print("WARNING: Not all cars have valid paths. Simulation will not start until paths are fixed.")
    
    def initialize_training_mode(self):
        """Initialize training mode with population"""
        from evolution import load_population, create_car_from_data, assign_shared_destination
        import random
        
        # Load existing population or create new one
        population_data, loaded_generation = load_population()
        
        if population_data:
            print(f"Loading existing population from generation {loaded_generation}")
            self.generation = loaded_generation + 1
            self.cars = []
            
            # Recreate cars from saved data
            for car_data in population_data['cars']:
                if len(self.cars) >= POPULATION_SIZE:
                    break
                car = create_car_from_data(car_data, self.road_system, self.pathfinder)
                self.cars.append(car)
            
            # Fill remaining slots if needed
            while len(self.cars) < POPULATION_SIZE:
                spawn_x, spawn_y, spawn_angle = self.road_system.get_random_spawn_point()
                car = Car(spawn_x, spawn_y, random.choice([RED, GREEN, BLUE]), 
                         use_ai=True, road_system=self.road_system, pathfinder=self.pathfinder)
                car.angle = spawn_angle
                car.generate_individual_destination()
                if car.path_waypoints:
                    car.orient_to_first_waypoint()
                self.cars.append(car)
        else:
            print("Creating new population for training...")
            self.cars = []
            for _ in range(POPULATION_SIZE):
                spawn_x, spawn_y, spawn_angle = self.road_system.get_random_spawn_point()
                car = Car(spawn_x, spawn_y, random.choice([RED, GREEN, BLUE]), 
                         use_ai=True, road_system=self.road_system, pathfinder=self.pathfinder)
                car.angle = spawn_angle
                car.generate_individual_destination()
                if car.path_waypoints:
                    car.orient_to_first_waypoint()
                self.cars.append(car)
        
        # Assign a shared destination for fair training
        assign_shared_destination(self.cars, self.pathfinder)
        print(f"Training mode: Created population of {len(self.cars)} cars")
        
        # Set camera to follow center of action
        if self.cars:
            center_x = sum(car.x for car in self.cars) / len(self.cars)
            center_y = sum(car.y for car in self.cars) / len(self.cars)
            self.camera.follow_target(center_x, center_y)
    
    def initialize_destination_mode(self):
        """Initialize destination mode with enhanced path validation"""
        print("Initializing destination mode with path validation...")
        
        max_attempts = 5
        cars_created = 0
        target_cars = 1  # For destination mode, we usually want 1 car
        
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts} to create cars with valid paths...")
            
            # Try to create a car with a valid route
            car = self.create_valid_route_car(max_attempts=10)
            if car and hasattr(car, 'path_waypoints') and car.path_waypoints and len(car.path_waypoints) > 1:
                self.cars = [car]
                self.destination_mode_car = car
                self.destination_mode_state = "driving"
                self.camera.follow_target(car.x, car.y)
                cars_created = 1
                print(f"Created car with {len(car.path_waypoints)} waypoints")
                break
            else:
                print(f"Failed to create car with valid route on attempt {attempt + 1}")
        
        if cars_created == 0:
            print("CRITICAL: Failed to create any cars with valid paths after all attempts")
            print("Creating fallback car with simplified path to allow simulation to start")
            from car import Car
            
            # Create a fallback car with a simple straight-line path
            fallback_car = Car(640, 360, 0)
            
            # Create a simple fallback path (straight line)
            start_x, start_y = 640, 360
            end_x, end_y = start_x + 200, start_y  # Simple straight path
            
            fallback_car.path_waypoints = [
                (start_x, start_y),
                (start_x + 50, start_y),
                (start_x + 100, start_y),
                (start_x + 150, start_y),
                (end_x, end_y)
            ]
            fallback_car.destination = (end_x, end_y)
            
            self.cars = [fallback_car]
            self.destination_mode_car = fallback_car
            self.destination_mode_state = "driving"
            cars_created = 1
            print(f"Created fallback car with simple {len(fallback_car.path_waypoints)} waypoint path")
            
        print(f"Destination mode initialized with {len(self.cars)} car(s)")
    
    def create_valid_route_car(self, max_attempts=10):
        """Create a car with a valid route using enhanced pathfinding validation"""
        print(f"Attempting to create car with valid route (max {max_attempts} attempts)...")
        
        for attempt in range(max_attempts):
            try:
                # Get available roads
                roads = list(self.road_system.roads.values()) if hasattr(self.road_system, 'roads') else []
                
                # If no roads in self.road_system.roads, try road_segments
                if len(roads) < 2:
                    roads = self.road_system.road_segments if hasattr(self.road_system, 'road_segments') else []
                
                if len(roads) < 2:
                    print(f"ERROR: Not enough roads for pathfinding (found {len(roads)})")
                    print("Checking if road system was properly initialized...")
                    
                    # Force road system reload
                    try:
                        self.road_system.load_roads()
                        roads = self.road_system.road_segments if hasattr(self.road_system, 'road_segments') else []
                        print(f"After reload: found {len(roads)} road segments")
                    except Exception as e:
                        print(f"Road reload failed: {e}")
                    
                    if len(roads) < 2:
                        print("Still no roads available, will use fallback after all attempts")
                        continue
                
                # Select random start and end roads with minimum distance
                attempts_for_distance = 20
                start_road, end_road = None, None
                
                for dist_attempt in range(attempts_for_distance):
                    potential_start = random.choice(roads)
                    potential_end = random.choice([r for r in roads if r != potential_start])
                    
                    # Calculate distance between roads
                    distance = math.hypot(
                        potential_end.start[0] - potential_start.start[0],
                        potential_end.start[1] - potential_start.start[1]
                    )
                    
                    # Ensure minimum distance for meaningful path
                    if distance > 200:  # Minimum distance for valid route
                        start_road, end_road = potential_start, potential_end
                        break
                
                if not start_road or not end_road:
                    print(f"Attempt {attempt + 1}: Could not find roads with sufficient distance")
                    continue
                
                # Get start and end positions
                start_pos = start_road.start
                end_pos = end_road.end
                
                print(f"Attempt {attempt + 1}: Finding path from {start_pos} to {end_pos}")
                
                # Find path using pathfinder
                path_nodes = self.pathfinder.find_path(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
                
                if path_nodes and len(path_nodes) > 1:
                    # Convert node IDs to waypoint coordinates
                    path = self.pathfinder.path_to_waypoints(path_nodes)
                    
                    if path and len(path) > 1:
                        # Validate path quality
                        path_length = self._calculate_path_length(path)
                        direct_distance = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
                        
                        # Path should not be too much longer than direct distance (reasonable routing)
                        if path_length > 0 and path_length < direct_distance * 3:  # Allow up to 3x direct distance
                            # Create car at start position
                            car = self.create_destination_mode_car(start_pos, end_pos)
                            if car:
                                # Keep the path prepared inside create_destination_mode_car
                                # (already offset/resampled/pruned and oriented). Do not overwrite it.
                                car.destination = end_pos  # Store destination
                            print(f"Created car with valid path: {len(path)} waypoints, {path_length:.1f} units long")
                            return car
                        else:
                            print(f"Attempt {attempt + 1}: Failed to create car object")
                    else:
                        print(f"Attempt {attempt + 1}: Path too long or invalid ({path_length:.1f} vs {direct_distance:.1f})")
                else:
                    print(f"Attempt {attempt + 1}: No path found or path too short")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with exception: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"FAILED: Could not create valid route car after {max_attempts} attempts")
        return None
    
    def _calculate_path_length(self, path):
        """Calculate total length of a path"""
        if not path or len(path) < 2:
            return 0
            
        total_length = 0
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            total_length += math.hypot(next_point[0] - current[0], next_point[1] - current[1])
            
        return total_length
    
    def create_destination_mode_car(self, start_pos, end_pos):
        """Create a car for destination mode with pathfinding"""
        from evolution import create_destination_mode_car
        return create_destination_mode_car(start_pos, end_pos, self.road_system, self.pathfinder)
    

    

    def validate_all_paths(self):
        """Validate that all cars have valid paths before allowing simulation to start"""
        if not self.cars:
            print("No cars to validate paths for")
            return False
            
        valid_count = 0
        total_count = len(self.cars)
        
        for car in self.cars:
            if hasattr(car, 'path_waypoints') and car.path_waypoints and len(car.path_waypoints) > 1:
                # Additional validation: check if path actually connects start to destination
                if self._validate_path_connectivity(car):
                    valid_count += 1
                    print(f"Car {id(car)} has valid path with {len(car.path_waypoints)} waypoints")
                else:
                    print(f"Car {id(car)} has invalid or disconnected path")
            else:
                print(f"Car {id(car)} has no valid path waypoints")
        
        self.valid_paths_count = valid_count
        self.required_paths_count = total_count
        self.paths_validated = (valid_count == total_count and valid_count > 0)
        
        print(f"Path validation: {valid_count}/{total_count} cars have valid paths")
        if self.paths_validated:
            print("All cars have valid paths - simulation can start")
        else:
            print("Not all cars have valid paths - simulation blocked")
            
        return self.paths_validated
    
    def _validate_path_connectivity(self, car):
        """Check if a car's path is actually connected and reachable"""
        if not hasattr(car, 'path_waypoints') or not car.path_waypoints:
            return False
            
        path = car.path_waypoints
        if len(path) < 2:
            return False
            
        # Check if waypoints are reasonably connected (no huge gaps)
        max_gap = 200  # Increased maximum allowed distance between consecutive waypoints
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            distance = math.hypot(next_point[0] - current[0], next_point[1] - current[1])
            if distance > max_gap:
                print(f"Path gap too large: {distance} units between waypoints {i} and {i+1}")
                return False
                
        # Check if start point is close to car's actual position (more lenient)
        start_point = path[0]
        car_distance_to_start = math.hypot(start_point[0] - car.x, start_point[1] - car.y)
        if car_distance_to_start > 100:  # Increased tolerance - car should be reasonably near start of path
            print(f"Car too far from path start: {car_distance_to_start} units (allowing up to 100)")
            return False
            
        return True
    
    def update_simulation(self):
        """Update one frame of the simulation (headless)"""
        if not self.running:
            return
            
        # Check if paths are validated before allowing updates
        if not getattr(self, 'paths_validated', False):
            # Try to re-validate paths (but with limits to prevent infinite loops)
            if not hasattr(self, '_validation_attempts'):
                self._validation_attempts = 0
                
            if self._validation_attempts < 3:  # Maximum 3 validation attempts
                self._validation_attempts += 1
                print(f"Validation attempt {self._validation_attempts}/3...")
                if not self.validate_all_paths():
                    if self._validation_attempts >= 3:
                        print("FORCING SIMULATION START: Path validation failed after 3 attempts")
                        print("Setting paths_validated=True to prevent infinite blocking")
                        self.paths_validated = True  # Force validation to prevent infinite loop
                        self.valid_paths_count = len(self.cars)
                        self.required_paths_count = len(self.cars)
                    else:
                        print("Simulation blocked: No valid paths found. Retrying...")
                        return  # Block simulation updates for now
                else:
                    print("Path validation successful")
            else:
                # Already tried 3 times, force simulation to start
                print("Path validation bypassed after maximum attempts")
                self.paths_validated = True
            
        if self.mode == 0:  # Training mode
            self.update_training_mode()
        elif self.mode == 1:  # Destination mode
            self.update_destination_mode()
    
    def update_training_mode(self):
        """Update training mode logic"""
        # Find best car and update cars
        alive_cars = 0
        saved_cars = 0
        best_car = None
        
        for car in self.cars:
            if not car.crashed or car.saved_state:
                car.calculate_fitness()
                if best_car is None or car.fitness > best_car.fitness:
                    best_car = car
                alive_cars += 1
                if car.saved_state:
                    saved_cars += 1
        
        # Update cars
        active_best_car = None
        for car in self.cars:
            if not car.crashed or car.saved_state:
                if not car.saved_state:
                    car.calculate_predictive_path()
                    car.update_path_following_accuracy()
                    car.move({})  # No keyboard input in headless mode
                    
                    if active_best_car is None or car.fitness > active_best_car.fitness:
                        if not car.crashed and not car.saved_state:
                            active_best_car = car
                
                car.update_raycasts()
        
        # Update camera to follow best car
        if active_best_car:
            self.camera.follow_target(active_best_car.x, active_best_car.y)
        
        # Evolution logic
        self.evolution_timer += 1
        unsaved_active_exists = any((not c.crashed) and (not c.saved_state) for c in self.cars)
        saved_active_exists = any((not c.crashed) and c.saved_state for c in self.cars)
        only_saved_remaining = (not unsaved_active_exists) and saved_active_exists
        
        if self.evolution_timer >= self.evolution_interval or alive_cars == 0 or only_saved_remaining:
            from evolution import evolve_population
            self.cars = evolve_population(self.cars, len(self.cars), self.road_system, self.pathfinder, self.generation)
            self.evolution_timer = 0
            self.generation += 1
            print(f"Generation {self.generation} started with {len(self.cars)} cars")
            
            # Update statistics
            if best_car:
                self.fitness_history.append(best_car.fitness)
                self.generation_history.append(self.generation)
                self.saved_cars_history.append(saved_cars)
    
    def update_destination_mode(self):
        """Update destination mode logic with path validation"""
        # Double-check path validation before allowing car movement
        if not getattr(self, 'paths_validated', False):
            print("Destination mode blocked: Cars do not have valid paths")
            return
            
        if self.destination_mode_car and not self.destination_mode_car.crashed:
            car = self.destination_mode_car
            
            # Verify car still has valid path before updating
            if not hasattr(car, 'path_waypoints') or not car.path_waypoints or len(car.path_waypoints) < 2:
                print(f"Car {id(car)} lost its path! Attempting to regenerate...")
                new_car = self.create_valid_route_car(max_attempts=3)
                if new_car:
                    # Transfer some state but use new path
                    new_car.x = car.x
                    new_car.y = car.y
                    new_car.angle = car.angle
                    self.destination_mode_car = new_car
                    self.cars = [new_car]
                    # Re-validate paths
                    self.validate_all_paths()
                    print("Route regenerated and validated successfully")
                else:
                    print("Failed to regenerate path. Stopping car.")
                    car.crashed = True
                return
            
            car.calculate_predictive_path()
            car.update_path_following_accuracy()
            car.move({})
            car.update_raycasts()
            self.camera.follow_target(car.x, car.y)
            
            # Check if car reached destination
            if hasattr(car, 'current_waypoint_index') and car.path_waypoints:
                if car.current_waypoint_index >= len(car.path_waypoints) - 1:
                    print(f"Car reached destination! Final fitness: {car.fitness}")
                    # Could create new route or stop here
        
        # Handle case where car crashed or lost
        active_cars = [car for car in self.cars if not car.crashed]
        if len(active_cars) == 0 and len(self.cars) > 0:
            print("All cars crashed or lost paths. Attempting to create new car with valid path...")
            new_car = self.create_valid_route_car(max_attempts=5)
            if new_car:
                self.cars = [new_car]
                self.destination_mode_car = new_car
                self.validate_all_paths()  # Re-validate after creating new car
                print("New car created and paths validated")
            else:
                print("Failed to create replacement car with valid path")
        else:
            self.cars = active_cars

class SimulationData:
    """Class to manage and format simulation data for 3D visualization"""
    
    def __init__(self):
        self.cars_data = []
        self.roads_data = []
        self.camera_data = {}
    
    def update_cars(self, cars: List):
        """Convert car objects to 3D visualization data"""
        self.cars_data = []
        for car in cars:
            if car.crashed and not car.saved_state:
                continue
                
            # Generate upcoming segments with turn instructions
            upcoming_segments = self._generate_upcoming_segments(car, simulation_core.road_system if simulation_core else None)
            
            car_data = {
                'id': id(car),
                'position': {
                    'x': car.x / 20,  # Scale down more for better 3D viewing
                    'y': 0.2,  # Lower height above ground
                    'z': -car.y / 20  # Flip Y to Z for 3D (Y is up in 3D)
                },
                'rotation': {
                    'x': 0,
                    'y': math.radians(car.angle),  # Convert to radians for Three.js
                    'z': 0
                },
                'color': self._pygame_color_to_hex(car.color),
                'speed': car.speed,
                'fitness': car.fitness,
                'crashed': car.crashed,
                'saved_state': car.saved_state,
                'raycast_distances': car.raycast_distances[:],
                'raycast_angles': car.raycast_angles[:],
                'path_waypoints': [(x/20, -y/20) for x, y in car.path_waypoints] if car.path_waypoints else [],
                'current_waypoint_index': getattr(car, 'current_waypoint_index', 0),  # Add waypoint progress tracking
                'upcoming_segments': upcoming_segments  # Add real navigation data
            }
            self.cars_data.append(car_data)
    
    def update_roads(self, road_system):
        """Convert road system to 3D visualization data"""
        if not road_system or not hasattr(road_system, 'road_segments'):
            return
            
        self.roads_data = []
        for segment in road_system.road_segments:
            road_data = {
                'id': id(segment),
                'start': {
                    'x': segment.start[0] / 20,
                    'y': 0,
                    'z': -segment.start[1] / 20
                },
                'end': {
                    'x': segment.end[0] / 20,
                    'y': 0,
                    'z': -segment.end[1] / 20
                },
                'width': segment.width / 20,
                'lane_count': segment.lane_count,
                'oneway': segment.oneway
            }
            self.roads_data.append(road_data)
    
    def update_camera(self, camera):
        """Update camera data for 3D visualization"""
        if camera:
            self.camera_data = {
                'x': camera.x / 20,
                'y': 25,  # Lower height for 3D camera
                'z': -camera.y / 20,
                'zoom': camera.zoom
            }
    
    def _pygame_color_to_hex(self, pygame_color) -> str:
        """Convert pygame color tuple to hex string"""
        if isinstance(pygame_color, tuple) and len(pygame_color) >= 3:
            return f"#{pygame_color[0]:02x}{pygame_color[1]:02x}{pygame_color[2]:02x}"
        return "#ffffff"  # Default to white
    
    def _format_turn_instruction(self, ang_diff_deg: float) -> str:
        """Format turn instruction based on angle difference"""
        a = abs(ang_diff_deg)
        if a < 10: return "Continue straight"
        # Sign convention appears inverted; swap left/right
        if a < 25: return ("Slight right" if ang_diff_deg > 0 else "Slight left")
        if a < 55: return ("Turn right" if ang_diff_deg > 0 else "Turn left")
        if a < 120: return ("Sharp right" if ang_diff_deg > 0 else "Sharp left")
        return "U-turn right" if ang_diff_deg > 0 else "U-turn left"
    
    def _collect_upcoming_segments(self, car, road_system, max_items=7):
        """Collect upcoming road segments with street names for navigation"""
        names = []
        if not getattr(car, 'path_waypoints', None) or len(car.path_waypoints) < 2:
            return names
        start_idx = max(0, getattr(car, 'current_waypoint_index', 0))
        last_name = None
        dist_accum = 0.0
        for i in range(start_idx, len(car.path_waypoints)-1):
            x1, y1 = car.path_waypoints[i]
            x2, y2 = car.path_waypoints[i+1]
            seg_len = math.hypot(x2-x1, y2-y1)
            if seg_len <= 1e-6:
                continue
            midx = (x1 + x2) * 0.5
            midy = (y1 + y2) * 0.5
            seg_name = None
            try:
                candidates = []
                if hasattr(road_system, 'spatial_grid'):
                    candidates = road_system.spatial_grid.get_nearby_segments(midx, midy, radius=60)
                if not candidates and hasattr(road_system, 'road_segments'):
                    candidates = road_system.road_segments[:200]
                best_seg = None
                best_d2 = 1e12
                for seg in candidates:
                    sx, sy = seg.start; ex, ey = seg.end
                    dx = ex - sx; dy = ey
                    L2 = dx*dx + dy*dy
                    if L2 <= 1e-6:
                        continue
                    t = ((midx - sx)*dx + (midy - sy)*dy)/L2
                    t = 0 if t < 0 else (1 if t > 1 else t)
                    px = sx + dx*t; py = sy + dy*t
                    d2 = (px - midx)**2 + (py - midy)**2
                    if d2 < best_d2:
                        best_d2 = d2
                        best_seg = seg
                if best_seg is not None:
                    seg_name = getattr(best_seg, 'name', None) or getattr(best_seg, 'road_name', None)
            except Exception:
                seg_name = None
            if not seg_name:
                seg_name = f"Segment {i:03d}"
            dist_accum += seg_len
            if seg_name != last_name:
                names.append((seg_name, dist_accum))
                last_name = seg_name
            if len(names) >= max_items:
                break
        return names
    
    def _generate_upcoming_segments(self, car, road_system):
        """Generate upcoming navigation segments with turn instructions based on car's current progress"""
        segments = []
        if not getattr(car, 'path_waypoints', None) or len(car.path_waypoints) < 3:
            return segments
        
        # Get car's current position and waypoint progress
        current_wp_idx = max(0, getattr(car, 'current_waypoint_index', 0))
        car_x, car_y = car.x, car.y
        
        # Get upcoming road segments based on current progress
        upcoming_roads = self._collect_upcoming_segments(car, road_system, max_items=8)
        
        # Calculate segments starting from current position
        remaining_waypoints = car.path_waypoints[current_wp_idx:]
        if len(remaining_waypoints) < 3:
            return segments
        
        # Calculate distance to next waypoint for more accurate "in X distance" display
        if len(remaining_waypoints) > 0:
            next_wp_x, next_wp_y = remaining_waypoints[0]
            distance_to_next_wp = math.hypot(next_wp_x - car_x, next_wp_y - car_y)
        else:
            distance_to_next_wp = 0
        
        segment_idx = 0
        accumulated_distance = distance_to_next_wp  # Start with distance to next waypoint
        
        for i in range(len(remaining_waypoints) - 2):
            if segment_idx >= len(upcoming_roads) or len(segments) >= 4:
                break
                
            # Calculate turn angle between three consecutive waypoints
            p1 = remaining_waypoints[i]
            p2 = remaining_waypoints[i + 1]
            p3 = remaining_waypoints[i + 2]
            
            # Only add distance between waypoints for segments after the first
            if i > 0:
                segment_distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                accumulated_distance += segment_distance
            
            # Vector from p1 to p2
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            # Vector from p2 to p3
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate angle between vectors
            angle1 = math.atan2(v1[1], v1[0])
            angle2 = math.atan2(v2[1], v2[0])
            angle_diff = angle2 - angle1
            
            # Normalize angle difference to [-π, π]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Convert to degrees for turn instruction
            angle_diff_deg = math.degrees(angle_diff)
            
            # Only create a segment if there's a significant turn (more than 15 degrees)
            if abs(angle_diff_deg) > 15 or len(segments) == 0:  # Always include first segment
                # Get turn instruction
                instruction = self._format_turn_instruction(angle_diff_deg)
                
                # Get street name from road segments
                street_name = "Unknown Street"
                if segment_idx < len(upcoming_roads):
                    street_name = upcoming_roads[segment_idx][0]
                
                # Convert accumulated distance to appropriate units
                distance_m = accumulated_distance * 0.6  # Convert pixels to approximate meters
                
                # Format distance with more accurate representation
                if distance_m > 1000:
                    distance_str = f"{distance_m/1000:.1f} km"
                elif distance_m > 100:
                    distance_str = f"{int(distance_m)} m"
                else:
                    distance_str = f"{int(distance_m)} m"
                
                segments.append({
                    'name': street_name,
                    'street_name': street_name,
                    'distance': distance_m,
                    'distance_str': distance_str,
                    'action': instruction,
                    'turn_direction': instruction.lower(),
                    'angle_diff': angle_diff_deg,
                    'waypoint_index': current_wp_idx + i + 1  # Track which waypoint this relates to
                })
                
                segment_idx += 1
        
        return segments
    
    def update_from_simulation(self, simulation_core):
        """Update all data from the simulation core"""
        if simulation_core:
            self.update_cars(simulation_core.cars)
            self.update_roads(simulation_core.road_system)
            self.update_camera(simulation_core.camera)
    
    def to_dict(self):
        """Convert all data to dictionary for JSON serialization"""
        pathfinding_ok = True
        if simulation_core and simulation_core.cars:
            for car in simulation_core.cars:
                if not car.crashed and (not hasattr(car, 'path_waypoints') or not car.path_waypoints):
                    pathfinding_ok = False
                    break
        
        return {
            'cars': self.cars_data,
            'roads': self.roads_data,
            'camera': self.camera_data,
            'timestamp': time.time(),
            'generation': getattr(simulation_core, 'generation', 0) if simulation_core else 0,
            'mode': getattr(simulation_core, 'mode', 0) if simulation_core else 0,
            'pathfinding_ok': pathfinding_ok,
            'path_validation': {
                'paths_validated': getattr(simulation_core, 'paths_validated', False) if simulation_core else False,
                'valid_paths_count': getattr(simulation_core, 'valid_paths_count', 0) if simulation_core else 0,
                'required_paths_count': getattr(simulation_core, 'required_paths_count', 0) if simulation_core else 0
            }
        }

simulation_data = SimulationData()

# HTML template with Three.js 3D visualization
@app.route('/')
def index():
    with open('3dsim.html', 'r', encoding='utf-8') as file:
        html_template = file.read()
    return render_template_string(html_template)

async def websocket_handler(websocket, path):
    """Handle WebSocket connections"""
    print(f"New WebSocket client connected from {websocket.remote_address}")
    websocket_clients.add(websocket)
    print(f"Total WebSocket clients: {len(websocket_clients)}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                command = data.get('command')
                message_type = data.get('type')
                
                # Handle emergency stop messages
                if message_type == 'emergency_stop':
                    global emergency_stop_active
                    emergency_stop_active = data.get('active', False)
                    if emergency_stop_active:
                        print("EMERGENCY STOP ACTIVATED - Simulation paused")
                    else:
                        print("Emergency stop deactivated - Simulation resumed")
                
                elif command == 'pause':
                    if simulation_core:
                        simulation_core.running = False
                        print("Simulation paused")
                elif command == 'resume':
                    if simulation_core:
                        simulation_core.running = True
                        print("Simulation resumed")
                elif command == 'reset_generation':
                    if simulation_core and simulation_core.mode == 0:  # Training mode
                        print("Resetting generation...")
                        simulation_core.evolution_timer = simulation_core.evolution_interval
                elif command == 'change_mode':
                    if simulation_core:
                        new_mode = (simulation_core.mode + 1) % 3
                        simulation_core.mode = new_mode
                        simulation_core.initialize_mode()
                        print(f"Changed to mode {new_mode}")
                        
            except json.JSONDecodeError:
                print(f"Invalid JSON received: {message}")
            except Exception as e:
                print(f"Error handling WebSocket message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket client disconnected")
    finally:
        websocket_clients.remove(websocket)
        print(f"WebSocket client removed. Remaining clients: {len(websocket_clients)}")

async def broadcast_data():
    """Broadcast simulation data to all connected clients"""
    if websocket_clients and simulation_core and simulation_data:
        try:
            # Update visualization data from simulation
            simulation_data.update_from_simulation(simulation_core)
            data_dict = simulation_data.to_dict()
            
            # Debug logging
            if len(data_dict['cars']) == 0:
                print(f"No cars data to send. Simulation_core cars: {len(simulation_core.cars) if simulation_core.cars else 0}")
            elif len(data_dict['cars']) > 0:
                print(f"Broadcasting {len(data_dict['cars'])} cars and {len(data_dict['roads'])} roads")
            if len(data_dict['roads']) == 0:
                print(f"No roads data to send. Road system: {simulation_core.road_system is not None if simulation_core else 'No sim core'}")
            
            data = json.dumps(data_dict)
            
            # Create a copy of clients to avoid modification during iteration
            clients = websocket_clients.copy()
            for client in clients:
                try:
                    await client.send(data)
                except websockets.exceptions.ConnectionClosed:
                    websocket_clients.discard(client)
                except Exception as e:
                    print(f"Error sending data to client: {e}")
                    websocket_clients.discard(client)
        except Exception as e:
            print(f"Error in broadcast_data: {e}")
            import traceback
            traceback.print_exc()

def run_simulation():
    """Run the headless simulation core"""
    global simulation_core, simulation_running
    
    print("Starting 3D Car AI Simulation...")
    print("Open http://localhost:5000 in your browser to view the 3D visualization")
    
    # Initialize headless simulation
    simulation_core = HeadlessSimulationCore(
        width=1280,
        height=720,
        mode=1,  # Default to destination mode
        auto_mode=True,
        initial_scale=1.0
    )
    
    simulation_running = True
    
    # Simulation loop
    FPS = 60
    frame_duration = 1.0 / FPS
    
    try:
        while simulation_running:
            frame_start = time.time()
            
            # Update simulation only if running and not in emergency stop
            if simulation_core.running and not emergency_stop_active:
                simulation_core.update_simulation()
            elif emergency_stop_active:
                # During emergency stop, don't update car physics but keep everything else running
                pass
            
            # Control frame rate
            frame_end = time.time()
            frame_time = frame_end - frame_start
            if frame_time < frame_duration:
                time.sleep(frame_duration - frame_time)
                
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_running = False

def run_websocket_server():
    """Run the WebSocket server"""
    import websockets
    import asyncio
    
    # Set event loop policy for Windows threading
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Create new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def handle_client(websocket, path):
        await websocket_handler(websocket, path)
    
    async def periodic_broadcast():
        """Periodically broadcast data to clients"""
        print("Starting periodic broadcast loop...")
        broadcast_count = 0
        while True:  # Run indefinitely, check websocket_clients instead
            broadcast_count += 1
            if websocket_clients:
                print(f"Broadcasting to {len(websocket_clients)} clients... (#{broadcast_count})")
                await broadcast_data()
            elif broadcast_count % 50 == 0:  # Print every 5 seconds when no clients
                print("No WebSocket clients connected")
            await asyncio.sleep(1/30)  # 30 FPS data updates with 60fps interpolation
    
    async def main():
        # Start WebSocket server on all interfaces
        server = await websockets.serve(handle_client, "0.0.0.0", 8765)
        print("WebSocket server started on ws://localhost:8765")
        
        # Start periodic broadcasting
        broadcast_task = asyncio.create_task(periodic_broadcast())
        
        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            print("WebSocket server stopped")
        finally:
            broadcast_task.cancel()
            server.close()
    
    # Run the WebSocket server
    try:
        loop.run_until_complete(main())
    except Exception as e:
        print(f"WebSocket server error: {e}")
    finally:
        loop.close()

if __name__ == '__main__':
    print("Starting 3D Car AI Simulation...")
    print("Open http://localhost:5000 in your browser to view the 3D visualization")
    
    # Start simulation in a separate thread
    simulation_thread = threading.Thread(target=run_simulation)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    # Start WebSocket server in a separate thread
    websocket_thread = threading.Thread(target=run_websocket_server)
    websocket_thread.daemon = True
    websocket_thread.start()
    
    # Give WebSocket server time to start
    time.sleep(2)
    
    # Run Flask app
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        simulation_running = False
        print("Shutting down 3D visualization...")
