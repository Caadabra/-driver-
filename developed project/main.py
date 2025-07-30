import pygame
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
import os
from collections import deque
from osm_roads import OSMRoadSystem
from camera import Camera
from dijkstra_pathfinding import DijkstraPathfinder

pygame.init()

population_size = 30

WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Cars Learning on Real Roads")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

clock = pygame.time.Clock()
FPS = 60  # Display FPS
SIMULATION_SPEED = 5  # Simulation runs 5x faster than display

# Global variables for tracking statistics
fitness_history = deque(maxlen=100)  # Keep last 100 generations
generation_history = deque(maxlen=100)
saved_cars_history = deque(maxlen=100)

# Raycast configuration - Simple setup
NUM_RAYCASTS = 5
RAYCAST_SPREAD = 90  # 90 degree spread for front sensors

def set_raycast_config(num_rays, spread_degrees=90):
    """
    Simple raycast configuration.
    """
    global NUM_RAYCASTS, RAYCAST_SPREAD
    NUM_RAYCASTS = num_rays
    RAYCAST_SPREAD = spread_degrees
    print(f"Raycast config: {NUM_RAYCASTS} rays with {RAYCAST_SPREAD}° spread")

class SimpleCarAI(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, output_size=3):
        super(SimpleCarAI, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        return self.network(x)

class Car:
    def __init__(self, x, y, color, use_ai=False, road_system=None, pathfinder=None):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.color = color
        self.width = 12/1.4
        self.height = 20/1.4
        
        # Simple raycast setup - 5 sensors
        self.raycast_length = 100
        self.raycast_angles = [-45, -22.5, 0, 22.5, 45]  # Left, front-left, front, front-right, right
        self.raycast_distances = [self.raycast_length] * 5
        
        self.use_ai = use_ai
        self.fitness = 0
        self.time_alive = 0
        self.distance_traveled = 0
        self.crashed = False
        self.road_system = road_system
        self.pathfinder = pathfinder
        self.last_valid_position = (x, y)
        self.off_road_time = 0
        self.max_off_road_time = 180  # 3 seconds at 60 FPS
        
        # Simple stationary detection
        self.stationary_timer = 0
        self.stationary_threshold = 300  # 5 seconds
        self.last_position = (x, y)
        self.position_check_timer = 0
        
        # Time-to-checkpoint tracking for enhanced reward system
        self.checkpoint_times = []  # Track time taken to reach each checkpoint
        self.last_checkpoint_time = 0  # When the last checkpoint was reached
        self.current_checkpoint_start_time = 0  # When we started moving toward current checkpoint
        
        if self.use_ai:
            # Enhanced AI with pathfinding: 5 raycasts + speed + angle + target_direction + target_distance
            self.ai = SimpleCarAI(input_size=9, hidden_size=64, output_size=3)
            self.randomize_weights()
            
        # AI control visualization
        self.ai_acceleration = 0
        self.ai_steering = 0
        self.ai_brake = 0
        
        # Pathfinding system
        self.current_path = []  # List of node IDs
        self.path_waypoints = []  # List of (x, y) waypoints
        self.current_waypoint_index = 0
        self.waypoint_reach_distance = 30
        self.target_x = None
        self.target_y = None
        self.destination_node = None
        self.pathfinding_timer = 0
        self.pathfinding_interval = 300  # Recalculate path every 5 seconds
        self.destination_reached = False  # Track if destination was reached
        self.saved_state = False  # Track if car is in saved state
        self.shared_destination = None  # All cars share the same destination
        
        # Don't generate path immediately - wait for shared destination to be set
    
    def generate_path_to_destination(self, destination_pos):
        """Generate a path to the shared destination using Dijkstra's algorithm"""
        if not self.pathfinder or not destination_pos:
            print("No pathfinder or destination available")
            return
        
        # Find path from current position to the shared destination
        path = self.pathfinder.find_path(self.x, self.y, destination_pos[0], destination_pos[1])
        
        if path:
            self.current_path = path
            self.path_waypoints = self.pathfinder.path_to_waypoints(path)
            self.shared_destination = destination_pos
            self.current_waypoint_index = 0
            self.update_current_target()
            print(f"Car generated individual path with {len(path)} waypoints to shared destination {destination_pos}")
        else:
            print(f"No path found to shared destination at {destination_pos}. Trying fallback...")
            # Fallback: try to find a path to a nearby reachable point
            self.try_fallback_destination()
    
    def try_fallback_destination(self):
        """Try to find an alternative destination if the shared one is unreachable"""
        if not self.pathfinder:
            return
        
        max_attempts = 5
        for attempt in range(max_attempts):
            # Try to find a random reachable destination
            fallback_node, fallback_pos = self.pathfinder.get_random_destination()
            if fallback_node:
                path = self.pathfinder.find_path(self.x, self.y, fallback_pos[0], fallback_pos[1])
                if path:
                    self.current_path = path
                    self.path_waypoints = self.pathfinder.path_to_waypoints(path)
                    self.current_waypoint_index = 0
                    self.update_current_target()
                    print(f"Car using fallback destination at {fallback_pos}")
                    return
        
        print("No fallback destination found - car will wander")
    
    def generate_new_path_to_shared_destination(self):
        """Generate a new path to the shared destination"""
        if not self.pathfinder or not self.shared_destination:
            return
        
        # Find path from current position to shared destination
        path = self.pathfinder.find_path(self.x, self.y, self.shared_destination[0], self.shared_destination[1])
        
        if path:
            self.current_path = path
            self.path_waypoints = self.pathfinder.path_to_waypoints(path)
            self.current_waypoint_index = 0
            self.update_current_target()
            print(f"Car generated new individual path with {len(path)} waypoints to shared destination")
        else:
            print("No path found to shared destination, trying fallback...")
            self.try_fallback_destination()
    
    def set_route(self, waypoints):
        """Set a route for the car to follow (for compatibility)"""
        self.path_waypoints = waypoints.copy()
        self.current_waypoint_index = 0
        self.update_current_target()
    
    def update_current_target(self):
        """Update the current target waypoint"""
        if self.current_waypoint_index < len(self.path_waypoints):
            self.target_x, self.target_y = self.path_waypoints[self.current_waypoint_index]
            # Reset checkpoint timer when starting toward new waypoint
            self.current_checkpoint_start_time = self.time_alive
        else:
            # Path completed - destination reached!
            if not self.destination_reached:
                self.destination_reached = True
                self.saved_state = True
                print(f"Car reached shared destination! Entering saved state with bonus fitness.")
                # Give massive bonus for reaching destination
                self.fitness += 1000
                # Generate a new path to the same shared destination for continuous pathfinding
                if self.shared_destination:
                    self.generate_new_path_to_shared_destination()
            else:
                self.target_x = None
                self.target_y = None
    
    def check_waypoint_reached(self):
        """Check if current waypoint is reached and advance to next"""
        if self.target_x is None or self.target_y is None:
            return
        
        distance_to_target = math.sqrt((self.x - self.target_x)**2 + (self.y - self.target_y)**2)
        if distance_to_target < self.waypoint_reach_distance:
            # Calculate time taken to reach this checkpoint
            time_to_checkpoint = self.time_alive - self.current_checkpoint_start_time
            self.checkpoint_times.append(time_to_checkpoint)
            self.last_checkpoint_time = self.time_alive
            
            # Advance to next waypoint
            self.current_waypoint_index += 1
            self.update_current_target()
            
            # Bonus fitness for reaching waypoints quickly
            if self.use_ai:
                base_waypoint_bonus = 50
                # Time efficiency bonus - reward faster checkpoint completion
                if time_to_checkpoint > 0:
                    # Ideal time per checkpoint (adjust based on your game)
                    ideal_time = 180  # 3 seconds at 60 FPS
                    time_efficiency = max(0.1, ideal_time / time_to_checkpoint)  # Better if faster
                    time_bonus = min(base_waypoint_bonus * time_efficiency, 200)  # Cap at 200
                    self.fitness += time_bonus
                    print(f"Checkpoint reached in {time_to_checkpoint} frames, bonus: {time_bonus:.1f}")
                else:
                    self.fitness += base_waypoint_bonus
    
    def get_target_direction(self):
        """Get the direction to the current target as an angle difference"""
        if self.target_x is None or self.target_y is None:
            return 0
        
        # Calculate angle to target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        target_angle = math.degrees(math.atan2(dx, -dy))
        
        # Calculate difference from current angle
        angle_diff = target_angle - self.angle
        
        # Normalize to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        return angle_diff
    
    def get_target_distance(self):
        """Get the distance to the current target waypoint"""
        if self.target_x is None or self.target_y is None:
            return 0
        
        return math.sqrt((self.x - self.target_x)**2 + (self.y - self.target_y)**2)
        
    def draw(self, camera, is_best=False):
        screen_x, screen_y = camera.world_to_screen(self.x, self.y)
        
        if (screen_x < -50 or screen_x > camera.screen_width + 50 or 
            screen_y < -50 or screen_y > camera.screen_height + 50):
            return
        
        car_surface = pygame.Surface((self.width * camera.zoom, self.height * camera.zoom), pygame.SRCALPHA)
        
        if self.saved_state:
            # Saved cars get a special golden color
            car_surface.fill((255, 215, 0))  # Gold for saved cars
        elif is_best:
            car_surface.fill((255, 0, 0))  # Red for best car
        else:
            car_surface.fill(self.color)
        
        rotated_surface = pygame.transform.rotate(car_surface, -self.angle)
        rotated_rect = rotated_surface.get_rect(center=(screen_x, screen_y))
        screen.blit(rotated_surface, rotated_rect.topleft)
        
        # Draw special effect around saved cars
        if self.saved_state:
            # Draw a glowing effect around saved cars
            pygame.draw.circle(screen, (255, 255, 0), (screen_x, screen_y), int(15 * camera.zoom), 2)
        
        # Draw raycasts for best car
        if is_best:
            self.draw_raycasts(camera)
            self.draw_route(camera)
    
    def draw_route(self, camera):
        """Draw the pathfinding route waypoints and current target"""
        if not self.path_waypoints:
            return
        
        # Draw all waypoints in the path
        for i, (wx, wy) in enumerate(self.path_waypoints):
            screen_pos = camera.world_to_screen(wx, wy)
            if i == self.current_waypoint_index:
                # Current target - larger, bright circle
                pygame.draw.circle(screen, (0, 255, 255), screen_pos, 12)  # Cyan for current target
                pygame.draw.circle(screen, (255, 255, 255), screen_pos, 12, 3)  # White border
            elif i == len(self.path_waypoints) - 1:
                # Final destination - special marker for shared destination
                pygame.draw.circle(screen, (255, 100, 100), screen_pos, 12)  # Red-orange for shared destination
                pygame.draw.circle(screen, (255, 255, 255), screen_pos, 12, 3)  # White border
                # Add destination text
                font = pygame.font.Font(None, 20)
                dest_text = "SHARED"
                text_surface = font.render(dest_text, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(screen_pos[0], screen_pos[1] - 20))
                screen.blit(text_surface, text_rect)
            else:
                # Other waypoints - smaller, blue circles
                pygame.draw.circle(screen, (0, 100, 200), screen_pos, 6)  # Blue
                pygame.draw.circle(screen, (255, 255, 255), screen_pos, 6, 2)  # White border
        
        # Draw lines connecting waypoints (the calculated path)
        if len(self.path_waypoints) > 1:
            screen_points = [camera.world_to_screen(wx, wy) for wx, wy in self.path_waypoints]
            for i in range(len(screen_points) - 1):
                # Color code the path: green for completed, yellow for current, white for future
                if i < self.current_waypoint_index:
                    color = (0, 150, 0)  # Green for completed segments
                elif i == self.current_waypoint_index:
                    color = (255, 255, 0)  # Yellow for current segment
                else:
                    color = (150, 150, 150)  # Gray for future segments
                pygame.draw.line(screen, color, screen_points[i], screen_points[i + 1], 3)
        
        # Draw line to current target
        if self.target_x is not None and self.target_y is not None:
            car_screen = camera.world_to_screen(self.x, self.y)
            target_screen = camera.world_to_screen(self.target_x, self.target_y)
            pygame.draw.line(screen, (0, 255, 255), car_screen, target_screen, 2)  # Cyan line to target
        screen_x, screen_y = camera.world_to_screen(self.x, self.y)
        
        if (screen_x < -50 or screen_x > camera.screen_width + 50 or 
            screen_y < -50 or screen_y > camera.screen_height + 50):
            return
        
        car_surface = pygame.Surface((self.width * camera.zoom, self.height * camera.zoom), pygame.SRCALPHA)
        
        if is_best:
            car_surface.fill((255, 0, 0))  # Red for best car
        else:
            car_surface.fill(self.color)
        
        rotated_surface = pygame.transform.rotate(car_surface, -self.angle)
        rotated_rect = rotated_surface.get_rect(center=(screen_x, screen_y))
        screen.blit(rotated_surface, rotated_rect.topleft)
        
        # Draw raycasts for best car
        if is_best:
            self.draw_raycasts(camera)
    
    def move(self, keys):
        # Skip movement if in saved state (destination reached)
        if self.saved_state:
            return
            
        if self.use_ai:
            self.ai_move()
        else:
            # Manual controls
            if keys[pygame.K_UP]:
                self.speed += 0.2
            if keys[pygame.K_DOWN]:
                self.speed -= 0.3
            # Only allow turning if car has some speed
            if abs(self.speed) > 0.1:
                if keys[pygame.K_LEFT]:
                    self.angle -= 5
                if keys[pygame.K_RIGHT]:
                    self.angle += 5

        # Update position
        prev_x, prev_y = self.x, self.y
        self.x += math.sin(math.radians(self.angle)) * self.speed
        self.y -= math.cos(math.radians(self.angle)) * self.speed
        
        # Check waypoint progress for pathfinding
        if self.use_ai and self.path_waypoints and not self.saved_state:
            self.check_waypoint_reached()
            
            # Periodically recalculate path to ensure optimal route
            self.pathfinding_timer += 1
            if self.pathfinding_timer >= self.pathfinding_interval:
                self.pathfinding_timer = 0
                if self.pathfinder and not self.destination_reached and self.shared_destination:
                    # Only recalculate if we're not close to destination
                    if len(self.path_waypoints) - self.current_waypoint_index > 3:
                        self.generate_new_path_to_shared_destination()
        
        # Track distance for AI
        if self.use_ai:
            distance = math.sqrt((self.x - prev_x)**2 + (self.y - prev_y)**2)
            self.distance_traveled += distance
            self.time_alive += 1

        # Apply friction
        self.speed *= 0.95
        if self.speed < 0:
            self.speed = 0

        # Check if on road - crash immediately if off road (hitting green)
        # Skip this check if in saved state
        if self.road_system and not self.saved_state:
            if self.road_system.is_point_on_road(self.x, self.y):
                self.last_valid_position = (self.x, self.y)
                self.off_road_time = 0
            else:
                # Immediate crash when hitting green (off-road)
                self.crashed = True
                self.off_road_time += 1

        # Check if stationary (for AI cars) - skip if in saved state
        if self.use_ai and not self.saved_state:
            self.position_check_timer += 1
            if self.position_check_timer >= 60:  # Check every second
                distance_moved = math.sqrt((self.x - self.last_position[0])**2 + (self.y - self.last_position[1])**2)
                if distance_moved < 10:  # Less than 10 pixels moved in 1 second
                    self.stationary_timer += 60
                else:
                    self.stationary_timer = 0
                
                if self.stationary_timer >= self.stationary_threshold:
                    self.crashed = True
                    
                self.last_position = (self.x, self.y)
                self.position_check_timer = 0

    def update_raycasts(self):
        if not self.road_system:
            self.raycast_distances = [self.raycast_length] * 5
            return
        
        self.raycast_distances = []
        for angle_offset in self.raycast_angles:
            ray_angle = self.angle + angle_offset
            distance = self.road_system.raycast_to_road_edge(
                self.x, self.y, ray_angle, self.raycast_length
            )
            self.raycast_distances.append(distance)
    
    def draw_raycasts(self, camera):
        for i, angle_offset in enumerate(self.raycast_angles):
            if i < len(self.raycast_distances):
                distance = self.raycast_distances[i]
                ray_angle = math.radians(self.angle + angle_offset)
                
                end_x = self.x + math.sin(ray_angle) * distance
                end_y = self.y - math.cos(ray_angle) * distance
                
                start_screen = camera.world_to_screen(self.x, self.y)
                end_screen = camera.world_to_screen(end_x, end_y)
                pygame.draw.line(screen, YELLOW, start_screen, end_screen, 1)
                pygame.draw.circle(screen, RED, end_screen, 3)

    def randomize_weights(self):
        for param in self.ai.parameters():
            param.data.uniform_(-1, 1)
    
    def ai_move(self):
        if len(self.raycast_distances) < 5:
            return
        
        # Prepare enhanced inputs for AI with route following
        inputs = []
        
        # Normalize raycast distances (5 values)
        normalized_distances = [d / self.raycast_length for d in self.raycast_distances]
        inputs.extend(normalized_distances)
        
        # Add speed (normalized)
        normalized_speed = min(1.0, abs(self.speed) / 5.0)
        inputs.append(normalized_speed)
        
        # Add current angle (normalized to -1 to 1)
        normalized_angle = math.sin(math.radians(self.angle))
        inputs.append(normalized_angle)
        
        # Add route following information
        target_direction = self.get_target_direction() / 180.0  # Normalize to [-1, 1]
        inputs.append(target_direction)
        
        target_distance = min(1.0, self.get_target_distance() / 200.0)  # Normalize to [0, 1]
        inputs.append(target_distance)
        
        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.ai(input_tensor)
        
        # AI outputs: acceleration, steering, brake
        acceleration = outputs[0].item()
        steering = outputs[1].item()
        brake = outputs[2].item()
        
        # Store AI decisions for visualization
        self.ai_acceleration = acceleration
        self.ai_steering = steering
        self.ai_brake = brake
        
        # Apply gradual acceleration and braking (mutually exclusive)
        # Determine which action to take based on which signal is stronger
        if abs(brake) > abs(acceleration) and brake > 0:
            # Braking takes priority - gradual deceleration based on brake intensity
            self.speed -= brake * 0.4  # Brake strength
        elif acceleration > 0:
            # Acceleration - gradual speed change based on acceleration value
            self.speed += acceleration * 0.25  # Acceleration strength
        elif acceleration < 0:
            # Negative acceleration acts as light braking
            self.speed += acceleration * 0.2  # Slower deceleration than brake
        
        # Prevent going backwards
        if self.speed < 0:
            self.speed = 0
            
        # Cap maximum speed to prevent runaway acceleration
        max_speed = 4.0
        if self.speed > max_speed:
            self.speed = max_speed
            
        # Only apply steering if car has some speed (realistic car physics)
        if abs(self.speed) > 0.1:
            self.angle += steering * 4.0

    def calculate_fitness(self):
        # Enhanced fitness calculation with route following
        base_fitness = 0
        
        # Time alive bonus
        time_bonus = self.time_alive * 0.1
        
        # Distance traveled bonus
        distance_bonus = self.distance_traveled * 0.5
        
        # Road staying bonus - big bonus for never going off road
        road_bonus = 0
        if self.off_road_time == 0:
            road_bonus = self.time_alive * 0.3  # Bigger bonus for staying on road
        
        # Enhanced speed bonus system - reward good speed management
        avg_speed = self.distance_traveled / max(1, self.time_alive)
        speed_bonus = 0
        
        # Time-to-checkpoint efficiency reward system
        checkpoint_efficiency_bonus = 0
        if self.checkpoint_times:
            # Calculate average time per checkpoint
            avg_checkpoint_time = sum(self.checkpoint_times) / len(self.checkpoint_times)
            
            # Reward faster checkpoint completion
            ideal_checkpoint_time = 180  # 3 seconds at 60 FPS (adjust as needed)
            if avg_checkpoint_time > 0:
                time_efficiency = ideal_checkpoint_time / avg_checkpoint_time
                # Bonus increases exponentially for faster times, but capped
                checkpoint_efficiency_bonus = min(time_efficiency * 100, 300)
                
                # Additional bonus for consistent checkpoint times (low variance)
                if len(self.checkpoint_times) > 1:
                    time_variance = sum(abs(t - avg_checkpoint_time) for t in self.checkpoint_times) / len(self.checkpoint_times)
                    consistency_bonus = max(0, (60 - time_variance) * 2)  # Reward consistent timing
                    checkpoint_efficiency_bonus += consistency_bonus
        
        # Current checkpoint progress bonus - reward making progress toward current target
        current_checkpoint_bonus = 0
        if self.target_x is not None and self.target_y is not None:
            time_on_current = self.time_alive - self.current_checkpoint_start_time
            if time_on_current > 0:
                # Bonus decreases over time to encourage faster completion
                urgency_multiplier = max(0.1, 1.0 - (time_on_current / 600))  # Decrease over 10 seconds
                distance_to_target = self.get_target_distance()
                proximity_bonus = max(0, (100 - distance_to_target) / 100) * 50 * urgency_multiplier
                current_checkpoint_bonus = proximity_bonus
        
        # Combine all checkpoint-related bonuses
        speed_bonus = checkpoint_efficiency_bonus + current_checkpoint_bonus
        
        # Cap total speed bonus
        speed_bonus = min(speed_bonus, 400)
        
        # Pathfinding bonus - reward following calculated paths
        pathfinding_bonus = 0
        if self.path_waypoints:
            # Bonus for progressing through waypoints
            waypoint_progress = self.current_waypoint_index / max(1, len(self.path_waypoints))
            pathfinding_bonus += waypoint_progress * 300  # Big bonus for following path
            
            # Bonus for being close to current target
            if self.target_x is not None and self.target_y is not None:
                target_distance = self.get_target_distance()
                proximity_bonus = max(0, (100 - target_distance) / 100) * 100
                pathfinding_bonus += proximity_bonus
                
        # Massive bonus for reaching destination (saved state)
        destination_bonus = 0
        if self.destination_reached or self.saved_state:
            destination_bonus = 10000  # Huge bonus for reaching destination + time bonus
            if self.time_alive < 600:
                destination_bonus += (600 - self.time_alive) * 10

        # Penalties
        crash_penalty = 300 if self.crashed else 0  # Higher crash penalty
        off_road_penalty = self.off_road_time * 5  # Higher off-road penalty
        stationary_penalty = self.stationary_timer * 1
        
        self.fitness = (
            time_bonus + distance_bonus + road_bonus + speed_bonus + pathfinding_bonus + destination_bonus -
            crash_penalty - off_road_penalty - stationary_penalty
        )
        
        return self.fitness
    
    def clone(self):
        new_car = Car(self.x, self.y, self.color, use_ai=True, road_system=self.road_system, pathfinder=self.pathfinder)
        if self.use_ai:
            new_car.ai.load_state_dict(self.ai.state_dict())
        return new_car
    
    def mutate(self, mutation_rate=0.15, mutation_strength=0.3):
        if not self.use_ai:
            return
        
        with torch.no_grad():
            for param in self.ai.parameters():
                mutation_mask = torch.rand_like(param) < mutation_rate
                mutation = torch.randn_like(param) * mutation_strength
                param.data += mutation_mask.float() * mutation

def save_population(cars, generation, filename="population.pkl"):
    """Save the current population to a file"""
    try:
        # Create trained_models directory if it doesn't exist
        os.makedirs("trained_models", exist_ok=True)
        
        # Prepare data to save
        population_data = {
            'generation': generation,
            'cars': []
        }
        
        # Save each car's AI state
        for car in cars:
            if car.use_ai:
                car_data = {
                    'fitness': car.fitness,
                    'ai_state': car.ai.state_dict(),
                    'saved_state': car.saved_state,
                    'destination_reached': car.destination_reached
                }
                population_data['cars'].append(car_data)
        
        # Save to file
        filepath = os.path.join("trained_models", filename)
        with open(filepath, 'wb') as f:
            pickle.dump(population_data, f)
        
        print(f"Population saved to {filepath} - Generation {generation}, {len(population_data['cars'])} cars")
        return True
    except Exception as e:
        print(f"Error saving population: {e}")
        return False

def load_population(filename="population.pkl"):
    """Load a population from a file"""
    try:
        filepath = os.path.join("trained_models", filename)
        if not os.path.exists(filepath):
            print(f"No saved population found at {filepath}")
            return None, 0
        
        with open(filepath, 'rb') as f:
            population_data = pickle.load(f)
        
        print(f"Population loaded from {filepath} - Generation {population_data['generation']}, {len(population_data['cars'])} cars")
        return population_data, population_data['generation']
    except Exception as e:
        print(f"Error loading population: {e}")
        return None, 0

def create_car_from_data(car_data, road_system, pathfinder):
    """Create a new car from saved data"""
    spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
    color = random.choice([RED, GREEN, BLUE])
    car = Car(spawn_x, spawn_y, color, use_ai=True, road_system=road_system, pathfinder=pathfinder)
    car.angle = spawn_angle
    
    # Load AI state
    if car.use_ai and 'ai_state' in car_data:
        car.ai.load_state_dict(car_data['ai_state'])
    
    return car


def evolve_population(cars, population_size=population_size, road_system=None, pathfinder=None, shared_destination=None, generation=1):
    # Calculate fitness for all cars
    for car in cars:
        car.calculate_fitness()
    
    # Sort by fitness (best first), prioritizing saved cars
    cars.sort(key=lambda x: (x.saved_state, x.fitness), reverse=True)
    saved_count = sum(1 for car in cars if car.saved_state)
    print(f"Generation evolved! Best fitness: {cars[0].fitness:.2f} | Saved cars: {saved_count}")
    
    # Track statistics for graphs
    fitness_history.append(cars[0].fitness)
    generation_history.append(generation)
    saved_cars_history.append(saved_count)
    
    # Generate new shared destination if needed or if many cars reached the current one
    new_destination_needed = (shared_destination is None or saved_count >= population_size // 3)
    
    if new_destination_needed and pathfinder:
        max_attempts = 10
        valid_destination = None
        
        for attempt in range(max_attempts):
            destination_node, test_destination = pathfinder.get_random_destination()
            if destination_node and test_destination:
                # Test if at least one spawn point can reach this destination
                test_spawn_x, test_spawn_y, _ = road_system.get_random_spawn_point()
                test_path = pathfinder.find_path(test_spawn_x, test_spawn_y, test_destination[0], test_destination[1])
                
                if test_path:
                    valid_destination = test_destination
                    print(f"New valid shared destination set at: {valid_destination} (attempt {attempt + 1})")
                    break
                else:
                    print(f"Destination at {test_destination} is unreachable, trying again... (attempt {attempt + 1})")
        
        if valid_destination:
            shared_destination = valid_destination
        else:
            print("WARNING: Could not find a reachable shared destination after 10 attempts!")
            # Keep the old destination if we can't find a new one
    
    # Keep top performers as elites, ensuring saved cars are prioritized
    elite_count = max(1, population_size // 5)
    elites = cars[:elite_count]
    new_cars = []
    
    # Clone elites (reset their state)
    for elite in elites:
        new_car = elite.clone()
        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
        new_car.x = spawn_x
        new_car.y = spawn_y
        new_car.angle = spawn_angle
        new_car.speed = 0
        new_car.crashed = False
        new_car.fitness = 0
        new_car.time_alive = 0
        new_car.distance_traveled = 0
        new_car.off_road_time = 0
        new_car.stationary_timer = 0
        new_car.last_position = (spawn_x, spawn_y)
        new_car.position_check_timer = 0
        new_car.checkpoint_times = []  # Reset checkpoint times
        new_car.last_checkpoint_time = 0  # Reset checkpoint timing
        new_car.current_checkpoint_start_time = 0  # Reset checkpoint start time
        new_car.pathfinder = pathfinder  # Ensure pathfinder is set
        new_car.destination_reached = False  # Reset destination status
        new_car.saved_state = False  # Reset saved state
        new_car.shared_destination = shared_destination  # Set shared destination
        if shared_destination:
            new_car.generate_path_to_destination(shared_destination)  # Generate individual path to shared destination
        new_cars.append(new_car)
    
    # Create offspring with mutations
    while len(new_cars) < population_size:
        # Select parent from top half
        parent_index = random.randint(0, min(len(cars) - 1, population_size // 2 - 1))
        parent = cars[parent_index]
        offspring = parent.clone()
        
        # Apply mutation (stronger for worse performers)
        base_mutation_rate = 0.12
        base_mutation_strength = 0.25
        if parent.fitness > cars[0].fitness * 0.8:
            offspring.mutate(mutation_rate=base_mutation_rate * 0.7, mutation_strength=base_mutation_strength * 0.8)
        else:
            offspring.mutate(mutation_rate=base_mutation_rate * 1.3, mutation_strength=base_mutation_strength * 1.2)
        
        # Reset offspring state
        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
        offspring.x = spawn_x
        offspring.y = spawn_y
        offspring.angle = spawn_angle
        offspring.speed = 0
        offspring.crashed = False
        offspring.fitness = 0
        offspring.time_alive = 0
        offspring.distance_traveled = 0
        offspring.off_road_time = 0
        offspring.stationary_timer = 0
        offspring.last_position = (spawn_x, spawn_y)
        offspring.position_check_timer = 0
        offspring.checkpoint_times = []  # Reset checkpoint times
        offspring.last_checkpoint_time = 0  # Reset checkpoint timing
        offspring.current_checkpoint_start_time = 0  # Reset checkpoint start time
        offspring.color = random.choice([RED, GREEN, BLUE])
        offspring.pathfinder = pathfinder  # Ensure pathfinder is set
        offspring.destination_reached = False  # Reset destination status
        offspring.saved_state = False  # Reset saved state
        offspring.shared_destination = shared_destination  # Set shared destination
        if shared_destination:
            offspring.generate_path_to_destination(shared_destination)  # Generate individual path to shared destination
        new_cars.append(offspring)
    
    # Auto-save the population after evolution
    save_population(new_cars, generation)
    
    return new_cars, shared_destination

def draw_ai_controls(screen, car, x, y):
    """Draw AI control visualization showing what the AI is 'pressing'"""
    if not car or not car.use_ai:
        return
    
    # Colors
    inactive_color = (100, 100, 100)  # Gray
    active_color = (255, 255, 0)      # Yellow
    brake_color = (255, 0, 0)         # Red
    
    # Arrow key size and spacing
    key_size = 40
    spacing = 10
    
    # Calculate colors based on AI outputs (mutually exclusive)
    # Determine which action is being taken based on priority
    if abs(car.ai_brake) > abs(car.ai_acceleration) and car.ai_brake > 0.1:
        # Braking takes priority
        up_color = inactive_color
        down_color = brake_color
    elif car.ai_acceleration > 0.1:
        # Accelerating
        up_color = active_color
        down_color = inactive_color
    elif car.ai_acceleration < -0.1:
        # Negative acceleration (light braking)
        up_color = inactive_color
        down_color = (255, 100, 0)  # Orange for light braking
    else:
        # Neither accelerating nor braking significantly
        up_color = inactive_color
        down_color = inactive_color
    
    left_color = active_color if car.ai_steering < -0.1 else inactive_color
    right_color = active_color if car.ai_steering > 0.1 else inactive_color
    
    # Draw arrow keys layout
    # Up arrow (acceleration)
    up_points = [
        (x + key_size//2, y),
        (x, y + key_size//2),
        (x + key_size//4, y + key_size//2),
        (x + key_size//4, y + key_size),
        (x + 3*key_size//4, y + key_size),
        (x + 3*key_size//4, y + key_size//2),
        (x + key_size, y + key_size//2)
    ]
    pygame.draw.polygon(screen, up_color, up_points)
    pygame.draw.polygon(screen, WHITE, up_points, 2)
    
    # Down arrow (brake)
    down_y = y + key_size + spacing
    down_points = [
        (x + key_size//2, down_y + key_size),
        (x, down_y + key_size//2),
        (x + key_size//4, down_y + key_size//2),
        (x + key_size//4, down_y),
        (x + 3*key_size//4, down_y),
        (x + 3*key_size//4, down_y + key_size//2),
        (x + key_size, down_y + key_size//2)
    ]
    pygame.draw.polygon(screen, down_color, down_points)
    pygame.draw.polygon(screen, WHITE, down_points, 2)
    
    # Left arrow (steer left)
    left_x = x - key_size - spacing
    left_y = y + key_size//2 + spacing//2
    left_points = [
        (left_x, left_y + key_size//2),
        (left_x + key_size//2, left_y),
        (left_x + key_size//2, left_y + key_size//4),
        (left_x + key_size, left_y + key_size//4),
        (left_x + key_size, left_y + 3*key_size//4),
        (left_x + key_size//2, left_y + 3*key_size//4),
        (left_x + key_size//2, left_y + key_size)
    ]
    pygame.draw.polygon(screen, left_color, left_points)
    pygame.draw.polygon(screen, WHITE, left_points, 2)
    
    # Right arrow (steer right)
    right_x = x + key_size + spacing
    right_y = y + key_size//2 + spacing//2
    right_points = [
        (right_x + key_size, right_y + key_size//2),
        (right_x + key_size//2, right_y),
        (right_x + key_size//2, right_y + key_size//4),
        (right_x, right_y + key_size//4),
        (right_x, right_y + 3*key_size//4),
        (right_x + key_size//2, right_y + 3*key_size//4),
        (right_x + key_size//2, right_y + key_size)
    ]
    pygame.draw.polygon(screen, right_color, right_points)
    pygame.draw.polygon(screen, WHITE, right_points, 2)
    
    # Add labels
    font = pygame.font.Font(None, 24)
    
    # Acceleration value
    if car.ai_acceleration > 0.01:
        acc_text = f"ACC: {car.ai_acceleration:.2f}"
        acc_surface = font.render(acc_text, True, WHITE)
        screen.blit(acc_surface, (x - 20, y - 30))
    
    # Brake value
    if car.ai_brake > 0.01:
        brake_text = f"BRAKE: {car.ai_brake:.2f}"
        brake_surface = font.render(brake_text, True, WHITE)
        screen.blit(brake_surface, (x - 20, down_y + key_size + 10))
    
    # Steering value
    if abs(car.ai_steering) > 0.01:
        steer_text = f"STEER: {car.ai_steering:.2f}"
        steer_surface = font.render(steer_text, True, WHITE)
        screen.blit(steer_surface, (x - 40, y + key_size*2 + spacing + 20))
    
    # Speed display
    speed_text = f"SPEED: {car.speed:.2f}"
    speed_surface = font.render(speed_text, True, WHITE)
    screen.blit(speed_surface, (x - 20, y - 60))

def draw_neural_network(screen, car, x, y, width=200, height=150):
    """Draw a visualization of the neural network in the top right corner"""
    if not car or not car.use_ai:
        return
    
    # Background
    nn_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(screen, (20, 20, 20, 180), nn_rect)
    pygame.draw.rect(screen, WHITE, nn_rect, 2)
    
    # Network architecture: input(9) -> hidden(64) -> hidden(64) -> output(3)
    input_size = 9
    hidden_size = 64
    output_size = 3
    
    # Layer positions
    input_x = x + 20
    hidden1_x = x + width * 0.4
    hidden2_x = x + width * 0.6
    output_x = x + width - 40
    
    # Node spacing
    input_spacing = height / (input_size + 1)
    hidden_spacing = height / 9  # Show 8 nodes + dots indicator
    output_spacing = height / (output_size + 1)
    
    # Get current network activations
    try:
        with torch.no_grad():
            # Prepare inputs (same as in ai_move)
            inputs = []
            
            # Raycast distances
            normalized_distances = [d / car.raycast_length for d in car.raycast_distances]
            inputs.extend(normalized_distances)
            
            # Speed
            normalized_speed = min(1.0, abs(car.speed) / 5.0)
            inputs.append(normalized_speed)
            
            # Angle
            normalized_angle = math.sin(math.radians(car.angle))
            inputs.append(normalized_angle)
            
            # Target direction and distance
            target_direction = car.get_target_direction() / 180.0
            inputs.append(target_direction)
            
            target_distance = min(1.0, car.get_target_distance() / 200.0)
            inputs.append(target_distance)
            
            input_tensor = torch.tensor(inputs, dtype=torch.float32)
            
            # Get layer outputs
            layer1 = car.ai.network[0](input_tensor)  # First linear layer
            layer1_activated = car.ai.network[1](layer1)  # ReLU
            layer2 = car.ai.network[2](layer1_activated)  # Second linear layer  
            layer2_activated = car.ai.network[3](layer2)  # ReLU
            outputs = car.ai.network[4](layer2_activated)  # Output layer
            final_outputs = car.ai.network[5](outputs)  # Tanh
            
    except Exception as e:
        # If we can't get activations, just draw the structure
        inputs = [0.5] * input_size
        layer1_activated = torch.zeros(hidden_size)
        layer2_activated = torch.zeros(hidden_size)
        final_outputs = torch.zeros(output_size)
    
    # Draw input layer
    input_labels = ["Ray1", "Ray2", "Ray3", "Ray4", "Ray5", "Speed", "Angle", "TargetDir", "TargetDist"]
    for i in range(input_size):
        node_y = y + (i + 1) * input_spacing
        activation = inputs[i] if i < len(inputs) else 0
        
        # Color based on activation
        intensity = max(0, min(255, int(abs(activation) * 255)))
        color = (intensity, intensity, intensity) if activation >= 0 else (intensity, 0, 0)
        
        pygame.draw.circle(screen, color, (input_x, int(node_y)), 6)
        pygame.draw.circle(screen, WHITE, (input_x, int(node_y)), 6, 1)
        
        # Label
        if i < len(input_labels):
            font = pygame.font.Font(None, 16)
            label = font.render(input_labels[i], True, WHITE)
            screen.blit(label, (input_x - 30, int(node_y) - 8))
    
    # Draw hidden layers (show subset of nodes)
    for layer_idx, (layer_x, activations) in enumerate([(hidden1_x, layer1_activated), (hidden2_x, layer2_activated)]):
        for i in range(min(8, len(activations))):  # Show first 8 nodes to represent 64
            node_y = y + (i + 1) * hidden_spacing
            activation = activations[i].item() if i < len(activations) else 0
            
            # Color based on activation
            intensity = max(0, min(255, int(abs(activation) * 255)))
            color = (0, intensity, 0) if activation >= 0 else (intensity, 0, 0)
            
            pygame.draw.circle(screen, color, (int(layer_x), int(node_y)), 4)
            pygame.draw.circle(screen, WHITE, (int(layer_x), int(node_y)), 4, 1)
        
        # Add "..." indicator to show there are more nodes
        if len(activations) > 8:
            dots_y = y + (9) * hidden_spacing
            font = pygame.font.Font(None, 16)
            dots_text = font.render("...", True, WHITE)
            text_rect = dots_text.get_rect(center=(int(layer_x), int(dots_y)))
            screen.blit(dots_text, text_rect)
    
    # Draw output layer
    output_labels = ["Accel", "Steer", "Brake"]
    for i in range(output_size):
        node_y = y + (i + 1) * output_spacing
        activation = final_outputs[i].item() if i < len(final_outputs) else 0
        
        # Color based on activation
        intensity = max(0, min(255, int(abs(activation) * 255)))
        color = (0, 0, intensity) if activation >= 0 else (intensity, 0, 0)
        
        pygame.draw.circle(screen, color, (output_x, int(node_y)), 8)
        pygame.draw.circle(screen, WHITE, (output_x, int(node_y)), 8, 1)
        
        # Label and value
        font = pygame.font.Font(None, 16)
        label = font.render(f"{output_labels[i]}: {activation:.2f}", True, WHITE)
        screen.blit(label, (output_x + 15, int(node_y) - 8))
    
    # Draw connections (simplified - show representative connections)
    # Input to first hidden layer - show connections from each input to a few hidden nodes
    for i in range(input_size):
        start_y = y + (i + 1) * input_spacing
        for j in range(min(3, 8)):  # Connect to first 3 visible hidden nodes
            end_y = y + (j + 1) * hidden_spacing
            # Use a subtle gray color for connections
            pygame.draw.line(screen, (60, 60, 60), 
                           (input_x + 6, int(start_y)), 
                           (int(hidden1_x) - 4, int(end_y)), 1)
    
    # First hidden layer to second hidden layer
    for i in range(min(5, 8)):  # Show connections from first 5 visible nodes
        start_y = y + (i + 1) * hidden_spacing
        for j in range(min(5, 8)):  # To first 5 visible nodes in second layer
            end_y = y + (j + 1) * hidden_spacing
            pygame.draw.line(screen, (60, 60, 60), 
                           (int(hidden1_x) + 4, int(start_y)), 
                           (int(hidden2_x) - 4, int(end_y)), 1)
    
    # Second hidden layer to output
    for i in range(min(5, 8)):  # Show connections from first 5 visible hidden nodes
        start_y = y + (i + 1) * hidden_spacing
        for j in range(output_size):  # To all output nodes
            end_y = y + (j + 1) * output_spacing
            pygame.draw.line(screen, (60, 60, 60), 
                           (int(hidden2_x) + 4, int(start_y)), 
                           (output_x - 8, int(end_y)), 1)
    
    # Title
    title_font = pygame.font.Font(None, 24)
    title = title_font.render("Neural Network", True, WHITE)
    screen.blit(title, (x + 5, y - 25))
    
    # Checkpoint timing information display
    timing_info_font = pygame.font.Font(None, 16)
    
    if hasattr(car, 'checkpoint_times') and len(car.checkpoint_times) > 0:
        avg_checkpoint_time = sum(car.checkpoint_times) / len(car.checkpoint_times)
        avg_time_text = timing_info_font.render(f"Avg Checkpoint Time: {avg_checkpoint_time:.1f}f", True, WHITE)
        screen.blit(avg_time_text, (x + 5, y + height + 25))
        
        last_checkpoint_text = timing_info_font.render(f"Last Checkpoint: {car.checkpoint_times[-1]:.1f}f", True, WHITE)
        screen.blit(last_checkpoint_text, (x + 5, y + height + 40))
    
    # Current checkpoint progress
    if car.target_x is not None and car.target_y is not None:
        time_on_current = car.time_alive - car.current_checkpoint_start_time
        current_progress_text = timing_info_font.render(f"Current Checkpoint: {time_on_current}f", True, WHITE)
        screen.blit(current_progress_text, (x + 5, y + height + 55))
    
    # Layer labels
    label_font = pygame.font.Font(None, 14)
    input_label = label_font.render("Input", True, WHITE)
    hidden1_label = label_font.render("Hidden1", True, WHITE)
    hidden2_label = label_font.render("Hidden2", True, WHITE)
    output_label = label_font.render("Output", True, WHITE)
    
    screen.blit(input_label, (input_x - 20, y + height + 5))
    screen.blit(hidden1_label, (int(hidden1_x) - 25, y + height + 5))
    screen.blit(hidden2_label, (int(hidden2_x) - 25, y + height + 5))
    screen.blit(output_label, (output_x - 20, y + height + 5))

def draw_performance_graph(screen, x, y, width=300, height=200):
    """Draw a performance graph in the bottom left corner"""
    if len(fitness_history) == 0:
        return
    
    # Background
    graph_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(screen, (20, 20, 20, 180), graph_rect)
    pygame.draw.rect(screen, WHITE, graph_rect, 2)
    
    # Title
    title_font = pygame.font.Font(None, 24)
    title = title_font.render("Performance Over Time", True, WHITE)
    screen.blit(title, (x + 5, y - 25))
    
    if len(fitness_history) < 2:
        return
    
    # Graph area
    graph_x = x + 20
    graph_y = y + 20
    graph_width = width - 40
    graph_height = height - 40
    
    # Find min/max values for scaling
    max_fitness = max(fitness_history)
    min_fitness = min(fitness_history)
    max_saved = max(saved_cars_history) if saved_cars_history else 1
    
    # Avoid division by zero
    if max_fitness == min_fitness:
        max_fitness = min_fitness + 1
    
    # Ensure max_saved is not zero to avoid division by zero
    if max_saved == 0:
        max_saved = 1
    
    # Draw grid lines
    for i in range(5):
        y_pos = graph_y + (i * graph_height // 4)
        pygame.draw.line(screen, (60, 60, 60), 
                        (graph_x, y_pos), 
                        (graph_x + graph_width, y_pos), 1)
    
    # Draw fitness line (blue)
    fitness_points = []
    for i, fitness in enumerate(fitness_history):
        x_pos = graph_x + (i * graph_width // max(1, len(fitness_history) - 1))
        y_pos = graph_y + graph_height - int(((fitness - min_fitness) / (max_fitness - min_fitness)) * graph_height)
        fitness_points.append((x_pos, y_pos))
    
    if len(fitness_points) > 1:
        pygame.draw.lines(screen, BLUE, False, fitness_points, 2)
    
    # Draw saved cars line (green)
    if saved_cars_history:
        saved_points = []
        for i, saved_count in enumerate(saved_cars_history):
            x_pos = graph_x + (i * graph_width // max(1, len(saved_cars_history) - 1))
            y_pos = graph_y + graph_height - int((saved_count / max_saved) * graph_height)
            saved_points.append((x_pos, y_pos))
        
        if len(saved_points) > 1:
            pygame.draw.lines(screen, GREEN, False, saved_points, 2)
    
    # Legend
    legend_font = pygame.font.Font(None, 18)
    
    # Fitness legend
    fitness_text = legend_font.render("Fitness", True, BLUE)
    screen.blit(fitness_text, (graph_x, graph_y + graph_height + 5))
    
    # Saved cars legend
    saved_text = legend_font.render("Saved Cars", True, GREEN)
    screen.blit(saved_text, (graph_x + 80, graph_y + graph_height + 5))
    
    # Current values
    current_fitness = fitness_history[-1] if fitness_history else 0
    current_saved = saved_cars_history[-1] if saved_cars_history else 0
    current_gen = generation_history[-1] if generation_history else 0
    
    values_text = legend_font.render(f"Gen: {current_gen} | Fitness: {current_fitness:.1f} | Saved: {current_saved}", True, WHITE)
    screen.blit(values_text, (graph_x, graph_y + graph_height + 25))

# --- Setup ---
print("Loading road system...")
road_system = OSMRoadSystem(center_lat=-36.8825825, center_lon=174.9143453, radius=1000)

print("Setting up pathfinding...")
pathfinder = DijkstraPathfinder(road_system)

camera = Camera(0, 0, WIDTH, HEIGHT)
road_bounds = road_system.get_road_bounds()
camera.set_bounds(*road_bounds)

cars = []

# Create initial population with shared destination pathfinding
print("Creating initial population with shared destination pathfinding...")

# Try to load existing population first
population_data, loaded_generation = load_population()

if population_data is not None:
    print(f"Loading saved population from generation {loaded_generation}...")
    generation = loaded_generation
    
    # Create cars from saved data
    cars = []
    for car_data in population_data['cars']:
        car = create_car_from_data(car_data, road_system, pathfinder)
        cars.append(car)
    
    # Fill up to population size if needed
    while len(cars) < population_size:
        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
        color = random.choice([RED, GREEN, BLUE])
        car = Car(spawn_x, spawn_y, color, use_ai=True, road_system=road_system, pathfinder=pathfinder)
        car.angle = spawn_angle
        cars.append(car)
    
    print(f"Loaded {len(population_data['cars'])} cars, created {len(cars)} total")
else:
    print("No saved population found, creating new population...")
    generation = 1
    cars = []
    
    # Create new population
    for i in range(population_size):
        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
        color = random.choice([RED, GREEN, BLUE])
        car = Car(spawn_x, spawn_y, color, use_ai=True, road_system=road_system, pathfinder=pathfinder)
        car.angle = spawn_angle
        cars.append(car)

# Generate the first shared destination with validation
shared_destination = None
if pathfinder:
    print("Finding initial reachable shared destination...")
    max_attempts = 10
    
    for attempt in range(max_attempts):
        destination_node, test_destination = pathfinder.get_random_destination()
        if destination_node and test_destination:
            # Test if a spawn point can reach this destination
            test_spawn_x, test_spawn_y, _ = road_system.get_random_spawn_point()
            test_path = pathfinder.find_path(test_spawn_x, test_spawn_y, test_destination[0], test_destination[1])
            
            if test_path:
                shared_destination = test_destination
                print(f"Initial valid shared destination set at: {shared_destination}")
                break
            else:
                print(f"Destination at {test_destination} is unreachable, trying again... (attempt {attempt + 1})")
    
    if not shared_destination:
        print("WARNING: Could not find a reachable initial destination!")

# Set up pathfinding for all cars
for car in cars:
    car.shared_destination = shared_destination
    if shared_destination:
        car.generate_path_to_destination(shared_destination)  # Generate individual path to shared destination

evolution_timer = 0
evolution_interval = 20 * FPS  # 20 seconds per generation (in display time)
simulation_steps = 0  # Track total simulation steps

print("Starting AI cars with Dijkstra pathfinding evolution...")
print(f"Simulation running at {SIMULATION_SPEED}x speed")
print("Controls:")
print("  + - Increase simulation speed")
print("  - - Decrease simulation speed")
print("  SPACE - Toggle pause")

paused = False

running = True
while running:
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("Saving population before exit...")
            save_population(cars, generation)
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                print("Resetting to new location...")
                # Load new random location
                road_system = OSMRoadSystem(
                    center_lat=random.uniform(-40, 50), 
                    center_lon=random.uniform(-120, 120), 
                    radius=1000
                )
                # Rebuild pathfinder for new location
                pathfinder = DijkstraPathfinder(road_system)
                road_bounds = road_system.get_road_bounds()
                camera.set_bounds(*road_bounds)
                
                # Reset all cars to new random positions with individual paths to shared destination
                # Generate new valid shared destination for new location
                new_shared_destination = None
                if pathfinder:
                    for attempt in range(10):
                        destination_node, test_destination = pathfinder.get_random_destination()
                        if destination_node and test_destination:
                            test_spawn_x, test_spawn_y, _ = road_system.get_random_spawn_point()
                            test_path = pathfinder.find_path(test_spawn_x, test_spawn_y, test_destination[0], test_destination[1])
                            if test_path:
                                new_shared_destination = test_destination
                                print(f"New location shared destination set at: {new_shared_destination}")
                                break
                
                shared_destination = new_shared_destination
                
                for car in cars:
                    if not car.crashed:
                        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
                        car.x, car.y, car.angle = spawn_x, spawn_y, spawn_angle
                        car.road_system = road_system
                        car.pathfinder = pathfinder
                        car.shared_destination = shared_destination
                        if shared_destination:
                            car.generate_path_to_destination(shared_destination)  # Generate individual path to shared destination
            elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:  # + key
                SIMULATION_SPEED = min(20, SIMULATION_SPEED + 1)
                print(f"Simulation speed: {SIMULATION_SPEED}x")
            elif event.key == pygame.K_MINUS:  # - key
                SIMULATION_SPEED = max(1, SIMULATION_SPEED - 1)
                print(f"Simulation speed: {SIMULATION_SPEED}x")
            elif event.key == pygame.K_SPACE:
                paused = not paused
                print(f"Simulation {'paused' if paused else 'resumed'}")
    
    keys = pygame.key.get_pressed()
    
    # Run simulation steps (multiple per frame for speed)
    if not paused:
        for sim_step in range(SIMULATION_SPEED):
            simulation_steps += 1
            
            # Find best car and count alive cars (including saved cars)
            alive_cars = 0
            saved_cars = 0
            best_car = None
            for car in cars:
                if not car.crashed or car.saved_state:
                    if simulation_steps % 30 == 0:  # Update fitness occasionally
                        car.calculate_fitness()
                    if best_car is None or car.fitness > best_car.fitness:
                        best_car = car
                    alive_cars += 1
                    if car.saved_state:
                        saved_cars += 1
            
            # Update cars (simulation step)
            for car in cars:
                if not car.crashed or car.saved_state:  # Process saved cars too for display
                    if not car.saved_state:  # Only move cars that aren't saved
                        car.update_raycasts()
                        car.move(keys)
                        
                        # Check if car reached destination
                        if (car.target_x is None and car.target_y is None and 
                            len(car.path_waypoints) > 0 and 
                            car.current_waypoint_index >= len(car.path_waypoints) and
                            not car.destination_reached):
                            car.destination_reached = True
                            car.saved_state = True
                            print(f"Car reached destination! Total time: {car.time_alive} frames")
            
            # Evolution logic
            evolution_timer += 1
            if evolution_timer >= evolution_interval or alive_cars == 0:
                cars, shared_destination = evolve_population(cars, population_size, road_system, pathfinder, shared_destination, generation)
                evolution_timer = 0
                generation += 1
                print(f"Generation {generation} started")
                break  # Exit simulation loop to render
    
    # Rendering (once per display frame)
    screen.fill((34, 139, 34))  # Dark green background
    
    # Find best car for display (recalculate for rendering)
    best_car = None
    alive_cars = 0
    saved_cars = 0
    for car in cars:
        if not car.crashed or car.saved_state:
            if best_car is None or car.fitness > best_car.fitness:
                best_car = car
            alive_cars += 1
            if car.saved_state:
                saved_cars += 1
    
    # Camera follows best car
    if best_car and not camera.manual_mode:
        camera.follow_target(best_car.x, best_car.y)
    camera.update(keys)
    
    # Draw roads
    road_system.draw_roads(screen, camera.x, camera.y, WIDTH, HEIGHT, camera.zoom)
    
    # Draw cars (rendering only)
    for car in cars:
        if not car.crashed or car.saved_state:  # Draw saved cars too
            is_best = (car == best_car)
            car.draw(camera, is_best)
    
    # Display info with simulation speed and saved cars count
    font = pygame.font.Font(None, 36)
    info_text = f"Gen: {generation} | Alive: {alive_cars} | Saved: {saved_cars} | Speed: {SIMULATION_SPEED}x"
    if paused:
        info_text += " | PAUSED"
    else:
        info_text += f" | Time: {evolution_timer // FPS}s"
    if best_car:
        info_text += f" | Best: {best_car.fitness:.1f}"
        if best_car.saved_state:
            info_text += " (SAVED!)"
    
    text_surface = font.render(info_text, True, WHITE)
    text_rect = text_surface.get_rect()
    text_rect.topleft = (10, 10)
    
    pygame.draw.rect(screen, (0, 0, 0, 128), text_rect.inflate(10, 5))
    screen.blit(text_surface, text_rect)
    
    # Display pathfinding info with performance stats and controls
    pathfinding_info = f"Shared Destination: All cars find individual paths to the same goal!"
    pathfinding_surface = font.render(pathfinding_info, True, WHITE)
    pathfinding_rect = pathfinding_surface.get_rect()
    pathfinding_rect.topleft = (10, text_rect.bottom + 10)
    
    pygame.draw.rect(screen, (0, 0, 0, 128), pathfinding_rect.inflate(10, 5))
    screen.blit(pathfinding_surface, pathfinding_rect)
    
    # Display controls
    controls_font = pygame.font.Font(None, 24)
    controls_text = "Controls: +/- Speed | SPACE Pause | R Reset"
    controls_surface = controls_font.render(controls_text, True, WHITE)
    controls_rect = controls_surface.get_rect()
    controls_rect.topleft = (10, pathfinding_rect.bottom + 5)
    
    pygame.draw.rect(screen, (0, 0, 0, 128), controls_rect.inflate(10, 5))
    screen.blit(controls_surface, controls_rect)
    
    # Draw AI control visualization for best car
    if best_car:
        # Position in bottom right corner
        control_x = WIDTH - 120
        control_y = HEIGHT - 150
        draw_ai_controls(screen, best_car, control_x, control_y)
        
        # Draw neural network visualization in top right corner
        nn_x = WIDTH - 220
        nn_y = 50
        draw_neural_network(screen, best_car, nn_x, nn_y)
    
    # Draw performance graph in bottom left corner
    graph_x = 20
    graph_y = HEIGHT - 250
    draw_performance_graph(screen, graph_x, graph_y)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
