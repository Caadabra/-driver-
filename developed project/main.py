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
pygame.display.set_caption("{driver}")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

clock = pygame.time.Clock()
FPS = 60

# Waypoint spacing configuration (can be tuned without editing method bodies)
WAYPOINT_SPACING_DEFAULT = 150  # pixels between resampled checkpoints
WAYPOINT_MIN_SPACING = 100
WAYPOINT_MAX_SPACING = 220
TARGET_WAYPOINT_COUNT = 25  # Adaptive goal for long paths

def compute_adaptive_waypoint_spacing(total_length):
    """Derive spacing based on total path length aiming for TARGET_WAYPOINT_COUNT.
    Clamped between min/max spacing. Falls back to default for short paths."""
    if total_length <= 0:
        return WAYPOINT_SPACING_DEFAULT
    raw = total_length / TARGET_WAYPOINT_COUNT
    if raw < WAYPOINT_MIN_SPACING:
        return WAYPOINT_MIN_SPACING
    if raw > WAYPOINT_MAX_SPACING:
        return WAYPOINT_MAX_SPACING
    return raw

# Global variables for tracking statistics
fitness_history = deque(maxlen=100)  # Keep last 100 generations
generation_history = deque(maxlen=100)
saved_cars_history = deque(maxlen=100)

# Raycast configuration - Enhanced setup with more sensors
NUM_RAYCASTS = 8
RAYCAST_SPREAD = 180  # 180 degree spread for comprehensive sensing

def set_raycast_config(num_rays, spread_degrees=180):
    """
    Enhanced raycast configuration with more sensors.
    """
    global NUM_RAYCASTS, RAYCAST_SPREAD
    NUM_RAYCASTS = num_rays
    RAYCAST_SPREAD = spread_degrees
    print(f"Raycast config: {NUM_RAYCASTS} rays with {RAYCAST_SPREAD}° spread")

class SimpleCarAI(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, output_size=3):
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
        
        # Enhanced raycast setup - 8 sensors for better awareness
        self.raycast_length = 100
        self.raycast_angles = [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5]  # Full 180-degree coverage
        self.raycast_distances = [self.raycast_length] * 8
        
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
        
        # Enhanced checkpoint tracking with streak rewards
        self.checkpoint_times = []  # Track time taken to reach each checkpoint
        self.last_checkpoint_time = 0  # When the last checkpoint was reached
        self.current_checkpoint_start_time = 0  # When we started moving toward current checkpoint
        self.checkpoint_streak = 0  # Current streak of successful checkpoints
        self.max_checkpoint_streak = 0  # Best streak achieved
        self.consecutive_fast_checkpoints = 0  # Streak of fast checkpoint completions
        
        if self.use_ai:
            # Enhanced AI with more raycasts: 8 raycasts + speed + angle + target_direction + target_distance
            self.ai = SimpleCarAI(input_size=17, hidden_size=64, output_size=3)
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
        self.individual_destination = None  # Each car has its own unique destination

        # Predictive path system for 3-second lookahead
        self.prediction_horizon = 180  # 3 seconds at 60 FPS
        self.predicted_path = []  # List of (x, y) positions for next 3 seconds (OPTIMAL path)
        self.ai_predicted_path = []  # List of (x, y) positions showing what AI will actually do
        self.path_following_accuracy = 0.0  # How well following predicted path
        self.path_deviation_history = deque(maxlen=60)  # Track recent deviations

        # Progress validation system to prevent undeserved rewards
        self.initial_destination_distance = None  # Distance to destination when path was generated
        self.last_destination_distance = None  # Track if we're actually getting closer
        self.progress_validation_timer = 0  # Check progress periodically
        self.path_generation_time = 0  # When the current path was generated
        self.valid_checkpoints_only = True  # Only award checkpoints for real progress

        # Generate individual destination and path immediately
        if self.pathfinder:
            self.generate_individual_destination()

    def _resample_waypoints_evenly(self, spacing=None):
        """Resample current path_waypoints so checkpoints are evenly spaced.
        If spacing not provided, compute adaptively from total path length."""
        if not self.path_waypoints or len(self.path_waypoints) < 2:
            return
        pts = self.path_waypoints
        # Compute cumulative distances & cache geometry fundamentals
        dists = [0.0]
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i-1][0]
            dy = pts[i][1] - pts[i-1][1]
            dists.append(dists[-1] + math.hypot(dx, dy))
        total = dists[-1]
        if spacing is None:
            spacing = compute_adaptive_waypoint_spacing(total)
        if total < spacing * 0.5:  # Too short to resample
            # Still build geometry cache for original points
            self._rebuild_path_geometry()
            return
        new_pts = [pts[0]]
        target = spacing
        seg_index = 1
        while target < total:
            while seg_index < len(dists) and dists[seg_index] < target:
                seg_index += 1
            if seg_index >= len(dists):
                break
            prev_idx = seg_index - 1
            seg_len = dists[seg_index] - dists[prev_idx]
            if seg_len == 0:
                target += spacing
                continue
            t = (target - dists[prev_idx]) / seg_len
            x = pts[prev_idx][0] + (pts[seg_index][0] - pts[prev_idx][0]) * t
            y = pts[prev_idx][1] + (pts[seg_index][1] - pts[prev_idx][1]) * t
            new_pts.append((x, y))
            target += spacing
        if new_pts[-1] != pts[-1]:
            new_pts.append(pts[-1])
        self.path_waypoints = new_pts
        self._rebuild_path_geometry()

    def _rebuild_path_geometry(self):
        """Cache geometry for current path_waypoints: segment vectors, lengths, cumulative lengths, headings."""
        self.path_segment_vectors = []  # (dx, dy)
        self.path_segment_lengths = []
        self.path_segment_headings = []  # radians
        self.path_cumulative_lengths = [0.0]
        pts = self.path_waypoints
        if not pts or len(pts) < 2:
            return
        total = 0.0
        for i in range(len(pts) - 1):
            dx = pts[i+1][0] - pts[i][0]
            dy = pts[i+1][1] - pts[i][1]
            seg_len = math.hypot(dx, dy)
            self.path_segment_vectors.append((dx, dy))
            self.path_segment_lengths.append(seg_len)
            heading = math.atan2(dx, -dy)  # Consistent with existing angle usage
            self.path_segment_headings.append(heading)
            total += seg_len
            self.path_cumulative_lengths.append(total)
        self.path_total_length = total
    
    def generate_individual_destination(self):
        """Generate a unique destination for this car"""
        if not self.pathfinder:
            return
        
        # Try to find a random reachable destination
        max_attempts = 10
        for attempt in range(max_attempts):
            destination_node, destination_pos = self.pathfinder.get_random_destination()
            if destination_node:
                # Make sure it's not too close to starting position
                distance_to_dest = math.sqrt(
                    (self.x - destination_pos[0])**2 + (self.y - destination_pos[1])**2
                )
                if distance_to_dest > 300:  # Minimum 300 pixels away
                    path = self.pathfinder.find_path(self.x, self.y, destination_pos[0], destination_pos[1])
                    if path:
                        self.current_path = path
                        self.path_waypoints = self.pathfinder.path_to_waypoints(path)
                        self._resample_waypoints_evenly()
                        self.individual_destination = destination_pos
                        self.current_waypoint_index = 0
                        
                        # Initialize progress validation
                        self.initial_destination_distance = distance_to_dest
                        self.last_destination_distance = self.initial_destination_distance
                        self.path_generation_time = self.time_alive
                        
                        self.update_current_target()
                        print(f"Car generated individual destination at {destination_pos} (distance: {distance_to_dest:.1f})")
                        return
        
        print("Failed to generate individual destination - car will use fallback")
        self.try_fallback_destination()
    
    def generate_path_to_destination(self, destination_pos):
        """Generate a path to a specific destination using Dijkstra's algorithm"""
        if not self.pathfinder or not destination_pos:
            print("No pathfinder or destination available")
            return
        
        # Find path from current position to the destination
        path = self.pathfinder.find_path(self.x, self.y, destination_pos[0], destination_pos[1])
        
        if path:
            self.current_path = path
            self.path_waypoints = self.pathfinder.path_to_waypoints(path)
            self._resample_waypoints_evenly()
            self.individual_destination = destination_pos
            self.current_waypoint_index = 0
            
            # Initialize progress validation
            self.initial_destination_distance = math.sqrt(
                (self.x - destination_pos[0])**2 + (self.y - destination_pos[1])**2
            )
            self.last_destination_distance = self.initial_destination_distance
            self.path_generation_time = self.time_alive
            
            self.update_current_target()
            print(f"Car generated individual path with {len(path)} waypoints to destination {destination_pos}")
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
                    self._resample_waypoints_evenly()
                    self.current_waypoint_index = 0
                    self.update_current_target()
                    print(f"Car using fallback destination at {fallback_pos}")
                    return
        
        print("No fallback destination found - car will wander")
    
    def generate_new_path_to_individual_destination(self):
        """Generate a new path to the individual destination"""
        if not self.pathfinder or not self.individual_destination:
            # If no individual destination, generate a new one
            self.generate_individual_destination()
            return
        
        # Find path from current position to individual destination
        path = self.pathfinder.find_path(self.x, self.y, self.individual_destination[0], self.individual_destination[1])
        
        if path:
            self.current_path = path
            self.path_waypoints = self.pathfinder.path_to_waypoints(path)
            self._resample_waypoints_evenly()
            self.current_waypoint_index = 0
            
            # Update progress validation for new path
            if self.individual_destination:
                current_destination_distance = math.sqrt(
                    (self.x - self.individual_destination[0])**2 + (self.y - self.individual_destination[1])**2
                )
                self.last_destination_distance = current_destination_distance
                self.path_generation_time = self.time_alive
            
            self.update_current_target()
            print(f"Car generated new individual path with {len(path)} waypoints to destination")
        else:
            print("No path found to individual destination, generating new destination...")
            self.generate_individual_destination()
    
    def set_route(self, waypoints):
        """Set a route for the car to follow (for compatibility)"""
        self.path_waypoints = waypoints.copy()
        self._resample_waypoints_evenly()
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
                print(f"Car reached individual destination! Entering saved state with bonus fitness.")
                # Give massive bonus for reaching destination
                self.fitness += 1000
                # Generate a new individual destination for continuous learning
                self.generate_individual_destination()
            else:
                self.target_x = None
                self.target_y = None
    
    def check_waypoint_reached(self):
        """Check if current waypoint is reached and advance to next"""
        if self.target_x is None or self.target_y is None:
            return
        
        distance_to_target = math.sqrt((self.x - self.target_x)**2 + (self.y - self.target_y)**2)
        if distance_to_target < self.waypoint_reach_distance:
            # VALIDATION: Only award checkpoint bonus if actually making progress toward destination
            is_valid_checkpoint = self.validate_checkpoint_progress()
            
            # Calculate time taken to reach this checkpoint
            time_to_checkpoint = self.time_alive - self.current_checkpoint_start_time
            self.checkpoint_times.append(time_to_checkpoint)
            self.last_checkpoint_time = self.time_alive
            
            # Update checkpoint streaks for valid checkpoints
            if is_valid_checkpoint:
                self.checkpoint_streak += 1
                self.max_checkpoint_streak = max(self.max_checkpoint_streak, self.checkpoint_streak)
                
                # Check if this was a fast checkpoint completion
                ideal_time = 180  # 3 seconds at 60 FPS
                if time_to_checkpoint <= ideal_time * 1.2:  # Within 20% of ideal time
                    self.consecutive_fast_checkpoints += 1
                else:
                    self.consecutive_fast_checkpoints = 0  # Reset fast streak
            else:
                # Invalid checkpoint - reset streaks but no harsh penalty
                self.checkpoint_streak = 0
                self.consecutive_fast_checkpoints = 0
            
            # Advance to next waypoint
            self.current_waypoint_index += 1
            self.update_current_target()
            
            # Enhanced bonus fitness for reaching waypoints (ONLY for valid checkpoints)
            if self.use_ai and is_valid_checkpoint:
                base_waypoint_bonus = 50
                
                # Time efficiency bonus - reward faster checkpoint completion
                if time_to_checkpoint > 0:
                    time_efficiency = max(0.1, ideal_time / time_to_checkpoint)  # Better if faster
                    time_bonus = min(base_waypoint_bonus * time_efficiency, 200)  # Cap at 200
                    
                    # Streak multiplier bonus - reward consecutive checkpoints
                    streak_multiplier = 1.0 + (self.checkpoint_streak * 0.1)  # 10% bonus per checkpoint in streak
                    streak_multiplier = min(streak_multiplier, 3.0)  # Cap at 3x multiplier
                    
                    # Fast checkpoint streak bonus
                    fast_streak_bonus = self.consecutive_fast_checkpoints * 25  # 25 points per fast checkpoint in streak
                    
                    total_bonus = (time_bonus * streak_multiplier) + fast_streak_bonus
                    self.fitness += total_bonus
                    
                    print(f"Checkpoint {self.checkpoint_streak} reached in {time_to_checkpoint} frames, "
                          f"bonus: {total_bonus:.1f} (streak: {streak_multiplier:.1f}x, fast streak: {self.consecutive_fast_checkpoints})")
                else:
                    self.fitness += base_waypoint_bonus
    
    def validate_checkpoint_progress(self):
        """Validate that reaching this checkpoint represents real progress toward the destination"""
        if not self.individual_destination:
            return True  # No destination to validate against
        
        # Calculate current distance to individual destination
        current_destination_distance = math.sqrt(
            (self.x - self.individual_destination[0])**2 + (self.y - self.individual_destination[1])**2
        )
        
        # Validation criteria:
        # 1. Must be closer to destination than when path was generated (or very recently)
        time_since_path_gen = self.time_alive - self.path_generation_time
        if time_since_path_gen < 120:  # Within 2 seconds of path generation, be lenient
            min_required_progress = 0.95  # Allow 5% tolerance for new paths
        else:
            min_required_progress = 0.98  # Require actual progress for older paths
        
        # 2. Must be making overall progress (closer than initial distance)
        if self.initial_destination_distance:
            progress_ratio = current_destination_distance / self.initial_destination_distance
            making_overall_progress = progress_ratio <= min_required_progress
        else:
            making_overall_progress = True
        
        # 3. Must be closer than the last time we checked (or similar)
        if self.last_destination_distance:
            recent_progress_ratio = current_destination_distance / self.last_destination_distance
            making_recent_progress = recent_progress_ratio <= 1.02  # Allow 2% tolerance for local detours
        else:
            making_recent_progress = True
        
        # Update tracking
        self.last_destination_distance = current_destination_distance
        
        # Checkpoint is valid if making both overall and recent progress
        is_valid = making_overall_progress and making_recent_progress
        
        if not is_valid:
            print(f"Invalid checkpoint: overall_progress={progress_ratio:.3f}, recent_progress={recent_progress_ratio:.3f}")
        
        return is_valid
    
    def get_target_direction(self):
        """Get the direction to the current target as an angle difference"""
        if self.target_x is None or self.target_y is None:
            return 0
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        target_angle = math.degrees(math.atan2(dx, -dy))
        angle_diff = target_angle - self.angle
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        return angle_diff

    def get_target_distance(self):
        """Return distance in pixels to current target waypoint or 0 if no target."""
        if self.target_x is None or self.target_y is None:
            return 0
        return math.hypot(self.target_x - self.x, self.target_y - self.y)

    def calculate_predictive_path(self):
        """Calculate where the car should be in the next 3 seconds based on optimal path following using cached geometry."""
        if not self.path_waypoints or self.current_waypoint_index >= len(self.path_waypoints):
            self.predicted_path = []
            return
        if not hasattr(self, 'path_segment_lengths') or len(self.path_segment_lengths) == 0:
            self._rebuild_path_geometry()
        predicted_positions = []
        current_x, current_y = self.x, self.y
        current_speed = max(1.5, abs(self.speed))
        waypoint_idx = self.current_waypoint_index
        for frame in range(0, self.prediction_horizon, 6):
            if waypoint_idx >= len(self.path_waypoints):
                break
            target_x, target_y = self.path_waypoints[waypoint_idx]
            dx = target_x - current_x
            dy = target_y - current_y
            distance_to_target = math.hypot(dx, dy)
            if distance_to_target < 30:
                waypoint_idx += 1
                continue
            if distance_to_target > 0:
                inv = 1.0 / distance_to_target
                dx *= inv
                dy *= inv
            optimal_speed = current_speed
            if waypoint_idx < len(self.path_segment_headings) - 1:
                h1 = self.path_segment_headings[waypoint_idx]
                h2 = self.path_segment_headings[waypoint_idx + 1]
                turn_angle = abs(h2 - h1)
                if turn_angle > math.pi:
                    turn_angle = 2 * math.pi - turn_angle
                if turn_angle > 0.3:
                    turn_factor = max(0.4, 1.0 - (turn_angle / math.pi))
                    optimal_speed = current_speed * turn_factor
            move_distance = optimal_speed * 6
            current_x += dx * move_distance
            current_y += dy * move_distance
            predicted_positions.append((current_x, current_y))
        self.predicted_path = predicted_positions
    
    def calculate_ai_predicted_path(self):
        """Calculate where the car will actually go based on AI decisions (not optimal path)"""
        if not self.use_ai or len(self.raycast_distances) < 8:
            self.ai_predicted_path = []
            return
        
        ai_predicted_positions = []
        
        # Start with current state
        sim_x = self.x
        sim_y = self.y
        sim_angle = self.angle
        sim_speed = self.speed
        
        # Simulate AI behavior for next 2 seconds
        for frame in range(0, 120, 3):  # 2 seconds, sample every 3 frames for performance
            # Simulate raycasts from this predicted position
            sim_raycast_distances = []
            if self.road_system:
                for angle_offset in self.raycast_angles:
                    ray_angle = sim_angle + angle_offset
                    distance = self.road_system.raycast_to_road_edge(
                        sim_x, sim_y, ray_angle, self.raycast_length
                    )
                    sim_raycast_distances.append(distance)
            else:
                sim_raycast_distances = [self.raycast_length] * 8
            
            # Prepare AI inputs using the same logic as ai_move()
            inputs = []
            
            # 1. Raycast data (8 values) - normalized
            for distance in sim_raycast_distances:
                inputs.append(distance / self.raycast_length)
            
            # 2. Speed (1 value) - normalized
            inputs.append(min(1.0, abs(sim_speed) / 5.0))
            
            # 3. Current target direction (1 value) - normalized angle difference
            if self.target_x is not None and self.target_y is not None:
                dx = self.target_x - sim_x
                dy = self.target_y - sim_y
                target_angle = math.degrees(math.atan2(dx, -dy))
                angle_diff = target_angle - sim_angle
                while angle_diff > 180:
                    angle_diff -= 360
                while angle_diff < -180:
                    angle_diff += 360
                target_direction = angle_diff / 180.0
            else:
                target_direction = 0.0
            inputs.append(target_direction)
            
            # 4. Current target distance (1 value) - normalized
            if self.target_x is not None and self.target_y is not None:
                target_distance = min(1.0, math.sqrt((sim_x - self.target_x)**2 + (sim_y - self.target_y)**2) / 200.0)
            else:
                target_distance = 0.0
            inputs.append(target_distance)
            
            # 5. Velocity components (2 values) - normalized
            velocity_x = math.sin(math.radians(sim_angle)) * sim_speed / 5.0
            velocity_y = -math.cos(math.radians(sim_angle)) * sim_speed / 5.0
            inputs.append(velocity_x)
            inputs.append(velocity_y)
            
            # 6-11. Add simplified versions of other inputs (using current values for simplicity)
            inputs.append(self.current_waypoint_index / max(1, len(self.path_waypoints)) if self.path_waypoints else 0.0)  # Progress
            inputs.append(0.0)  # Next waypoint direction (simplified)
            inputs.append(0.0)  # Path curvature (simplified)
            inputs.append(self.path_following_accuracy)  # Path following accuracy
            inputs.append(0.5)  # Average deviation (simplified)
            inputs.append(0.0)  # Predicted direction (simplified to avoid recursion)
            
            # Ensure we have exactly 17 inputs
            while len(inputs) < 17:
                inputs.append(0.0)
            inputs = inputs[:17]
            
            # Get AI decision for this simulated state
            input_tensor = torch.tensor(inputs, dtype=torch.float32)
            with torch.no_grad():
                outputs = self.ai(input_tensor)
            
            # Extract AI decisions
            acceleration = outputs[0].item()
            steering = outputs[1].item()
            brake = outputs[2].item()
            
            # Apply AI decisions to simulated car state
            if abs(brake) > abs(acceleration) and brake > 0:
                sim_speed -= brake * 0.4
            elif acceleration > 0:
                sim_speed += acceleration * 0.25
            elif acceleration < 0:
                sim_speed += acceleration * 0.2
            
            # Prevent going backwards and cap speed
            if sim_speed < 0:
                sim_speed = 0
            if sim_speed > 4.0:
                sim_speed = 4.0
            
            # Apply steering
            if abs(sim_speed) > 0.1:
                sim_angle += steering * 4.0
            
            # Apply friction
            sim_speed *= 0.95
            
            # Update position based on new speed and angle
            sim_x += math.sin(math.radians(sim_angle)) * sim_speed
            sim_y -= math.cos(math.radians(sim_angle)) * sim_speed
            
            ai_predicted_positions.append((sim_x, sim_y))
        
        self.ai_predicted_path = ai_predicted_positions
    
    def update_path_following_accuracy(self):
        """Update how well the car is following its predicted path"""
        if not self.predicted_path:
            self.path_following_accuracy = 0.0
            return
        
        # Find closest point on predicted path
        min_distance = float('inf')
        for pred_x, pred_y in self.predicted_path[:30]:  # Only check near-term predictions
            distance = math.sqrt((self.x - pred_x)**2 + (self.y - pred_y)**2)
            min_distance = min(min_distance, distance)
        
        # Convert distance to accuracy score (closer = higher accuracy)
        max_allowed_deviation = 50  # pixels
        accuracy = max(0.0, 1.0 - (min_distance / max_allowed_deviation))
        
        self.path_deviation_history.append(min_distance)
        self.path_following_accuracy = accuracy
        
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
        
        # Draw raycasts & route for best active car only
        if is_best:
            self.draw_raycasts(camera)
            self.draw_route(camera, is_best)
    
    def draw_route(self, camera, is_best=False):
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
                # Final destination - special marker for individual destination
                pygame.draw.circle(screen, (255, 100, 100), screen_pos, 12)  # Red-orange for individual destination
                pygame.draw.circle(screen, (255, 255, 255), screen_pos, 12, 3)  # White border
                # Add destination text
                font = pygame.font.Font(None, 20)
                dest_text = "GOAL"
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
            
        # NEW: Draw 3-second predictive path arrow with gradient effects
        if hasattr(self, 'predicted_path') and self.predicted_path:
            car_screen = camera.world_to_screen(self.x, self.y)
            
            # Draw gradient path showing next 3 seconds
            prev_screen = car_screen
            for i, (pred_x, pred_y) in enumerate(self.predicted_path):
                pred_screen = camera.world_to_screen(pred_x, pred_y)
                
                # Gradient from bright yellow to orange to red
                progress = i / max(1, len(self.predicted_path) - 1)
                if progress < 0.5:
                    # Yellow to orange
                    t = progress * 2
                    color = (255, int(255 * (1 - t * 0.5)), 0)  # Yellow to orange
                else:
                    # Orange to red
                    t = (progress - 0.5) * 2
                    color = (255, int(128 * (1 - t)), 0)  # Orange to red
                
                # Draw thicker line for early predictions, thinner for later
                thickness = max(2, int(8 * (1 - progress)))
                
                # Make sure we don't go off screen
                if (0 <= pred_screen[0] <= camera.screen_width and 
                    0 <= pred_screen[1] <= camera.screen_height):
                    pygame.draw.line(screen, color, prev_screen, pred_screen, thickness)
                    
                    # Draw direction indicators every few points
                    if i % 5 == 0 and i > 0:
                        # Small arrow head showing direction
                        if i + 1 < len(self.predicted_path):
                            next_x, next_y = self.predicted_path[i + 1]
                            next_screen = camera.world_to_screen(next_x, next_y)
                            
                            # Calculate arrow direction
                            dx = next_screen[0] - pred_screen[0]
                            dy = next_screen[1] - pred_screen[1]
                            arrow_length = 10 * (1 - progress * 0.5)
                            
                            if dx != 0 or dy != 0:
                                angle = math.atan2(dy, dx)
                                # Arrow head points
                                arrow_x1 = pred_screen[0] + arrow_length * math.cos(angle - 0.5)
                                arrow_y1 = pred_screen[1] + arrow_length * math.sin(angle - 0.5)
                                arrow_x2 = pred_screen[0] + arrow_length * math.cos(angle + 0.5)
                                arrow_y2 = pred_screen[1] + arrow_length * math.sin(angle + 0.5)
                                
                                # Draw mini arrow
                                pygame.draw.line(screen, color, pred_screen, (int(arrow_x1), int(arrow_y1)), 2)
                                pygame.draw.line(screen, color, pred_screen, (int(arrow_x2), int(arrow_y2)), 2)
                
                prev_screen = pred_screen
            
            # Display path following accuracy
            if hasattr(self, 'path_following_accuracy'):
                accuracy_color = (0, 255, 0) if self.path_following_accuracy > 0.8 else \
                               (255, 255, 0) if self.path_following_accuracy > 0.6 else \
                               (255, 128, 0) if self.path_following_accuracy > 0.4 else (255, 0, 0)
                
                # Draw accuracy indicator next to car
                accuracy_text = f"{self.path_following_accuracy:.1%}"
                font = pygame.font.Font(None, 16)
                text_surface = font.render(accuracy_text, True, accuracy_color)
                text_pos = (car_screen[0] + 20, car_screen[1] - 20)
                screen.blit(text_surface, text_pos)
        
        # NEW: Draw AI predicted path (what the AI will actually do) in a different color
        # Only show for best car to improve performance
        if is_best and hasattr(self, 'ai_predicted_path') and self.ai_predicted_path:
            car_screen = camera.world_to_screen(self.x, self.y)
            
            # Draw AI prediction path showing what AI will actually do (in purple/magenta tones)
            prev_screen = car_screen
            for i, (pred_x, pred_y) in enumerate(self.ai_predicted_path):
                pred_screen = camera.world_to_screen(pred_x, pred_y)
                
                # Purple to magenta to pink gradient for AI predictions
                progress = i / max(1, len(self.ai_predicted_path) - 1)
                if progress < 0.5:
                    # Purple to magenta
                    t = progress * 2
                    color = (int(128 + 127 * t), 0, int(128 + 127 * t))  # Purple to magenta
                else:
                    # Magenta to pink
                    t = (progress - 0.5) * 2
                    color = (255, int(128 * t), int(255 - 127 * t))  # Magenta to pink
                
                # Draw slightly thinner line than optimal path to distinguish
                thickness = max(1, int(6 * (1 - progress * 0.7)))
                
                # Make sure we don't go off screen
                if (0 <= pred_screen[0] <= camera.screen_width and 
                    0 <= pred_screen[1] <= camera.screen_height):
                    pygame.draw.line(screen, color, prev_screen, pred_screen, thickness)
                    
                    # Draw smaller direction indicators for AI predictions
                    if i % 6 == 0 and i > 0:  # Less frequent arrows
                        # Small arrow head showing AI's predicted direction
                        if i + 1 < len(self.ai_predicted_path):
                            next_x, next_y = self.ai_predicted_path[i + 1]
                            next_screen = camera.world_to_screen(next_x, next_y)
                            
                            # Calculate arrow direction
                            dx = next_screen[0] - pred_screen[0]
                            dy = next_screen[1] - pred_screen[1]
                            arrow_length = 8 * (1 - progress * 0.6)  # Smaller arrows
                            
                            if dx != 0 or dy != 0:
                                angle = math.atan2(dy, dx)
                                # Arrow head points
                                arrow_x1 = pred_screen[0] + arrow_length * math.cos(angle - 0.6)
                                arrow_y1 = pred_screen[1] + arrow_length * math.sin(angle - 0.6)
                                arrow_x2 = pred_screen[0] + arrow_length * math.cos(angle + 0.6)
                                arrow_y2 = pred_screen[1] + arrow_length * math.sin(angle + 0.6)
                                
                                # Draw mini arrow for AI prediction
                                pygame.draw.line(screen, color, pred_screen, (int(arrow_x1), int(arrow_y1)), 1)
                                pygame.draw.line(screen, color, pred_screen, (int(arrow_x2), int(arrow_y2)), 1)
                
                prev_screen = pred_screen
            
            # Add legend text to show what each path represents (only for best car)
            font = pygame.font.Font(None, 16)
            optimal_text = font.render("OPTIMAL PATH", True, (255, 255, 0))
            ai_text = font.render("AI PREDICTION", True, (255, 0, 255))
            
            # Position legend in top-right of car area
            legend_x = car_screen[0] + 30
            legend_y = car_screen[1] - 40
            
            screen.blit(optimal_text, (legend_x, legend_y))
            screen.blit(ai_text, (legend_x, legend_y + 16))
    # (Removed duplicate drawing block)
    
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
            
            # More conservative path recalculation to prevent exploitation
            self.pathfinding_timer += 1
            if self.pathfinding_timer >= self.pathfinding_interval:
                self.pathfinding_timer = 0
                if self.pathfinder and not self.destination_reached and self.individual_destination:
                    # Only recalculate if we're far from destination AND making poor progress
                    current_dest_distance = math.sqrt(
                        (self.x - self.individual_destination[0])**2 + (self.y - self.individual_destination[1])**2
                    )
                    time_since_path = self.time_alive - self.path_generation_time
                    
                    # Conditions for path recalculation:
                    # 1. Far from destination (>200 pixels)
                    # 2. Have been using current path for a while (>10 seconds)
                    # 3. Have many waypoints left (>5)
                    should_recalculate = (
                        current_dest_distance > 200 and
                        time_since_path > 600 and  # 10 seconds
                        len(self.path_waypoints) - self.current_waypoint_index > 5
                    )
                    
                    if should_recalculate:
                        print(f"Car recalculating path: dist={current_dest_distance:.1f}, time_since={time_since_path}, waypoints_left={len(self.path_waypoints) - self.current_waypoint_index}")
                        self.generate_new_path_to_individual_destination()
        
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
        
        # Progress validation - check if making progress toward destination
        if self.use_ai and not self.saved_state and self.individual_destination:
            self.progress_validation_timer += 1
            if self.progress_validation_timer >= 180:  # Check every 3 seconds
                self.progress_validation_timer = 0
                
                current_destination_distance = math.sqrt(
                    (self.x - self.individual_destination[0])**2 + (self.y - self.individual_destination[1])**2
                )
                
                # Update tracking for validation
                self.last_destination_distance = current_destination_distance

    def update_raycasts(self):
        if not self.road_system:
            self.raycast_distances = [self.raycast_length] * 8  # Updated for 8 raycasts
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
        if len(self.raycast_distances) < 8:  # Updated for 8 raycasts
            return
        
        # Prepare enhanced input data for better planning
        inputs = []
        
        # 1. Raycast data (8 values) - normalized
        for distance in self.raycast_distances:
            inputs.append(distance / self.raycast_length)
        
        # 2. Speed (1 value) - normalized
        inputs.append(min(1.0, abs(self.speed) / 5.0))
        
        # 3. Current target direction (1 value) - normalized angle difference
        target_direction = self.get_target_direction() / 180.0  # Normalize to [-1, 1]
        inputs.append(target_direction)
        
        # 4. Current target distance (1 value) - normalized
        target_distance = min(1.0, self.get_target_distance() / 200.0)
        inputs.append(target_distance)
        
        # 5. Velocity components (2 values) - normalized
        velocity_x = math.sin(math.radians(self.angle)) * self.speed / 5.0
        velocity_y = -math.cos(math.radians(self.angle)) * self.speed / 5.0
        inputs.append(velocity_x)
        inputs.append(velocity_y)
        
        # 6. Progress indicator (1 value) - how far through the path
        if self.path_waypoints:
            progress = self.current_waypoint_index / max(1, len(self.path_waypoints))
        else:
            progress = 0.0
        inputs.append(progress)
        
        # 7. Next waypoint direction (1 value) - encourage looking ahead
        next_waypoint_direction = 0.0
        if self.path_waypoints and self.current_waypoint_index + 1 < len(self.path_waypoints):
            next_x, next_y = self.path_waypoints[self.current_waypoint_index + 1]
            dx = next_x - self.x
            dy = next_y - self.y
            next_angle = math.degrees(math.atan2(dx, -dy))
            angle_diff = next_angle - self.angle
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360
            next_waypoint_direction = angle_diff / 180.0
        inputs.append(next_waypoint_direction)
        
        # 8. Path curvature ahead (1 value) - anticipate turns
        path_curvature = 0.0
        if (self.path_waypoints and self.current_waypoint_index + 2 < len(self.path_waypoints)):
            # Calculate angle between current->next and next->after_next
            curr_x, curr_y = self.path_waypoints[self.current_waypoint_index]
            next_x, next_y = self.path_waypoints[self.current_waypoint_index + 1]
            after_x, after_y = self.path_waypoints[self.current_waypoint_index + 2]
            
            angle1 = math.atan2(next_x - curr_x, -(next_y - curr_y))
            angle2 = math.atan2(after_x - next_x, -(after_y - next_y))
            curvature = abs(angle2 - angle1)
            if curvature > math.pi:
                curvature = 2 * math.pi - curvature
            path_curvature = min(1.0, curvature / math.pi)  # Normalize to [0,1]
        inputs.append(path_curvature)
        
        # 9. Path following accuracy (1 value) - how well following predicted path
        inputs.append(self.path_following_accuracy)
        
        # 10. Average path deviation (1 value) - recent deviation from optimal path
        if self.path_deviation_history:
            avg_deviation = sum(list(self.path_deviation_history)[-10:]) / min(10, len(self.path_deviation_history))
            normalized_deviation = min(1.0, avg_deviation / 50.0)  # Normalize with max 50 pixels
        else:
            normalized_deviation = 1.0  # High deviation if no history
        inputs.append(normalized_deviation)
        
        # 11. Predicted direction (1 value) - where path leads in next 1 second
        predicted_direction = 0.0
        if self.predicted_path and len(self.predicted_path) > 10:
            # Use prediction ~1 second ahead (30 frames ahead, sampled every 6, so index 5)
            pred_idx = min(5, len(self.predicted_path) - 1)
            pred_x, pred_y = self.predicted_path[pred_idx]
            dx = pred_x - self.x
            dy = pred_y - self.y
            pred_angle = math.degrees(math.atan2(dx, -dy))
            angle_diff = pred_angle - self.angle
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360
            predicted_direction = angle_diff / 180.0
        inputs.append(predicted_direction)
        
        # Ensure we have exactly 17 inputs
        while len(inputs) < 17:
            inputs.append(0.0)
        inputs = inputs[:17]  # Trim if too many
        
        # Convert to tensor
        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        
        # Run AI
        with torch.no_grad():
            outputs = self.ai(input_tensor)
        
        # Extract control outputs: acceleration, steering, brake
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
        # Balanced fitness calculation focused on pathfinding and progress
        
        # 1. SURVIVAL REWARDS - Basic staying alive and moving
        # Time alive bonus (small but steady)
        time_bonus = self.time_alive * 0.05
        
        # Distance traveled bonus (encourages movement)
        distance_bonus = self.distance_traveled * 0.5
        
        # Road staying bonus (important for not crashing)
        road_bonus = 0
        if self.off_road_time == 0:
            road_bonus = self.time_alive * 0.15  # Moderate bonus for staying on road
        
        # 2. PATHFINDING REWARDS - Main focus
        pathfinding_bonus = 0
        if self.path_waypoints:
            # Progress through path (linear scaling)
            waypoint_progress = self.current_waypoint_index / max(1, len(self.path_waypoints))
            pathfinding_bonus += waypoint_progress * 200  # Good bonus for following path
            
            # Proximity to current target
            if self.target_x is not None and self.target_y is not None:
                target_distance = self.get_target_distance()
                # Inverse distance bonus (closer = better)
                proximity_bonus = max(0, (150 - target_distance) / 150) * 75
                pathfinding_bonus += proximity_bonus
        
        # 3. CHECKPOINT REWARDS - Moderate but important
        checkpoint_bonus = 0
        if self.checkpoint_streak > 0:
            # Linear bonus for checkpoint streaks (not exponential)
            checkpoint_bonus = self.checkpoint_streak * 40
            
            # Small bonus for max streak achieved
            checkpoint_bonus += self.max_checkpoint_streak * 20
        
        # Fast checkpoint completion bonus
        fast_bonus = 0
        if self.consecutive_fast_checkpoints > 0:
            fast_bonus = self.consecutive_fast_checkpoints * 25  # Linear scaling
        
        # 4. PATH FOLLOWING ACCURACY - Moderate reward
        accuracy_bonus = 0
        if hasattr(self, 'path_following_accuracy'):
            # Reward good path following
            accuracy_bonus = self.path_following_accuracy * 30  # Per frame accuracy bonus
            
            # Sustained accuracy bonus (but not massive)
            if hasattr(self, 'path_deviation_history') and len(self.path_deviation_history) >= 30:
                recent_deviations = list(self.path_deviation_history)[-30:]
                recent_accuracy = []
                for deviation in recent_deviations:
                    frame_accuracy = max(0.0, 1.0 - (deviation / 50.0))
                    recent_accuracy.append(frame_accuracy)
                
                avg_recent_accuracy = sum(recent_accuracy) / len(recent_accuracy)
                
                # Moderate sustained accuracy bonus
                if avg_recent_accuracy > 0.7:  # 70%+ accuracy
                    sustained_bonus = (avg_recent_accuracy - 0.7) * 150
                    accuracy_bonus += sustained_bonus
        
        # 5. DESTINATION REWARD - Major achievement
        destination_bonus = 0
        if self.destination_reached or self.saved_state:
            destination_bonus = 2000  # Significant but not overwhelming bonus
            # Time bonus for reaching destination quickly
            if self.time_alive < 1200:  # Within 20 seconds
                destination_bonus += (1200 - self.time_alive) * 2
        
        # 6. PENALTIES - Keep minimal
        crash_penalty = 200 if self.crashed else 0  # Moderate crash penalty
        off_road_penalty = self.off_road_time * 2  # Light off-road penalty
        
        # Final fitness calculation - clean and balanced
        self.fitness = (
            time_bonus + distance_bonus + road_bonus + 
            pathfinding_bonus + checkpoint_bonus + fast_bonus + 
            accuracy_bonus + destination_bonus -
            crash_penalty - off_road_penalty
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
        try:
            car.ai.load_state_dict(car_data['ai_state'])
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"Neural network architecture mismatch detected. Creating new network for compatibility.")
                # The old network has different input size, so we'll keep the new randomly initialized network
                # This happens when upgrading from 12-input to 17-input networks
                pass  # Keep the new randomly initialized 17-input network
            else:
                # Re-raise other runtime errors
                raise e
    
    return car


def evolve_population(cars, population_size=population_size, road_system=None, pathfinder=None, generation=1):
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
    
    # No need to generate shared destinations - each car handles its own individual destination
    
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
        new_car.checkpoint_times = []  # Reset checkpoint times
        new_car.last_checkpoint_time = 0  # Reset checkpoint timing
        new_car.current_checkpoint_start_time = 0  # Reset checkpoint start time
        new_car.checkpoint_streak = 0  # Reset checkpoint streak
        new_car.max_checkpoint_streak = 0  # Reset max streak
        new_car.consecutive_fast_checkpoints = 0  # Reset fast streak
        new_car.pathfinder = pathfinder  # Ensure pathfinder is set
        new_car.destination_reached = False  # Reset destination status
        new_car.saved_state = False  # Reset saved state
        # Each car will generate its own individual destination
        new_car.generate_individual_destination()  # Generate unique destination for this car
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
        offspring.checkpoint_times = []  # Reset checkpoint times
        offspring.last_checkpoint_time = 0  # Reset checkpoint timing
        offspring.current_checkpoint_start_time = 0  # Reset checkpoint start time
        offspring.checkpoint_streak = 0  # Reset checkpoint streak
        offspring.max_checkpoint_streak = 0  # Reset max streak
        offspring.consecutive_fast_checkpoints = 0  # Reset fast streak
        offspring.color = random.choice([RED, GREEN, BLUE])
        offspring.pathfinder = pathfinder  # Ensure pathfinder is set
        offspring.destination_reached = False  # Reset destination status
        offspring.saved_state = False  # Reset saved state
        # Each car will generate its own individual destination
        offspring.generate_individual_destination()  # Generate unique destination for this car
        new_cars.append(offspring)
    
    # Auto-save the population after evolution
    save_population(new_cars, generation)
    
    return new_cars

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
    
    # Network architecture: input(17) -> hidden(64) -> hidden(64) -> output(3)
    input_size = 17
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
            # Prepare inputs (same as in ai_move) - 17 inputs total
            inputs = []
            
            # 1. Raycast distances (8 values)
            for distance in car.raycast_distances:
                inputs.append(distance / car.raycast_length)
            
            # 2. Speed (1 value)
            inputs.append(min(1.0, abs(car.speed) / 5.0))
            
            # 3. Current target direction (1 value)
            target_direction = car.get_target_direction() / 180.0
            inputs.append(target_direction)
            
            # 4. Current target distance (1 value)
            target_distance = min(1.0, car.get_target_distance() / 200.0)
            inputs.append(target_distance)
            
            # 5. Velocity components (2 values)
            velocity_x = math.sin(math.radians(car.angle)) * car.speed / 5.0
            velocity_y = -math.cos(math.radians(car.angle)) * car.speed / 5.0
            inputs.append(velocity_x)
            inputs.append(velocity_y)
            
            # 6. Progress indicator (1 value)
            if car.path_waypoints:
                progress = car.current_waypoint_index / max(1, len(car.path_waypoints))
            else:
                progress = 0.0
            inputs.append(progress)
            
            # 7. Next waypoint direction (1 value)
            next_waypoint_direction = 0.0
            if car.path_waypoints and car.current_waypoint_index + 1 < len(car.path_waypoints):
                next_x, next_y = car.path_waypoints[car.current_waypoint_index + 1]
                dx = next_x - car.x
                dy = next_y - car.y
                next_angle = math.degrees(math.atan2(dx, -dy))
                angle_diff = next_angle - car.angle
                while angle_diff > 180:
                    angle_diff -= 360
                while angle_diff < -180:
                    angle_diff += 360
                next_waypoint_direction = angle_diff / 180.0
            inputs.append(next_waypoint_direction)
            
            # 8. Path curvature ahead (1 value)
            path_curvature = 0.0
            if (car.path_waypoints and car.current_waypoint_index + 2 < len(car.path_waypoints)):
                curr_x, curr_y = car.path_waypoints[car.current_waypoint_index]
                next_x, next_y = car.path_waypoints[car.current_waypoint_index + 1]
                after_x, after_y = car.path_waypoints[car.current_waypoint_index + 2]
                
                angle1 = math.atan2(next_x - curr_x, -(next_y - curr_y))
                angle2 = math.atan2(after_x - next_x, -(after_y - next_y))
                curvature = abs(angle2 - angle1)
                if curvature > math.pi:
                    curvature = 2 * math.pi - curvature
                path_curvature = min(1.0, curvature / math.pi)
            inputs.append(path_curvature)
            
            # 9. Path following accuracy (1 value)
            inputs.append(car.path_following_accuracy if hasattr(car, 'path_following_accuracy') else 0.0)
            
            # Ensure we have exactly 17 inputs
            while len(inputs) < 17:
                inputs.append(0.0)
            inputs = inputs[:17]
            
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
    input_labels = ["Ray1", "Ray2", "Ray3", "Ray4", "Ray5", "Ray6", "Ray7", "Ray8", 
                   "Speed", "TargetDir", "TargetDist", "VelX", "VelY", "Progress", 
                   "NextWP", "Curvature", "PathAcc"]
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

# Each car generates its own individual destination automatically in the constructor
print("Cars initialized with individual destinations...")

evolution_timer = 0
evolution_interval = 40 * FPS  # 40 seconds per generation

print("Starting AI cars with individual pathfinding evolution...")

running = True
while running:
    screen.fill((34, 139, 34))  # Dark green background

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("Saving population before exit...")
            save_population(cars, generation)
            running = False
        elif event.type == pygame.KEYDOWN:
            # R key functionality disabled
            # if event.key == pygame.K_r:
            #     print("Resetting to new location...")
            #     # Load new random location
            #     road_system = OSMRoadSystem(
            #         center_lat=random.uniform(-40, 50), 
            #         center_lon=random.uniform(-120, 120), 
            #         radius=1000
            #     )
            #     # Rebuild pathfinder for new location
            #     pathfinder = DijkstraPathfinder(road_system)
            #     road_bounds = road_system.get_road_bounds()
            #     camera.set_bounds(*road_bounds)
            #     
            #     # Reset all cars to new random positions with individual paths to shared destination
            #     # Generate new valid shared destination for new location
            #     new_shared_destination = None
            #     if pathfinder:
            #         for attempt in range(10):
            #             destination_node, test_destination = pathfinder.get_random_destination()
            #             if destination_node and test_destination:
            #                 test_spawn_x, test_spawn_y, _ = road_system.get_random_spawn_point()
            #                 test_path = pathfinder.find_path(test_spawn_x, test_spawn_y, test_destination[0], test_destination[1])
            #                 if test_path:
            #                     new_shared_destination = test_destination
            #                     print(f"New location shared destination set at: {new_shared_destination}")
            #                     break
            #     
            #     shared_destination = new_shared_destination
            #     
            #     for car in cars:
            #         if not car.crashed:
            #             spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
            #             car.x, car.y, car.angle = spawn_x, spawn_y, spawn_angle
            #             car.road_system = road_system
            #             car.pathfinder = pathfinder
            #             car.shared_destination = shared_destination
            #             if shared_destination:
            #                 car.generate_path_to_destination(shared_destination)  # Generate individual path to shared destination
            pass  # R key does nothing now
    
    keys = pygame.key.get_pressed()
    
    # Find best car and count alive cars (including saved cars)
    alive_cars = 0
    saved_cars = 0
    best_car = None  # May be a saved (stationary) champion
    for car in cars:
        if not car.crashed or car.saved_state:
            if evolution_timer % 30 == 0:  # Update fitness occasionally
                car.calculate_fitness()
            if best_car is None or car.fitness > best_car.fitness:
                best_car = car
            alive_cars += 1
            if car.saved_state:
                saved_cars += 1
    # Choose an active (moving) best car for visualization (skip saved_state cars)
    active_best_car = None
    if best_car and not best_car.saved_state and not best_car.crashed:
        active_best_car = best_car
    else:
        for c in cars:
            if not c.crashed and not c.saved_state:
                if active_best_car is None or c.fitness > active_best_car.fitness:
                    active_best_car = c
    
    # Camera follows active_best_car; if none, it won't auto-follow
    if not camera.manual_mode and active_best_car:
        camera.follow_target(active_best_car.x, active_best_car.y)
    camera.update(keys)
    
    # Draw roads
    road_system.draw_roads(screen, camera.x, camera.y, WIDTH, HEIGHT, camera.zoom)
    
    # Update and draw cars
    for car in cars:
        if not car.crashed or car.saved_state:  # Draw saved cars too
            if not car.saved_state:  # Only move cars that aren't saved
                # Calculate predictive path and accuracy
                if evolution_timer % 10 == 0:  # Update every 10 frames for performance
                    car.calculate_predictive_path()
                    # Only calculate expensive AI prediction visualization for active best (moving) car
                    if car == active_best_car:
                        car.calculate_ai_predicted_path()  # Calculate what AI will actually do
                car.update_path_following_accuracy()
                car.move(keys)
            is_visualized = (car == active_best_car)
            if (is_visualized or evolution_timer % 3 == 0) and not car.saved_state:
                car.update_raycasts()
            car.draw(camera, is_visualized)
    
    # Evolution logic (also reset if only saved cars remain)
    evolution_timer += 1
    unsaved_active_exists = any((not c.crashed) and (not c.saved_state) for c in cars)
    saved_active_exists = any((not c.crashed) and c.saved_state for c in cars)
    only_saved_remaining = (not unsaved_active_exists) and saved_active_exists
    if evolution_timer >= evolution_interval or alive_cars == 0 or only_saved_remaining:
        if only_saved_remaining:
            print("All remaining cars are saved. Early reset/evolution triggered.")
        cars = evolve_population(cars, population_size, road_system, pathfinder, generation)
        evolution_timer = 0
        generation += 1
        print(f"Generation {generation} started")
    
    # Display info with saved cars count and checkpoint streaks
    font = pygame.font.Font(None, 36)
    info_text = f"Gen: {generation} | Alive: {alive_cars} | Saved: {saved_cars} | Time: {evolution_timer // FPS}s"
    if best_car:
        info_text += f" | Best: {best_car.fitness:.1f}"
        if best_car.saved_state:
            info_text += " (SAVED!)"
    
    text_surface = font.render(info_text, True, WHITE)
    text_rect = text_surface.get_rect()
    text_rect.topleft = (10, 10)
    
    pygame.draw.rect(screen, (0, 0, 0, 128), text_rect.inflate(10, 5))
    screen.blit(text_surface, text_rect)
    
    # Display checkpoint streak info for best car
    if best_car:
        streak_font = pygame.font.Font(None, 28)
        streak_text = f"Best Car - Checkpoint Streak: {best_car.checkpoint_streak} | Max Streak: {best_car.max_checkpoint_streak} | Fast Streak: {best_car.consecutive_fast_checkpoints}"
        streak_surface = streak_font.render(streak_text, True, YELLOW)
        streak_rect = streak_surface.get_rect()
        streak_rect.topleft = (10, text_rect.bottom + 5)
        
        pygame.draw.rect(screen, (0, 0, 0, 128), streak_rect.inflate(10, 5))
        screen.blit(streak_surface, streak_rect)
    
    # Display pathfinding info with performance stats
    pathfinding_info = f"Pathfinding: {len(cars)} cars with individual destinations"
    pathfinding_surface = font.render(pathfinding_info, True, WHITE)
    pathfinding_rect = pathfinding_surface.get_rect()
    pathfinding_rect.topleft = (10, (streak_rect.bottom if best_car else text_rect.bottom) + 10)
    
    pygame.draw.rect(screen, (0, 0, 0, 128), pathfinding_rect.inflate(10, 5))
    screen.blit(pathfinding_surface, pathfinding_rect)
    
    # Draw AI control visualization for active (moving) best car
    if active_best_car:
        control_x = WIDTH - 120
        control_y = HEIGHT - 150
        draw_ai_controls(screen, active_best_car, control_x, control_y)
        # Neural network visualization
        nn_x = WIDTH - 220
        nn_y = 50
        draw_neural_network(screen, active_best_car, nn_x, nn_y)
    
    # Draw performance graph in bottom left corner
    graph_x = 20
    graph_y = HEIGHT - 250
    draw_performance_graph(screen, graph_x, graph_y)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
