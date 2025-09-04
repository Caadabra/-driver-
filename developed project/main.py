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

# Mode configuration: 0 = Training, 1 = Destination
# Change this value to switch between modes:
# - Mode 0 (Training): AI cars evolve and learn to drive
# - Mode 1 (Destination): Click two points to see the best AI drive from start to destination
mode = 1

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
FPS = 120

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

# Destination mode variables
destination_mode_start = None
destination_mode_end = None
destination_mode_car = None
destination_mode_state = "waiting"  # "waiting", "selecting_start", "selecting_end", "driving"

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

def get_best_car_from_population():
    """Get the best performing car from saved population data"""
    population_data, _ = load_population()
    if population_data is None:
        return None
    
    best_car_data = None
    best_fitness = -float('inf')
    
    for car_data in population_data['cars']:
        if car_data['fitness'] > best_fitness:
            best_fitness = car_data['fitness']
            best_car_data = car_data
    
    return best_car_data

def create_destination_mode_car(start_pos, end_pos, road_system, pathfinder):
    """Create a car for destination mode using the best AI from training"""
    best_car_data = get_best_car_from_population()
    
    if best_car_data is None:
        print("No saved population found. Using random AI.")
        car = Car(start_pos[0], start_pos[1], RED, use_ai=True, road_system=road_system, pathfinder=pathfinder)
    else:
        car = create_car_from_data(best_car_data, road_system, pathfinder)
        car.x, car.y = start_pos
        car.crashed = False
        car.time_alive = 0
        car.distance_traveled = 0
        car.fitness = 0
    
    # Set custom destination
    path = pathfinder.find_path(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
    if path:
        car.current_path = path
        car.path_waypoints = pathfinder.path_to_waypoints(path)
        car._resample_waypoints_evenly()
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

class SimpleCarAI(nn.Module):
    def __init__(self, input_size=11, hidden_size=32, output_size=2):  # output: accel, steer
        super(SimpleCarAI, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    def forward(self, x):
        return self.network(x)

class Car:
    def __init__(self, x, y, color, use_ai=False, road_system=None, pathfinder=None):
        # Core state
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.color = color
        self.width = 12/1.4
        self.height = 20/1.4

        # Sensors
        self.raycast_length = 100
        self.raycast_angles = [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5]
        self.raycast_distances = [self.raycast_length] * 8

        # Status & environment references
        self.use_ai = use_ai
        self.fitness = 0
        self.time_alive = 0
        self.distance_traveled = 0
        self.crashed = False
        self.road_system = road_system
        self.pathfinder = pathfinder
        self.last_valid_position = (x, y)
        self.off_road_time = 0
        self.max_off_road_time = 180

        # Checkpoint tracking
        self.checkpoint_times = []
        self.last_checkpoint_time = 0
        self.current_checkpoint_start_time = 0
        self.checkpoint_streak = 0
        self.max_checkpoint_streak = 0
        self.consecutive_fast_checkpoints = 0
        self.checkpoint_timeout_frames = 300  # Must reach a waypoint within 5 seconds (at 60fps)
        self.timed_out = False  # Flag if car died due to checkpoint timeout

        # AI network
        if self.use_ai:
            self.ai = SimpleCarAI(input_size=11, hidden_size=32, output_size=2)
            self.randomize_weights()

        # Control visualization
        self.ai_acceleration = 0
        self.ai_steering = 0
        # self.ai_brake = 0  # Removed - no longer used
        # self.ai_steer_sens = 0  # Removed - no longer used

        # Motion constraints (reverse support)
        self.can_reverse = True
        self.max_forward_speed = 4.0
        self.max_reverse_speed = 2.0
        self.reverse_penalty_timer = 0

        # Path deviation & progress metrics
        self.lateral_deviation = 0.0
        self.distance_along_path = 0.0
        self.last_distance_along_path = 0.0
        self.off_route_frames = 0
        self.max_allowed_deviation = 120

        # Stagnation detection
        self.stagnation_origin = (self.x, self.y)
        self.stagnation_frames = 0
        self.stagnation_limit_frames = 300
        self.stagnation_radius = 40
        self.stagnated = False

        # Pathfinding / routing
        self.current_path = []
        self.path_waypoints = []
        self.current_waypoint_index = 0
        self.waypoint_reach_distance = 30
        self.target_x = None
        self.target_y = None
        self.destination_node = None
        # pathfinding_timer and pathfinding_interval removed - no more path recalculation
        self.destination_reached = False
        self.saved_state = False
        self.individual_destination = None
        # Predictive path system
        self.prediction_horizon = 180  # frames ahead simulated (sampled sparsely)
        # predicted_path: list of instruction dicts {'pos':(x,y),'speed':v,'steer':s,'note':str}
        self.predicted_path = []
        self.path_following_accuracy = 0.0
        self.path_deviation_history = deque(maxlen=60)
        # Anti-loop tracking
        self.spawn_pos = (x, y)
        self.total_steering_abs = 0.0
        self.total_forward_distance = 0.0
        self.frames_since_waypoint_advance = 0
        self.prev_steering_output = 0.0

        # Progress validation
        self.initial_destination_distance = None
        self.last_destination_distance = None
        self.progress_validation_timer = 0
        self.path_generation_time = 0
        self.valid_checkpoints_only = True

        if self.pathfinder:
            self.generate_individual_destination()
            self.orient_to_first_waypoint()

    def orient_to_first_waypoint(self):
        # Orient car toward first meaningful waypoint using current path after pruning
        if not self.path_waypoints:
            return
        sx, sy = self.x, self.y
        # Skip overlapping initial waypoint
        idx = 0
        if len(self.path_waypoints) > 1 and math.hypot(self.path_waypoints[0][0]-sx, self.path_waypoints[0][1]-sy) < 5:
            idx = 1
        if idx >= len(self.path_waypoints):
            idx = len(self.path_waypoints)-1
        tx, ty = self.path_waypoints[idx]
        dx = tx - sx
        dy = ty - sy
        # Movement uses sin(angle) for x and -cos(angle) for y; angle 0 points up (-Y)
        # We want: sin(a)=dx/r, -cos(a)=dy/r -> cos(a) = -dy/r
        r = math.hypot(dx, dy)
        if r < 1e-6:
            return
        self.angle = math.degrees(math.atan2(dx, -dy)) - 180

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
    
    def _prune_backward_waypoints(self):
        """Remove leading waypoints that lie behind the current heading.

        This prevents newly generated paths from starting with points that require
        an immediate 180° turn because the pathfinder chose a nearby node behind the car.
        Criteria: keep dropping the first waypoint while its forward dot product < 0.
        Always keep at least one waypoint.
        """
        if not self.path_waypoints or len(self.path_waypoints) < 2:
            return
        heading_rad = math.radians(self.angle)
        fwd = (math.sin(heading_rad), -math.cos(heading_rad))  # matches movement logic
        pruned = False
        # Skip any initial waypoints too close (spawn overlap) outright
        while len(self.path_waypoints) > 1:
            wx, wy = self.path_waypoints[0]
            dx = wx - self.x; dy = wy - self.y
            dist = math.hypot(dx, dy)
            if dist < 5:  # practically at the car position
                self.path_waypoints.pop(0)
                pruned = True
                continue
            dot = dx * fwd[0] + dy * fwd[1]
            if dot < 0:  # waypoint lies behind
                self.path_waypoints.pop(0)
                pruned = True
            else:
                break
        if pruned:
            # Rebuild geometry caches since list changed
            self._rebuild_path_geometry()
            self.current_waypoint_index = 0
            if self.path_waypoints:
                self.target_x, self.target_y = self.path_waypoints[0]
    
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
                        # Orient first, then prune based on that heading
                        self.orient_to_first_waypoint()
                        self._prune_backward_waypoints()
                        self.individual_destination = destination_pos
                        self.current_waypoint_index = 0
                        # Set target after pruning
                        if self.path_waypoints:
                            self.target_x, self.target_y = self.path_waypoints[0]
                        # Initialize progress validation
                        self.initial_destination_distance = distance_to_dest
    
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

    def generate_new_path_to_individual_destination(self):
        """Recompute a path to the current individual destination while preserving progress metrics.

        Called when periodic reassessment decides the current route may be sub‑optimal.
        """
        if not (self.pathfinder and self.individual_destination):
            return
        dest_x, dest_y = self.individual_destination
        new_path = self.pathfinder.find_path(self.x, self.y, dest_x, dest_y)
        if not new_path:
            return
        self.current_path = new_path
        self.path_waypoints = self.pathfinder.path_to_waypoints(new_path)
        self._resample_waypoints_evenly()
        # Orient before pruning so heading is aligned with path
        self.orient_to_first_waypoint()
        self._prune_backward_waypoints()
        self.current_waypoint_index = 0
        if self.path_waypoints:
            self.target_x, self.target_y = self.path_waypoints[0]
        # Record path generation time for progress validation heuristics
        self.path_generation_time = self.time_alive
        # Angle already oriented above
    
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
        """Generate instruction-rich predictive path (backwards compatible)."""
        if not self.path_waypoints or self.current_waypoint_index >= len(self.path_waypoints):
            self.predicted_path = []
            return
        if not hasattr(self, 'path_segment_lengths') or len(self.path_segment_lengths) == 0:
            self._rebuild_path_geometry()
        step = 6
        sim_x, sim_y = self.x, self.y
        sim_speed = max(1.0, abs(self.speed))
        idx = self.current_waypoint_index
        prev_heading = math.radians(self.angle)
        out = []
        for frame in range(0, self.prediction_horizon, step):
            if idx >= len(self.path_waypoints):
                break
            tx, ty = self.path_waypoints[idx]
            dx = tx - sim_x; dy = ty - sim_y
            dist = math.hypot(dx, dy)
            if dist < 25:
                idx += 1
                continue
            dirx = dx / dist if dist else 0.0
            diry = dy / dist if dist else 0.0
            desired_heading = math.atan2(dirx, -diry)
            hd = desired_heading - prev_heading
            while hd > math.pi: hd -= 2*math.pi
            while hd < -math.pi: hd += 2*math.pi
            turn_mag = abs(hd)
            target_speed = min(self.max_forward_speed, sim_speed)
            if turn_mag > 0.25:
                target_speed *= max(0.35, 1 - turn_mag)
            move_dist = target_speed * step
            sim_x += dirx * move_dist
            sim_y += diry * move_dist
            prev_heading = desired_heading
            sim_speed = target_speed * 0.9 + sim_speed * 0.1
            note = 'straight'
            if turn_mag > 0.35: note = 'hard_turn'
            elif turn_mag > 0.18: note = 'turn'
            elif target_speed < self.max_forward_speed * 0.5: note = 'slow'
            out.append({'pos': (sim_x, sim_y), 'speed': target_speed, 'steer': max(-1.0, min(1.0, hd/0.6)), 'note': note})
        self.predicted_path = out
    
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
            
            # Add new map/path features placeholders to reach 20 inputs
            # 12 lateral deviation (use current car's, not simulated)
            inputs.append(min(1.0, getattr(self, 'lateral_deviation', 0.0) / getattr(self, 'max_allowed_deviation', 120)))
            # 13 off-route frames normalized
            inputs.append(min(1.0, getattr(self, 'off_route_frames', 0) / 300.0))
            # 14 delta progress placeholder (0 in simulation)
            inputs.append(0.0)
            # 15 reverse flag
            inputs.append(1.0 if sim_speed < -0.1 else 0.0)
            # 16 heading alignment placeholder (reuse target_direction approximation)
            inputs.append((target_direction + 1.0)/2.0)
            # 17 spare
            inputs.append(0.0)
            # Pad / trim to exactly 20
            while len(inputs) < 20:
                inputs.append(0.0)
            inputs = inputs[:20]
            
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
        for entry in self.predicted_path[:30]:  # Only check near-term predictions
            pred_x, pred_y = entry['pos'] if isinstance(entry, dict) else entry
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
        
        # Visual-only flip: sprite base points opposite to physics forward, so add 180 deg for rendering
        rotated_surface = pygame.transform.rotate(car_surface, -(self.angle + 180))
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
            # Steering sensitivity visualization removed
    
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
            for i, entry in enumerate(self.predicted_path):
                pred_x, pred_y = entry['pos'] if isinstance(entry, dict) else entry
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
                            nxt = self.predicted_path[i + 1]
                            next_x, next_y = nxt['pos'] if isinstance(nxt, dict) else nxt
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
        
    # Legend adjustment removed (purple AI prediction path removed)
    
    def _update_path_deviation_metrics(self):
        """Compute lateral deviation from current path segment and progress along path."""
        if not self.path_waypoints or self.current_waypoint_index >= len(self.path_waypoints):
            self.lateral_deviation = 0.0
            return
        # Determine segment from previous waypoint to current target
        idx = max(0, self.current_waypoint_index - 1)
        if idx >= len(self.path_waypoints) - 1:
            self.lateral_deviation = 0.0
            return
        x1, y1 = self.path_waypoints[idx]
        x2, y2 = self.path_waypoints[idx + 1]
        px, py = self.x, self.y
        vx, vy = x2 - x1, y2 - y1
        seg_len2 = vx * vx + vy * vy
        if seg_len2 <= 1e-6:
            self.lateral_deviation = 0.0
            return
        # Projection factor t in [0,1]
        t = ((px - x1) * vx + (py - y1) * vy) / seg_len2
        t_clamped = max(0.0, min(1.0, t))
        proj_x = x1 + vx * t_clamped
        proj_y = y1 + vy * t_clamped
        self.lateral_deviation = math.hypot(px - proj_x, py - proj_y)
        # Progress along path (approx): cumulative length up to segment + segment * t
        if hasattr(self, 'path_cumulative_lengths') and len(self.path_cumulative_lengths) > idx:
            base_len = self.path_cumulative_lengths[idx]
            seg_len = math.hypot(vx, vy)
            self.distance_along_path = base_len + seg_len * t_clamped
        # Track off-route duration
        if self.lateral_deviation > self.max_allowed_deviation:
            self.off_route_frames += 1
        else:
            self.off_route_frames = 0

    def _update_stagnation(self):
        """Update stagnation counters; crash if stationary within small area too long."""
        if self.saved_state or self.crashed:
            return
        dx = self.x - self.stagnation_origin[0]
        dy = self.y - self.stagnation_origin[1]
        dist = math.hypot(dx, dy)
        # Reset if sufficiently moved
        if dist > self.stagnation_radius:
            self.stagnation_origin = (self.x, self.y)
            self.stagnation_frames = 0
            return
        # Increment stagnation frames only if very low speed (prevent circling in tight loop cheat)
        if abs(self.speed) < 0.8:
            self.stagnation_frames += 1
        else:
            self.stagnation_frames = max(0, self.stagnation_frames - 2)
        if self.stagnation_frames >= self.stagnation_limit_frames:
            self.crashed = True
            self.stagnated = True

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
            if keys[pygame.K_DOWN]:  # reverse / brake
                if self.can_reverse:
                    self.speed -= 0.15  # allow gentle reverse acceleration
                else:
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
            prev_wp = self.current_waypoint_index
            self.check_waypoint_reached()
            if self.current_waypoint_index != prev_wp:
                self.frames_since_waypoint_advance = 0
            else:
                self.frames_since_waypoint_advance += 1
            
            # Path recalculation disabled to prevent disrupting good paths
        
        # Track distance for AI
        if self.use_ai:
            distance = math.sqrt((self.x - prev_x)**2 + (self.y - prev_y)**2)
            self.distance_traveled += distance
            if self.speed > 0.05:
                self.total_forward_distance += distance
            self.time_alive += 1

            # Early loop detection kill: after 3s, if displacement ratio very low
            if self.time_alive > 180 and not self.crashed:
                net_disp = math.hypot(self.x - self.spawn_pos[0], self.y - self.spawn_pos[1])
                if self.distance_traveled > 60:
                    ratio = net_disp / max(1.0, self.distance_traveled)
                    if ratio < 0.18:  # extremely tight circle
                        self.crashed = True

            # Waypoint stagnation forced crash (already timeout by checkpoint, but stricter)
            if self.frames_since_waypoint_advance > 480 and not self.crashed:
                self.crashed = True

            # Enforce checkpoint timeout (must reach next waypoint within 5 seconds)
            if (not self.saved_state and not self.crashed and self.path_waypoints and
                (self.time_alive - self.current_checkpoint_start_time) > self.checkpoint_timeout_frames):
                self.crashed = True
                self.timed_out = True

        # Apply friction
        self.speed *= 0.95
        # Clamp forward / reverse speeds
        if self.speed > self.max_forward_speed:
            self.speed = self.max_forward_speed
        if self.speed < -self.max_reverse_speed:
            self.speed = -self.max_reverse_speed

        # Reverse usage penalty tracking (light)
        if self.speed < -0.2:
            self.reverse_penalty_timer += 1
        else:
            self.reverse_penalty_timer = max(0, self.reverse_penalty_timer - 2)

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
        
        # Update map-derived path deviation metrics
        self._update_path_deviation_metrics()
        # Update stagnation tracking
        self._update_stagnation()

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
    
    def check_waypoint_reached(self):
        """Advance waypoint if close enough; update streaks, timing, predictive path & metrics."""
        if not self.path_waypoints or self.current_waypoint_index >= len(self.path_waypoints):
            return
        wx, wy = self.path_waypoints[self.current_waypoint_index]
        dist = math.hypot(self.x - wx, self.y - wy)
        if dist <= self.waypoint_reach_distance:
            # Time since last checkpoint
            frame_time = self.time_alive - self.current_checkpoint_start_time
            fast_threshold = 90  # 1.5 seconds at 60fps
            if self.current_checkpoint_start_time != 0:
                # Only apply timing after the first waypoint
                if frame_time < fast_threshold:
                    self.consecutive_fast_checkpoints += 1
                else:
                    self.consecutive_fast_checkpoints = 0
                self.checkpoint_times.append(frame_time)
            # Update streaks (simple increment, reset only if crash happens elsewhere)
            self.checkpoint_streak += 1
            self.max_checkpoint_streak = max(self.max_checkpoint_streak, self.checkpoint_streak)
            # Advance waypoint index
            self.current_waypoint_index += 1
            self.current_checkpoint_start_time = self.time_alive
            # Set new target if exists
            if self.current_waypoint_index < len(self.path_waypoints):
                self.target_x, self.target_y = self.path_waypoints[self.current_waypoint_index]
            else:
                # Destination reached
                self.destination_reached = True
                self.saved_state = True
            # Recalculate predictive path shortly after hitting a waypoint for fresh instructions
            self.calculate_predictive_path()
            # Update path following accuracy reference
            self.update_path_following_accuracy()

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
        if len(self.raycast_distances) < 8:
            return
        # Minimal input: 8 raycast distances, speed, direction to next waypoint, and road center deviation
        inputs = []
        for distance in self.raycast_distances:
            inputs.append(distance / self.raycast_length)
        inputs.append(min(1.0, abs(self.speed) / 5.0))
        # Direction to next waypoint
        target_direction = self.get_target_direction() / 180.0  # Normalize to [-1, 1]
        inputs.append(target_direction)
        # Road center deviation (normalized)
        road_center_deviation = 0.0
        if self.road_system:
            road_center_deviation = self.road_system.get_normalized_center_deviation(self.x, self.y)
        inputs.append(road_center_deviation)
        # Ensure exactly 11 inputs
        while len(inputs) < 11:
            inputs.append(0.0)
        inputs = inputs[:11]
        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.ai(input_tensor)
        # Only acceleration and steering
        acceleration = outputs[0].item()
        steering = outputs[1].item()
        self.ai_acceleration = acceleration
        self.ai_steering = steering
        # Apply acceleration
        if acceleration > 0.01:
            self.speed += acceleration * 0.25
        elif acceleration < -0.01:
            self.speed += acceleration * 0.25  # allow reverse if needed
        # Apply steering
        self.angle += steering * 10.0  # scale steering output
        
        # Only acceleration and steering are used now; brake and steer_sens removed
        # Clamp speeds (prevent casual negative speeds)
        if self.speed > self.max_forward_speed: self.speed = self.max_forward_speed
        if self.speed < -self.max_reverse_speed: self.speed = -self.max_reverse_speed
        # Steering is now applied directly above; smoothing and dynamic gain removed for simplicity

    def calculate_fitness(self):
        # Balanced fitness calculation focused on pathfinding and progress (with anti-loop)
        # 1. SURVIVAL REWARDS - Basic staying alive and moving
        time_bonus = self.time_alive * 0.025
        # Distance traveled bonus (reduced) & net displacement bonus (scaled slightly down to shift emphasis to safety)
        distance_bonus = self.distance_traveled * 0.30
        net_disp = math.hypot(self.x - self.spawn_pos[0], self.y - self.spawn_pos[1])
        net_disp_bonus = net_disp * 0.9
        
        # Survival integrity bonus – rewards staying alive longer without crashing (applies only if not crashed)
        survival_integrity = 0
        if not self.crashed:
            survival_integrity = (self.time_alive ** 0.9) * 0.3  # Sub-linear growth
        
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
            accuracy_bonus = self.path_following_accuracy * 60  # Stronger incentive to follow predicted path
            
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
        
        # 6. PENALTIES - route deviation & misuse
        # Strong crash penalty plus suppression of other positives later
        crash_penalty = 1200 if self.crashed else 0
        off_road_penalty = self.off_road_time * 2  # Light off-road penalty
        # Strong penalty for sustained off-route behavior
        off_route_penalty = 0
        if self.off_route_frames > 60:  # >1 second far off
            off_route_penalty += (self.off_route_frames - 60) * 5  # escalate quickly
        # Reverse penalty if excessive
        reverse_penalty = max(0, self.reverse_penalty_timer - 90) * 1.5  # grace period ~1.5s before penalizing
        # Minor continuous lateral deviation penalty (quadratic scaling)
        lateral_penalty = (max(0, self.lateral_deviation - self.max_allowed_deviation/2) ** 2) / 500.0
        # Stagnation penalty if triggered (heavy)
        stagnation_penalty = 0
        if self.stagnated:
            stagnation_penalty = 800  # large penalty for not moving
        # Timeout penalty (failed to reach a checkpoint within required time)
        timeout_penalty = 600 if getattr(self, 'timed_out', False) else 0
        # Loop & steering efficiency penalties
        loop_penalty = 0
        if self.distance_traveled > 300:
            ratio = net_disp / max(1.0, self.distance_traveled)
            if ratio < 0.4:
                loop_penalty = (0.4 - ratio) * 800
        steering_efficiency_penalty = 0
        if self.total_forward_distance > 80:
            steering_density = self.total_steering_abs / max(1.0, self.total_forward_distance)
            if steering_density > 1.2:
                steering_efficiency_penalty = (steering_density - 1.2) * 300
        progress_stagnation_penalty = 0
        if self.frames_since_waypoint_advance > 240:
            progress_stagnation_penalty = (self.frames_since_waypoint_advance - 240) * 2
        
        # Final fitness calculation - clean and balanced
        # Raycast clearance safety bonus: encourage keeping more space (normalize by sensor length)
        safety_clearance_bonus = 0
        if hasattr(self, 'raycast_distances') and self.raycast_distances:
            norm = [d / max(1.0, self.raycast_length) for d in self.raycast_distances]
            # Emphasize worst few directions (min distance) inversely
            avg_front = sum(norm[len(norm)//2-1:len(norm)//2+1]) / 2 if len(norm) >= 2 else sum(norm)/len(norm)
            min_clear = min(norm)
            safety_clearance_bonus = (avg_front * 25) + (min_clear * 10)
            if min_clear < 0.3:
                safety_clearance_bonus *= 0.2  # discourage scraping obstacles

        raw_score = (
            time_bonus + distance_bonus + net_disp_bonus + road_bonus + survival_integrity +
            pathfinding_bonus + checkpoint_bonus + fast_bonus + 
            accuracy_bonus + destination_bonus + safety_clearance_bonus -
            crash_penalty - off_road_penalty - off_route_penalty - reverse_penalty - lateral_penalty - stagnation_penalty - timeout_penalty -
            loop_penalty - steering_efficiency_penalty - progress_stagnation_penalty
        )

        # If crashed, heavily dampen remaining positives (simulates near total failure)
        if self.crashed:
            raw_score = raw_score * 0.15 - 400  # extra post-crash penalty tail

        self.fitness = raw_score

        # Update last distance along path for delta progress metric
        self.last_distance_along_path = self.distance_along_path
        
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
                # Attempt graceful upgrade if only the final layer output size changed (3 -> 4)
                print("Model size mismatch detected. Attempting to adapt saved weights to new architecture (adding steer_sens output)...")
                saved_state = car_data['ai_state']
                model_state = car.ai.state_dict()
                # Identify final linear layer keys
                final_weight_key = None
                final_bias_key = None
                for k in saved_state.keys():
                    if k.endswith('.2.weight') or k.endswith('.4.weight') or k.endswith('.6.weight'):
                        # Heuristic: last occurrence will be final layer in Sequential
                        final_weight_key = k
                    if k.endswith('.2.bias') or k.endswith('.4.bias') or k.endswith('.6.bias'):
                        final_bias_key = k
                if final_weight_key and final_bias_key and final_weight_key in saved_state:
                    old_w = saved_state[final_weight_key]
                    old_b = saved_state[final_bias_key]
                    if old_w.shape[0] == 3 and model_state[final_weight_key].shape[0] == 4:
                        # Expand to 4 outputs
                        new_w = model_state[final_weight_key].clone()
                        new_b = model_state[final_bias_key].clone()
                        new_w[:3, :] = old_w
                        new_b[:3] = old_b
                        saved_state[final_weight_key] = new_w
                        saved_state[final_bias_key] = new_b
                        try:
                            car.ai.load_state_dict(saved_state)
                            print("Successfully adapted old model to new 4-output architecture.")
                        except Exception as e2:
                            print(f"Adaptation failed, using fresh network. Error: {e2}")
                    else:
                        print("Final layer shape not as expected for adaptation; using fresh network.")
                else:
                    print("Could not locate final layer for adaptation; using fresh network.")
            else:
                raise e
    
    return car


def evolve_population(cars, population_size=population_size, road_system=None, pathfinder=None, generation=1):
    # Safety: ensure we have cars
    if not cars:
        print("evolve_population: received empty car list, regenerating fresh population")
        cars = []
        for _ in range(population_size):
            sx, sy, sa = road_system.get_random_spawn_point()
            nc = Car(sx, sy, random.choice([RED,GREEN,BLUE]), use_ai=True, road_system=road_system, pathfinder=pathfinder)
            nc.angle = sa
            nc.generate_individual_destination()
            nc.orient_to_first_waypoint()
            cars.append(nc)
    # Calculate fitness for all cars (catch exceptions per car)
    for car in cars:
        try:
            car.calculate_fitness()
        except Exception as e:
            print(f"Fitness calc error for car: {e}")
            car.fitness = -1e9
    # Sort by fitness (best first), prioritizing saved cars; guard if list empty
    if cars:
        cars.sort(key=lambda x: (x.saved_state, x.fitness), reverse=True)
    saved_count = sum(1 for car in cars if car.saved_state)
    best_fit = cars[0].fitness if cars else 0.0
    print(f"Generation evolved! Best fitness: {best_fit:.2f} | Saved cars: {saved_count}")
    
    # Track statistics for graphs
    if cars:
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
        # Attempt to generate a valid path several times
        got_path = False
        for _ in range(5):
            new_car.generate_individual_destination()
            if new_car.path_waypoints:
                got_path = True
                break
        if got_path:
            new_car.orient_to_first_waypoint()
        else:
            # Fallback: leave angle at spawn; mark destination_reached False
            new_car.individual_destination = None
            print("Elite clone failed to get path after retries; using fallback orientation")
    new_cars.append(new_car)
    
    # Create offspring with mutations
    while len(new_cars) < population_size:
        # Select parent from top half (fallback to 0 if small)
        half_index_limit = max(0, population_size // 2 - 1)
        parent_index = random.randint(0, min(len(cars) - 1, half_index_limit))
        parent = cars[parent_index]
        offspring = parent.clone()
        # Apply mutation (stronger for worse performers)
        base_mutation_rate = 0.12
        base_mutation_strength = 0.25
        if cars and parent.fitness > cars[0].fitness * 0.8:
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
        offspring.checkpoint_times = []
        offspring.last_checkpoint_time = 0
        offspring.current_checkpoint_start_time = 0
        offspring.checkpoint_streak = 0
        offspring.max_checkpoint_streak = 0
        offspring.consecutive_fast_checkpoints = 0
        offspring.color = random.choice([RED, GREEN, BLUE])
        offspring.pathfinder = pathfinder
        offspring.destination_reached = False
        offspring.saved_state = False
        # Attempt to generate path with retries & adaptive distance threshold fallback
        got_path = False
        for attempt in range(6):
            offspring.generate_individual_destination()
            if offspring.path_waypoints:
                got_path = True
                break
            if attempt == 3:
                offspring.x += random.uniform(-20,20)
                offspring.y += random.uniform(-20,20)
        if got_path:
            offspring.orient_to_first_waypoint()
        else:
            offspring.individual_destination = None
            print("Offspring failed to get path after retries; leaving without path")
        new_cars.append(offspring)

    # Final safety: ensure population size and at least one path-having car
    pathful = sum(1 for c in new_cars if c.path_waypoints)
    if pathful == 0:
        print("No cars with valid paths; regenerating entire population fresh")
        new_cars = []
        for _ in range(population_size):
            sx, sy, sa = road_system.get_random_spawn_point()
            nc = Car(sx, sy, random.choice([RED,GREEN,BLUE]), use_ai=True, road_system=road_system, pathfinder=pathfinder)
            nc.angle = sa
            for _ in range(5):
                nc.generate_individual_destination()
                if nc.path_waypoints:
                    nc.orient_to_first_waypoint()
                    break
            new_cars.append(nc)
    if len(new_cars) < population_size:
        for _ in range(population_size - len(new_cars)):
            sx, sy, sa = road_system.get_random_spawn_point()
            nc = Car(sx, sy, random.choice([RED,GREEN,BLUE]), use_ai=True, road_system=road_system, pathfinder=pathfinder)
            nc.angle = sa
            nc.generate_individual_destination()
            if nc.path_waypoints:
                nc.orient_to_first_waypoint()
            new_cars.append(nc)

    print(f"Evolution produced {len(new_cars)} cars ({pathful} with paths)")
    
    # Assign a shared destination/path for the whole new generation for fair evaluation
    try:
        assign_shared_destination(new_cars, pathfinder)
    except Exception as e:
        print(f"assign_shared_destination failed: {e}")
    # Auto-save the population after evolution
    save_population(new_cars, generation)
    
    return new_cars

def assign_shared_destination(cars, pathfinder, min_distance=400, max_attempts=15):
    """Pick one destination and assign the path to all cars (shared route training)."""
    if not cars or not pathfinder:
        return None
    # Use first car's position as canonical start to validate destination
    sx, sy = cars[0].x, cars[0].y
    dest = None
    path0 = None
    for _ in range(max_attempts):
        node, pos = pathfinder.get_random_destination()
        if not node:
            continue
        if math.hypot(pos[0]-sx, pos[1]-sy) < min_distance:
            continue
        p = pathfinder.find_path(sx, sy, pos[0], pos[1])
        if p:
            dest = pos
            path0 = p
            break
    if dest is None:
        # Fallback: take whatever
        node, pos = pathfinder.get_random_destination()
        dest = pos
    # Assign to each car individually from its own start
    for car in cars:
        try:
            car.current_path = pathfinder.find_path(car.x, car.y, dest[0], dest[1]) or []
            car.path_waypoints = pathfinder.path_to_waypoints(car.current_path) if car.current_path else []
            car._resample_waypoints_evenly()
            car.orient_to_first_waypoint()
            car._prune_backward_waypoints()
            car.current_waypoint_index = 0
            car.individual_destination = dest
            if car.path_waypoints:
                car.target_x, car.target_y = car.path_waypoints[0]
                car.initial_destination_distance = math.hypot(car.x - dest[0], car.y - dest[1])
                car.last_destination_distance = None
        except Exception as e:
            print(f"assign_shared_destination: car failed to get path: {e}")
    return dest

# Modern AI Visualization System
class ModernHUD:
    def __init__(self):
        self.font_large = pygame.font.Font(None, 28)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)
        self.font_mono = pygame.font.SysFont('Consolas', 14)

        # Colors
        self.colors = {
            'bg_dark': (20, 20, 25, 220),
            'bg_medium': (30, 30, 35, 200),
            'bg_light': (40, 40, 45, 180),
            'accent': (0, 255, 200),
            'accent_dim': (0, 180, 140),
            'success': (0, 255, 100),
            'warning': (255, 200, 0),
            'error': (255, 80, 80),
            'text_primary': (240, 240, 245),
            'text_secondary': (180, 180, 185),
            'text_dim': (120, 120, 125),
            'border': (60, 60, 70),
            'border_highlight': (100, 100, 110)
        }

        # Panel positions and sizes
        self.panels = {
            'ai_status': {'x': 20, 'y': 20, 'w': 280, 'h': 200},
            'performance': {'x': 20, 'y': 240, 'w': 280, 'h': 180},
            'debug': {'x': 20, 'y': 440, 'w': 280, 'h': 200},
            'minimap': {'x': 320, 'y': 20, 'w': 200, 'h': 200},
            'network': {'x': 540, 'y': 20, 'w': 200, 'h': 200},
            'controls': {'x': 20, 'y': 660, 'w': 280, 'h': 100}
        }

    def draw_rounded_rect(self, surface, color, rect, radius=8, border_color=None, border_width=1):
        """Draw a rounded rectangle"""
        x, y, w, h = rect

        # Draw main rectangle (simplified - no rounded corners for compatibility)
        pygame.draw.rect(surface, color, rect)

        # Draw border if specified
        if border_color:
            pygame.draw.rect(surface, border_color, rect, border_width)

    def draw_panel(self, surface, title, panel_rect, content_func):
        """Draw a modern panel with title and content"""
        # Extract rect values from panel dictionary
        if isinstance(panel_rect, dict):
            rect = (panel_rect['x'], panel_rect['y'], panel_rect['w'], panel_rect['h'])
        else:
            rect = panel_rect

        # Panel background
        self.draw_rounded_rect(surface, self.colors['bg_dark'], rect, border_color=self.colors['border'])

        # Title bar
        title_rect = (rect[0], rect[1], rect[2], 30)
        self.draw_rounded_rect(surface, self.colors['bg_medium'], title_rect, radius=6)

        # Title text
        title_surf = self.font_medium.render(title, True, self.colors['text_primary'])
        surface.blit(title_surf, (rect[0] + 12, rect[1] + 6))

        # Content area
        content_rect = (rect[0] + 8, rect[1] + 35, rect[2] - 16, rect[3] - 43)
        content_func(surface, content_rect)

    def draw_ai_status_panel(self, surface, rect, car):
        """Draw AI status and decision visualization"""
        y_offset = rect[1]

        # Current decisions
        decisions = [
            ("Acceleration", car.ai_acceleration if hasattr(car, 'ai_acceleration') else 0, -1, 1),
            ("Steering", car.ai_steering if hasattr(car, 'ai_steering') else 0, -1, 1),
            ("Speed", car.speed, -4, 4)
        ]

        for label, value, min_val, max_val in decisions:
            # Label
            label_surf = self.font_small.render(label, True, self.colors['text_secondary'])
            surface.blit(label_surf, (rect[0], y_offset))

            # Progress bar background
            bar_rect = (rect[0] + 80, y_offset, 120, 16)
            self.draw_rounded_rect(surface, self.colors['bg_medium'], bar_rect, radius=4)

            # Progress bar fill
            normalized = (value - min_val) / (max_val - min_val)
            fill_width = max(0, min(120, int(120 * abs(normalized))))
            fill_color = self.colors['success'] if value >= 0 else self.colors['error']

            if normalized >= 0:
                fill_rect = (rect[0] + 80, y_offset, fill_width, 16)
            else:
                fill_rect = (rect[0] + 80 + 120 - fill_width, y_offset, fill_width, 16)

            self.draw_rounded_rect(surface, fill_color, fill_rect, radius=4)

            # Value text
            value_text = f"{value:.2f}"
            value_surf = self.font_small.render(value_text, True, self.colors['text_primary'])
            surface.blit(value_surf, (rect[0] + 210, y_offset))

            y_offset += 22

        # Target direction indicator
        target_dir = car.get_target_direction() if hasattr(car, 'get_target_direction') else 0
        compass_surf = self.font_small.render("Target Direction", True, self.colors['text_secondary'])
        surface.blit(compass_surf, (rect[0], y_offset))

        # Compass visualization
        compass_center = (rect[0] + 60, y_offset + 25)
        pygame.draw.circle(surface, self.colors['bg_medium'], compass_center, 20)
        pygame.draw.circle(surface, self.colors['border'], compass_center, 20, 2)

        # Direction indicator
        angle_rad = math.radians(car.angle + target_dir * 180)
        indicator_x = compass_center[0] + 15 * math.sin(angle_rad)
        indicator_y = compass_center[1] - 15 * math.cos(angle_rad)
        pygame.draw.line(surface, self.colors['accent'], compass_center, (indicator_x, indicator_y), 3)

    def draw_performance_panel(self, surface, rect, generation, best_fitness, avg_fitness, cars_alive):
        """Draw performance metrics"""
        y_offset = rect[1]

        metrics = [
            ("Generation", generation, "text_primary"),
            ("Best Fitness", best_fitness, "text_primary"),
            ("Avg Fitness", avg_fitness, "text_primary"),
            ("Cars Alive", cars_alive, "text_primary")
        ]

        for i, (label, value, color_key) in enumerate(metrics):
            label_surf = self.font_small.render(label, True, self.colors['text_secondary'])
            surface.blit(label_surf, (rect[0], y_offset))

            # Format fitness values
            if "Fitness" in label:
                display_value = ".1f"
            else:
                display_value = str(value)

            value_surf = self.font_small.render(display_value, True, self.colors[color_key])
            surface.blit(value_surf, (rect[0] + 120, y_offset))

            y_offset += 20

        # Progress bar for generation
        if hasattr(self, 'generation_progress'):
            progress = min(1.0, self.generation_progress)
            bar_rect = (rect[0], y_offset, rect[2], 8)
            self.draw_rounded_rect(surface, self.colors['bg_medium'], bar_rect, radius=4)

            fill_rect = (rect[0], y_offset, int(rect[2] * progress), 8)
            self.draw_rounded_rect(surface, self.colors['accent'], fill_rect, radius=4)

    def draw_debug_panel(self, surface, rect, car):
        """Draw debug information"""
        y_offset = rect[1]

        # Raycast visualization
        ray_surf = self.font_small.render("Raycasts", True, self.colors['text_secondary'])
        surface.blit(ray_surf, (rect[0], y_offset))
        y_offset += 18

        if hasattr(car, 'raycast_distances') and car.raycast_distances:
            # Show raycast bars
            for i, distance in enumerate(car.raycast_distances[:8]):
                bar_y = y_offset + i * 12
                bar_rect = (rect[0], bar_y, 60, 8)
                self.draw_rounded_rect(surface, self.colors['bg_medium'], bar_rect, radius=2)

                # Fill based on distance
                normalized = min(1.0, distance / (car.raycast_length if hasattr(car, 'raycast_length') else 100))
                fill_width = int(60 * (1 - normalized))  # Closer = more filled
                fill_rect = (rect[0], bar_y, fill_width, 8)
                fill_color = self.colors['error'] if normalized < 0.3 else self.colors['warning'] if normalized < 0.7 else self.colors['success']
                self.draw_rounded_rect(surface, fill_color, fill_rect, radius=2)

                # Distance text
                dist_text = f"{distance:.0f}"
                dist_surf = self.font_mono.render(dist_text, True, self.colors['text_dim'])
                surface.blit(dist_surf, (rect[0] + 65, bar_y - 1))

        # Path information
        y_offset += 110
        path_info = [
            ("Waypoints", len(car.path_waypoints) if hasattr(car, 'path_waypoints') else 0),
            ("Current WP", car.current_waypoint_index if hasattr(car, 'current_waypoint_index') else 0),
            ("Time Alive", car.time_alive if hasattr(car, 'time_alive') else 0)
        ]

        for label, value in path_info:
            info_surf = self.font_small.render(f"{label}: {value}", True, self.colors['text_primary'])
            surface.blit(info_surf, (rect[0], y_offset))
            y_offset += 16

        # Additional debug information
        y_offset += 10
        debug_info = [
            ("Position", f"({car.x:.0f}, {car.y:.0f})"),
            ("Angle", f"{car.angle:.1f}°"),
            ("Speed", f"{car.velocity:.2f}" if hasattr(car, 'velocity') else "N/A"),
            ("Crashed", "Yes" if car.crashed else "No")
        ]

        for label, value in debug_info:
            info_surf = self.font_small.render(f"{label}: {value}", True, self.colors['text_primary'])
            surface.blit(info_surf, (rect[0], y_offset))
            y_offset += 16

    def draw_controls_panel(self, surface, rect, mode):
        """Draw control information"""
        y_offset = rect[1]

        # Controls
        controls = [
            "SPACE: Toggle pause",
            "R: Reset simulation",
            "C: Toggle camera follow",
            "ESC: Quit"
        ]

        for control in controls:
            ctrl_surf = self.font_small.render(control, True, self.colors['text_secondary'])
            surface.blit(ctrl_surf, (rect[0], y_offset))
            y_offset += 16

    def draw_minimap(self, surface, rect, car, camera, road_system):
        """Draw a mini-map showing car position and route"""
        # Mini-map background
        self.draw_rounded_rect(surface, self.colors['bg_dark'], rect, border_color=self.colors['border'])

        # Title
        title_surf = self.font_small.render("Mini-Map", True, self.colors['text_primary'])
        surface.blit(title_surf, (rect[0] + 8, rect[1] + 4))

        # Mini-map area
        map_rect = (rect[0] + 4, rect[1] + 20, rect[2] - 8, rect[3] - 24)
        map_center = (map_rect[0] + map_rect[2]//2, map_rect[1] + map_rect[3]//2)

        # Draw simplified road network
        scale = 0.02  # World to mini-map scale
        if hasattr(road_system, 'road_segments') and road_system.road_segments:
            for segment in road_system.road_segments[:50]:  # Limit for performance
                start_x = map_center[0] + (segment.start[0] - car.x) * scale
                start_y = map_center[1] + (segment.start[1] - car.y) * scale
                end_x = map_center[0] + (segment.end[0] - car.x) * scale
                end_y = map_center[1] + (segment.end[1] - car.y) * scale

                # Only draw if within mini-map bounds
                if (map_rect[0] <= start_x <= map_rect[0] + map_rect[2] and
                    map_rect[1] <= start_y <= map_rect[1] + map_rect[3] and
                    map_rect[0] <= end_x <= map_rect[0] + map_rect[2] and
                    map_rect[1] <= end_y <= map_rect[1] + map_rect[3]):
                    pygame.draw.line(surface, self.colors['text_dim'], (start_x, start_y), (end_x, end_y), 1)
        else:
            # Fallback: draw a simple grid if no road segments
            for i in range(-5, 6):
                grid_x = map_center[0] + i * 20
                grid_y = map_center[1] + i * 20
                if map_rect[0] <= grid_x <= map_rect[0] + map_rect[2]:
                    pygame.draw.line(surface, self.colors['text_dim'], (grid_x, map_rect[1]), (grid_x, map_rect[1] + map_rect[3]), 1)
                if map_rect[1] <= grid_y <= map_rect[1] + map_rect[3]:
                    pygame.draw.line(surface, self.colors['text_dim'], (map_rect[0], grid_y), (map_rect[0] + map_rect[2], grid_y), 1)

        # Draw car position
        pygame.draw.circle(surface, self.colors['accent'], map_center, 3)
        pygame.draw.circle(surface, self.colors['text_primary'], map_center, 3, 1)

        # Draw waypoints
        if hasattr(car, 'path_waypoints') and car.path_waypoints:
            for i, wp in enumerate(car.path_waypoints):
                if i < 10:  # Limit waypoints shown
                    wp_x = map_center[0] + (wp[0] - car.x) * scale
                    wp_y = map_center[1] + (wp[1] - car.y) * scale
                    if (map_rect[0] <= wp_x <= map_rect[0] + map_rect[2] and
                        map_rect[1] <= wp_y <= map_rect[1] + map_rect[3]):
                        color = self.colors['success'] if i > car.current_waypoint_index else self.colors['warning']
                        pygame.draw.circle(surface, color, (wp_x, wp_y), 2)

    def draw_ai_network_simple(self, surface, rect, car):
        """Draw a representative neural network visualization showing actual activations"""
        # Network background
        self.draw_rounded_rect(surface, self.colors['bg_medium'], rect, border_color=self.colors['border_highlight'])

        # Title
        title_surf = self.font_small.render("Neural Network", True, self.colors['text_primary'])
        surface.blit(title_surf, (rect[0] + 8, rect[1] + 4))

        center_y = rect[1] + rect[3]//2

        # Input layer (11 neurons for raycasts, speed, direction, road deviation)
        input_x = rect[0] + 15
        input_spacing = min(12, (rect[3] - 20) // 11)

        # Get actual input values if available
        input_values = []
        if hasattr(car, 'raycast_distances') and len(car.raycast_distances) >= 8:
            # Raycast distances (normalized)
            for dist in car.raycast_distances[:8]:
                input_values.append(min(1.0, dist / car.raycast_length))
            # Speed
            input_values.append(min(1.0, abs(car.speed) / 5.0))
            # Target direction
            target_dir = car.get_target_direction() / 180.0 if hasattr(car, 'get_target_direction') else 0
            input_values.append(target_dir)
            # Road center deviation
            road_dev = 0.0
            if hasattr(car, 'road_system') and car.road_system:
                road_dev = car.road_system.get_normalized_center_deviation(car.x, car.y)
            input_values.append(road_dev)
        else:
            # Default values if no data available
            input_values = [0.5] * 11

        # Draw input neurons
        for i in range(11):
            y = rect[1] + 20 + i * input_spacing
            # Color based on activation level
            activation = (input_values[i] + 1) / 2  # Convert to 0-1 range
            color = self._lerp_color(self.colors['text_dim'], self.colors['accent'], activation)
            pygame.draw.circle(surface, color, (input_x, y), 3)

            # Label first few inputs
            if i < 8:
                label = self.font_mono.render(f"R{i}", True, self.colors['text_dim'])
                surface.blit(label, (input_x - 12, y - 4))

        # Hidden layer (32 neurons)
        hidden_x = rect[0] + rect[2]//2
        hidden_spacing = min(8, (rect[3] - 20) // 16)  # Show 16 neurons for space

        # Simulate hidden layer activations (we can't access internal layers easily)
        # Use a simple pattern based on inputs
        for i in range(16):
            y = rect[1] + 20 + i * hidden_spacing
            # Create a pseudo-activation based on input patterns
            activation = sum(input_values[j] * ((i + j) % 3) for j in range(min(11, i+1))) / (i+1)
            activation = min(1.0, max(0.0, activation))
            color = self._lerp_color(self.colors['text_dim'], self.colors['warning'], activation)
            pygame.draw.circle(surface, color, (hidden_x, y), 3)

        # Output layer (2 neurons: acceleration, steering)
        output_x = rect[0] + rect[2] - 15

        # Get actual output values
        accel_activation = car.ai_acceleration if hasattr(car, 'ai_acceleration') else 0
        steer_activation = car.ai_steering if hasattr(car, 'ai_steering') else 0

        # Acceleration neuron
        accel_y = center_y - 10
        accel_color = self._lerp_color(self.colors['text_dim'], self.colors['success'], (accel_activation + 1) / 2)
        pygame.draw.circle(surface, accel_color, (output_x, accel_y), 4)
        accel_label = self.font_mono.render("A", True, self.colors['text_dim'])
        surface.blit(accel_label, (output_x + 8, accel_y - 4))

        # Steering neuron
        steer_y = center_y + 10
        steer_color = self._lerp_color(self.colors['text_dim'], self.colors['error'], (steer_activation + 1) / 2)
        pygame.draw.circle(surface, steer_color, (output_x, steer_y), 4)
        steer_label = self.font_mono.render("S", True, self.colors['text_dim'])
        surface.blit(steer_label, (output_x + 8, steer_y - 4))

        # Draw some connections to show network structure
        # Input to hidden connections (show a few)
        for i in range(0, 11, 3):  # Every 3rd input
            for j in range(0, 16, 4):  # Every 4th hidden
                start_y = rect[1] + 20 + i * input_spacing
                end_y = rect[1] + 20 + j * hidden_spacing
                pygame.draw.line(surface, self.colors['text_dim'],
                               (input_x, start_y), (hidden_x, end_y), 1)

        # Hidden to output connections
        for i in range(0, 16, 4):
            hidden_y = rect[1] + 20 + i * hidden_spacing
            # Connect to both outputs
            pygame.draw.line(surface, self.colors['text_dim'], (hidden_x, hidden_y), (output_x, accel_y), 1)
            pygame.draw.line(surface, self.colors['text_dim'], (hidden_x, hidden_y), (output_x, steer_y), 1)

    def _lerp_color(self, color1, color2, t):
        """Linear interpolation between two colors"""
        return (
            int(color1[0] + (color2[0] - color1[0]) * t),
            int(color1[1] + (color2[1] - color1[1]) * t),
            int(color1[2] + (color2[2] - color1[2]) * t)
        )

    def draw_hud(self, surface, car, generation=0, best_fitness=0, avg_fitness=0, cars_alive=0, mode=0, road_system=None):
        """Draw the complete modern HUD"""
        # AI Status Panel
        self.draw_panel(surface, "AI Status", self.panels['ai_status'], lambda s, r: self.draw_ai_status_panel(s, r, car))

        # Performance Panel
        self.draw_panel(surface, "Performance", self.panels['performance'], lambda s, r: self.draw_performance_panel(s, r, generation, best_fitness, avg_fitness, cars_alive))

        # Debug Panel
        self.draw_panel(surface, "Debug Info", self.panels['debug'], lambda s, r: self.draw_debug_panel(s, r, car))

        # Mini-Map Panel
        if road_system:
            self.draw_panel(surface, "Mini-Map", self.panels['minimap'], lambda s, r: self.draw_minimap(s, r, car, None, road_system))

        # Neural Network Panel
        self.draw_panel(surface, "Neural Network", self.panels['network'], lambda s, r: self.draw_ai_network_simple(s, r, car))

        # Controls Panel
        self.draw_panel(surface, "Controls", self.panels['controls'], lambda s, r: self.draw_controls_panel(s, r, mode))

# Global HUD instance
hud = ModernHUD()

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
road_system = OSMRoadSystem(center_lat=-36.8825825, center_lon=174.9143453, radius=1600)

print("Setting up pathfinding...")
pathfinder = DijkstraPathfinder(road_system)

print(f"=== MODE {mode} SELECTED ===")
if mode == 0:
    print("Training Mode: AI cars will evolve and learn to drive")
elif mode == 1:
    print("Destination Mode: Click two points to see the best AI drive")

camera = Camera(0, 0, WIDTH, HEIGHT)
road_bounds = road_system.get_road_bounds()
camera.set_bounds(*road_bounds)

if mode == 0:  # Training Mode
    print("=== TRAINING MODE ===")
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

    # Assign a shared destination for fair training
    assign_shared_destination(cars, pathfinder)
    print("Cars initialized with shared destination for training...")

    evolution_timer = 0
    evolution_interval = 40 * FPS  # 40 seconds per generation

    print("Starting AI cars with individual pathfinding evolution...")

elif mode == 1:  # Destination Mode
    print("=== DESTINATION MODE ===")
    print("Click two points on the map: first for start position, then for destination")
    cars = []
    generation = 0
    evolution_timer = 0

running = True
while running:
    screen.fill((34, 139, 34))  # Dark green background

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if mode == 0:  # Only save in training mode
                print("Saving population before exit...")
                save_population(cars, generation)
            running = False
        elif event.type == pygame.KEYDOWN:
            # R key functionality disabled
            pass  # R key does nothing now
        elif event.type == pygame.MOUSEBUTTONDOWN and mode == 1:
            # Destination mode mouse handling
            if event.button == 1:  # Left click
                mouse_pos = pygame.mouse.get_pos()
                world_pos = camera.screen_to_world(mouse_pos[0], mouse_pos[1])
                
                if destination_mode_state == "waiting":
                    destination_mode_start = world_pos
                    destination_mode_state = "selecting_start"
                    print(f"Start position selected: {world_pos}")
                elif destination_mode_state == "selecting_start":
                    destination_mode_end = world_pos
                    destination_mode_state = "selecting_end"
                    print(f"End position selected: {world_pos}")
                    
                    # Create the best car for this route
                    car = create_destination_mode_car(destination_mode_start, destination_mode_end, road_system, pathfinder)
                    cars = [car]
                    destination_mode_car = car
                    destination_mode_state = "driving"
                    print("Car created and ready to drive!")
                elif destination_mode_state == "driving":
                    # Reset for new route
                    destination_mode_start = world_pos
                    destination_mode_end = None
                    destination_mode_car = None
                    cars = []
                    destination_mode_state = "selecting_start"
                    print(f"New start position selected: {world_pos}")

    keys = pygame.key.get_pressed()
    
    if mode == 0:  # Training mode logic
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
                        # Removed ai_predicted_path calculation (simplified visualization)
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
        
        # Draw modern HUD for active (moving) best car
        if active_best_car:
            # Calculate performance metrics
            current_generation = len(generation_history) if generation_history else 0
            best_fitness_val = max(fitness_history) if fitness_history else 0
            avg_fitness_val = sum(list(fitness_history)[-10:]) / len(list(fitness_history)[-10:]) if fitness_history else 0
            cars_alive_count = sum(1 for car in cars if not car.crashed) if 'cars' in locals() else 0

            hud.draw_hud(screen, active_best_car, current_generation, best_fitness_val, avg_fitness_val, cars_alive_count, mode, road_system)
        
        # Draw performance graph in bottom left corner
        graph_x = 20
        graph_y = HEIGHT - 250
        draw_performance_graph(screen, graph_x, graph_y)

    elif mode == 1:  # Destination mode logic
        camera.update(keys)
        
        # Draw roads
        road_system.draw_roads(screen, camera.x, camera.y, WIDTH, HEIGHT, camera.zoom)
        
        # Draw selection indicators
        if destination_mode_start:
            start_screen = camera.world_to_screen(destination_mode_start[0], destination_mode_start[1])
            pygame.draw.circle(screen, GREEN, start_screen, 10)
            pygame.draw.circle(screen, WHITE, start_screen, 10, 2)
        
        if destination_mode_end:
            end_screen = camera.world_to_screen(destination_mode_end[0], destination_mode_end[1])
            pygame.draw.circle(screen, RED, end_screen, 10)
            pygame.draw.circle(screen, WHITE, end_screen, 10, 2)
        
        # Update and draw car if in driving state
        if destination_mode_state == "driving" and destination_mode_car:
            car = destination_mode_car
            car.calculate_predictive_path()
            car.update_path_following_accuracy()
            car.move(keys)
            car.update_raycasts()
            car.draw(camera, True)  # Always visualize the destination mode car
            
            # Draw modern HUD for destination mode car
            hud.draw_hud(screen, car, 0, 0, 0, 1, mode, road_system)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
