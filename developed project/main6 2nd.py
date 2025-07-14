import pygame
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import types
from osm_roads import OSMRoadSystem
from camera import Camera
import heapq
from collections import defaultdict

pygame.init()

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
FPS = 60

# Raycast configuration
NUM_RAYCASTS = 11  # Number of raycasts to generate (must be odd for symmetric distribution)
RAYCAST_SPREAD = 180  # Total angle spread in degrees (e.g., 180 for -90 to +90)

# Enhanced road following constants
LOOK_AHEAD_DISTANCE = 3  # Number of waypoints to look ahead
INTERSECTION_DETECTION_RADIUS = 200
ROAD_FOLLOWING_STRENGTH = 0.3
TURN_PREDICTION_DISTANCE = 100
ADAPTIVE_RAYCAST_ENABLED = False

def set_raycast_config(num_rays, spread_degrees=180):
    """
    Change the raycast configuration. 
    
    Args:
        num_rays: Number of raycasts (will be made odd for symmetry)
        spread_degrees: Total angle spread in degrees
    
    Note: Call this before creating any Car objects to take effect.
    
    Examples:
        set_raycast_config(5, 90)    # 5 rays spanning 90° (-45° to +45°)
        set_raycast_config(9, 180)   # 9 rays spanning 180° (-90° to +90°)
        set_raycast_config(11, 270)  # 11 rays spanning 270° (-135° to +135°)
    """
    global NUM_RAYCASTS, RAYCAST_SPREAD
    # Ensure odd number for symmetric distribution
    NUM_RAYCASTS = num_rays if num_rays % 2 == 1 else num_rays + 1
    RAYCAST_SPREAD = spread_degrees
    print(f"Raycast config updated: {NUM_RAYCASTS} rays with {RAYCAST_SPREAD}° spread")

def set_advanced_raycast_config():
    """Configure advanced raycast system for better road following"""
    global NUM_RAYCASTS, RAYCAST_SPREAD, ADAPTIVE_RAYCAST_ENABLED
    NUM_RAYCASTS = 15  # Reduced for better performance
    RAYCAST_SPREAD = 120  # Narrower spread focused forward
    ADAPTIVE_RAYCAST_ENABLED = True
    print(f"Advanced raycast config: {NUM_RAYCASTS} rays with {RAYCAST_SPREAD}° spread (forward-focused)")

# Enable advanced raycasts for better turn detection
set_advanced_raycast_config()

class AdvancedCarAI(nn.Module):
    def __init__(self, input_size=24, hidden_size=256, output_size=6):
        super(AdvancedCarAI, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size),
            nn.Tanh()
        )
        
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        return self.network(x)

class AStarPathfinder:
    def __init__(self, road_system):
        self.road_system = road_system
        self.node_graph = self._build_node_graph()
        
    def _build_node_graph(self):
        nodes = {}
        connections = defaultdict(list)
        
        node_id = 0
        position_tolerance = 20
        
        for segment in self.road_system.road_segments:
            for point in [segment.start, segment.end]:
                existing_node = None
                for pos, nid in nodes.items():
                    if math.sqrt((pos[0] - point[0])**2 + (pos[1] - point[1])**2) < position_tolerance:
                        existing_node = nid
                        break
                
                if existing_node is None:
                    nodes[point] = node_id
                    node_id += 1
        
        for segment in self.road_system.road_segments:
            start_node = None
            end_node = None
            
            for pos, nid in nodes.items():
                if math.sqrt((pos[0] - segment.start[0])**2 + (pos[1] - segment.start[1])**2) < position_tolerance:
                    start_node = nid
                if math.sqrt((pos[0] - segment.end[0])**2 + (pos[1] - segment.end[1])**2) < position_tolerance:
                    end_node = nid
            
            if start_node is not None and end_node is not None and start_node != end_node:
                connections[start_node].append(end_node)
                connections[end_node].append(start_node)
        
        node_positions = {}
        for pos, nid in nodes.items():
            node_positions[nid] = pos
        
        return {
            'positions': node_positions,
            'connections': dict(connections),
            'nodes_by_position': nodes
        }
    
    def get_nearest_node(self, x, y):
        if not self.node_graph['positions']:
            return None
        
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, pos in self.node_graph['positions'].items():
            distance = math.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node
    
    def find_path(self, start_x, start_y, target_x, target_y):
        start_node = self.get_nearest_node(start_x, start_y)
        target_node = self.get_nearest_node(target_x, target_y)
        
        if start_node is None or target_node is None or start_node == target_node:
            return []
        
        open_set = [(0, start_node)]
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, target_node)}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == target_node:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_node)
                return list(reversed(path))
            
            for neighbor in self.node_graph['connections'].get(current, []):
                tentative_g = g_score[current] + self._distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, target_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []
    
    def _heuristic(self, node1, node2):
        pos1 = self.node_graph['positions'][node1]
        pos2 = self.node_graph['positions'][node2]
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _distance(self, node1, node2):
        return self._heuristic(node1, node2)

class CarAI(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, output_size=4):
        super(CarAI, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Tanh()
        )
    
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
        
        self.raycast_length = 120        
        self.raycast_angles = self._generate_raycast_angles(NUM_RAYCASTS, RAYCAST_SPREAD)
        self.raycast_distances = []
        
        self.short_raycast_length = 40
        self.short_raycast_angles = [-45, 0, 45]
        self.short_raycast_distances = []
        
        self.use_ai = use_ai
        self.fitness = 0
        self.time_alive = 0
        self.distance_traveled = 0
        self.crashed = False
        self.road_system = road_system
        self.last_valid_position = (x, y)
        self.off_road_time = 0
        self.max_off_road_time = 0
        
        self.intersection_detected = False
        self.last_intersection_decision = None
        self.frames_since_intersection = 0
        self.intersection_cooldown = 120  # Doubled cooldown time
        
        # Add steering smoothing to prevent erratic behavior
        self.steering_history = []
        self.max_steering_history = 5
        self.steering_smoothing = 0.7  # How much to smooth steering (0 = no smoothing, 1 = max smoothing)
        
        self.predicted_path_length = 80
        self.predicted_direction = 0
        self.path_curvature = 0
        
        self.position_history = [(x, y)]
        self.max_history_length = 30
        self.movement_smoothness = 0
        
        self.stationary_timer = 0
        self.stationary_threshold = 2 * FPS
        self.min_movement_distance = 40.0
        self.last_position = (x, y)
        self.movement_check_interval = 30
        self.frames_since_movement_check = 0        
        self.pathfinder = pathfinder
        self.current_path = []
        self.path_index = 0
        self.next_waypoint = None
        self.distance_to_target = float('inf')
        self.path_following_accuracy = 0
        self.reached_target_bonus = 0
        self.checkpoint_reached_bonus = 0
        self.use_advanced_ai = True
        
        if self.use_ai:
            if self.use_advanced_ai:
                # Calculate input size dynamically based on number of raycasts
                # raycasts + speed + intersection + target_dir + lookahead_dir + target_dist + path_progress + pathfinding_features(12)
                ai_input_size = NUM_RAYCASTS + 18  # Added 1 for lookahead direction
                self.ai = AdvancedCarAI(input_size=ai_input_size, hidden_size=256, output_size=6)
            else:
                self.ai = CarAI()
            self.randomize_weights()
            self._update_pathfinding()
    
    def _generate_raycast_angles(self, num_rays, spread_degrees):
        """Generate forward-focused raycast angles with higher density in front"""
        if num_rays == 1:
            return [0]
        
        # Ensure we have an odd number for symmetric distribution
        if num_rays % 2 == 0:
            num_rays += 1
        
        angles = []
        
        # Forward-focused distribution: more rays in the center, fewer at sides
        center_rays = max(5, num_rays // 2)  # At least 5 rays focused forward
        side_rays = (num_rays - center_rays) // 2
        
        # Center rays: dense coverage in front (-30° to +30°)
        forward_spread = 60  # Narrow forward focus
        for i in range(center_rays):
            if center_rays == 1:
                angle = 0
            else:
                angle = -forward_spread/2 + (i * forward_spread) / (center_rays - 1)
            angles.append(angle)
        
        # Side rays: sparse coverage for peripheral awareness
        if side_rays > 0:
            # Left side rays
            left_start = -spread_degrees/2
            left_end = -forward_spread/2
            for i in range(side_rays):
                if side_rays == 1:
                    angle = (left_start + left_end) / 2
                else:
                    angle = left_start + (i * (left_end - left_start)) / (side_rays - 1)
                angles.insert(0, angle)  # Insert at beginning
            
            # Right side rays
            right_start = forward_spread/2
            right_end = spread_degrees/2
            for i in range(side_rays):
                if side_rays == 1:
                    angle = (right_start + right_end) / 2
                else:
                    angle = right_start + (i * (right_end - right_start)) / (side_rays - 1)
                angles.append(angle)
        
        # Sort angles to maintain left-to-right order
        angles.sort()
        
        return angles

    def draw(self, camera, is_best=False):
        screen_x, screen_y = camera.world_to_screen(self.x, self.y)
        
        if (screen_x < -50 or screen_x > camera.screen_width + 50 or 
            screen_y < -50 or screen_y > camera.screen_height + 50):
            return
        
        car_surface = pygame.Surface((self.width * camera.zoom, self.height * camera.zoom), pygame.SRCALPHA)
        
        if is_best:
            car_surface.fill((255, 0, 0))
        else:
            car_surface.fill(self.color)
        
        rotated_surface = pygame.transform.rotate(car_surface, -self.angle)
        rotated_rect = rotated_surface.get_rect(center=(screen_x, screen_y))
        screen.blit(rotated_surface, rotated_rect.topleft)
        
        if is_best:
            self.draw_predicted_path(camera)
            self.draw_raycasts(camera)
    
    def get_road_angle_at_position(self, x, y):
        """Get the angle of the road at the given position"""
        if not self.road_system:
            return 0
        
        # Find the nearest road segment
        min_distance = float('inf')
        nearest_segment = None
        
        for segment in self.road_system.road_segments:
            # Calculate distance from point to line segment
            sx, sy = segment.start
            ex, ey = segment.end
            
            # Vector from start to end
            dx = ex - sx
            dy = ey - sy
            
            if dx == 0 and dy == 0:
                # Zero length segment
                distance = math.sqrt((x - sx)**2 + (y - sy)**2)
            else:
                # Calculate projection
                t = max(0, min(1, ((x - sx) * dx + (y - sy) * dy) / (dx * dx + dy * dy)))
                projection_x = sx + t * dx
                projection_y = sy + t * dy
                distance = math.sqrt((x - projection_x)**2 + (y - projection_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_segment = segment
        
        if nearest_segment and min_distance < 50:  # Within reasonable distance
            dx = nearest_segment.end[0] - nearest_segment.start[0]
            dy = nearest_segment.end[1] - nearest_segment.start[1]
            if dx != 0 or dy != 0:
                return math.degrees(math.atan2(dx, -dy))
        
        return 0

    def move(self, keys):
        if self.use_ai:
            self.ai_move()
        else:
            if keys[pygame.K_UP]:
                self.speed += 0.2
            if keys[pygame.K_DOWN]:
                self.speed -= 0.3  # Braking is stronger
            if keys[pygame.K_LEFT]:
                self.angle -= 5
            if keys[pygame.K_RIGHT]:
                self.angle += 5
            # Prevent going backwards
            if self.speed < 0:
                self.speed = 0

        prev_x, prev_y = self.x, self.y
        self.x += math.sin(math.radians(self.angle)) * self.speed
        self.y -= math.cos(math.radians(self.angle)) * self.speed
        
        # Gradually align car with road angle when on road
        if self.road_system and self.road_system.is_point_on_road(self.x, self.y):
            road_angle = self.get_road_angle_at_position(self.x, self.y)
            if road_angle != 0:
                # Calculate angle difference
                angle_diff = road_angle - self.angle
                # Normalize angle difference to [-180, 180]
                while angle_diff > 180:
                    angle_diff -= 360
                while angle_diff < -180:
                    angle_diff += 360
                
                # Gradually adjust toward road angle (stronger alignment when moving slowly)
                alignment_strength = 0.15 if abs(self.speed) > 1.0 else 0.3
                self.angle += angle_diff * alignment_strength
        
        # Track when car enters a new node
        if self.pathfinder:
            prev_node = self.last_node if hasattr(self, 'last_node') else None
            current_node = self.pathfinder.get_nearest_node(self.x, self.y)
            if current_node != prev_node:
                self.last_node = current_node
                # You can add logic here if you want to reward or log node changes
        if self.use_ai:
            distance = math.sqrt((self.x - prev_x)**2 + (self.y - prev_y)**2)
            self.distance_traveled += distance
            self.time_alive += 1

        self.speed *= 0.95
        # Prevent going backwards (AI)
        if self.speed < 0:
            self.speed = 0

        # Stationary logic: only kill if speed is very low for a long time
        if self.use_ai and self.time_alive > self.stationary_threshold:
            self.frames_since_movement_check += 1
            if self.frames_since_movement_check >= self.movement_check_interval:
                # Use speed instead of distance for stationary check
                if abs(self.speed) < 0.12:
                    self.stationary_timer += self.movement_check_interval
                    if self.stationary_timer >= 3 * self.stationary_threshold:  # Must be nearly stopped for a long time
                        self.crashed = True
                else:
                    self.stationary_timer = 0
                self.last_position = (self.x, self.y)
                self.frames_since_movement_check = 0

        if self.road_system:
            if self.road_system.is_point_on_road(self.x, self.y):
                self.last_valid_position = (self.x, self.y)
                self.off_road_time = 0
            else:
                self.off_road_time += 1
                if self.off_road_time > self.max_off_road_time:
                    self.crashed = True

        self.frames_since_movement_check += 1
        if self.frames_since_movement_check >= self.movement_check_interval:
            self.check_stationary()
            self.frames_since_movement_check = 0

    def check_stationary(self):
        distance_moved = math.sqrt((self.x - self.last_position[0]) ** 2 + (self.y - self.last_position[1]) ** 2)
        if distance_moved < self.min_movement_distance:
            self.stationary_timer += self.movement_check_interval
        else:
            self.stationary_timer = 0
        self.last_position = (self.x, self.y)
        
        if self.stationary_timer >= self.stationary_threshold:
            self.crashed = True

    def update_raycasts(self):
        """Enhanced raycast system with adaptive lengths and multi-layer detection"""
        if not self.road_system:
            self.raycast_distances = [self.raycast_length] * len(self.raycast_angles)
            self.short_raycast_distances = [self.short_raycast_length] * len(self.short_raycast_angles)
            return
        
        # Adaptive raycast length based on speed and road type
        base_length = self.raycast_length
        if self.intersection_detected:
            base_length *= 0.7  # Shorter at intersections
        elif abs(self.speed) > 3.0:
            base_length *= 1.3  # Longer at high speed
        
        self.raycast_distances = []
        for i, angle_offset in enumerate(self.raycast_angles):
            ray_angle = self.angle + angle_offset
            
            # Adaptive length per ray - side rays shorter, forward rays longer
            ray_length = base_length
            if abs(angle_offset) > 90:  # Side/rear rays
                ray_length *= 0.6
            elif abs(angle_offset) < 30:  # Forward rays
                ray_length *= 1.2
            
            distance = self.road_system.raycast_to_road_edge(
                self.x, self.y, ray_angle, ray_length
            )
            self.raycast_distances.append(distance)
        
        # Multi-layer short raycasts for obstacle avoidance
        self.short_raycast_distances = []
        short_angles = [-60, -30, 0, 30, 60]  # More comprehensive coverage
        for angle_offset in short_angles:
            ray_angle = self.angle + angle_offset
            distance = self.road_system.raycast_to_road_edge(
                self.x, self.y, ray_angle, self.short_raycast_length
            )
            self.short_raycast_distances.append(distance)
        
        self.detect_advanced_intersections()
        self.detect_road_curvature()
    
    def detect_advanced_intersections(self):
        """Advanced intersection detection with turn prediction - less sensitive"""
        if len(self.raycast_distances) < 5:
            return
        
        # Multi-criteria intersection detection with higher thresholds
        num_rays = len(self.raycast_distances)
        quarter = num_rays // 4
        center_idx = num_rays // 2
        
        # More conservative opening detection - need more clear rays
        left_opening = sum(1 for i in range(quarter) if self.raycast_distances[i] > self.raycast_length * 0.8) > 3
        right_opening = sum(1 for i in range(num_rays - quarter, num_rays) if self.raycast_distances[i] > self.raycast_length * 0.8) > 3
        forward_clear = self.raycast_distances[center_idx] > self.raycast_length * 0.7
        
        # More conservative intersection detection
        side_openings = left_opening and right_opening  # Require both sides open
        major_intersection = side_openings and forward_clear
        
        self.frames_since_intersection += 1
        
        # Only detect major, obvious intersections
        intersection_conditions = [
            major_intersection,  # Only major cross intersections
            self._detect_major_road_change()  # Only significant road changes
        ]
        
        # Longer cooldown and stricter requirements
        if (self.frames_since_intersection > self.intersection_cooldown * 2 and
            any(intersection_conditions)):
            
            if not self.intersection_detected:
                self.intersection_detected = True
                self.frames_since_intersection = 0
                self.make_smart_intersection_decision()
        else:
            self.intersection_detected = False
    
    def _detect_major_road_change(self):
        """Detect only major road width changes - more conservative"""
        if len(self.raycast_distances) < 9:
            return False
        
        # Look for dramatic changes in road width
        num_rays = len(self.raycast_distances)
        left_distances = self.raycast_distances[:num_rays//3]
        right_distances = self.raycast_distances[2*num_rays//3:]
        
        avg_left = sum(left_distances) / len(left_distances)
        avg_right = sum(right_distances) / len(right_distances)
        
        # Only trigger on very significant asymmetry (doubled threshold)
        return abs(avg_left - avg_right) > self.raycast_length * 0.8
    
    def _detect_road_width_change(self):
        """Detect changes in road width indicating intersections - more conservative"""
        if len(self.raycast_distances) < 7:
            return False
        
        # Compare left and right ray distances with higher threshold
        num_rays = len(self.raycast_distances)
        left_distances = self.raycast_distances[:num_rays//3]
        right_distances = self.raycast_distances[2*num_rays//3:]
        
        avg_left = sum(left_distances) / len(left_distances)
        avg_right = sum(right_distances) / len(right_distances)
        
        # Much more conservative threshold - only major asymmetry
        return abs(avg_left - avg_right) > self.raycast_length * 0.6

    def detect_road_curvature(self):
        """Detect upcoming road curves to adjust steering"""
        if len(self.raycast_distances) < 7:
            return
        
        num_rays = len(self.raycast_distances)
        center_idx = num_rays // 2
        
        # Analyze ray pattern to detect curves
        left_rays = self.raycast_distances[:center_idx]
        right_rays = self.raycast_distances[center_idx+1:]
        
        left_avg = sum(left_rays) / len(left_rays)
        right_avg = sum(right_rays) / len(right_rays)
        
        # Curve detection based on asymmetric ray patterns
        curve_threshold = self.raycast_length * 0.3
        if abs(left_avg - right_avg) > curve_threshold:
            if left_avg > right_avg:
                self.predicted_direction = -15  # Curve left
            else:
                self.predicted_direction = 15   # Curve right
        else:
            self.predicted_direction *= 0.8  # Decay prediction
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
        if len(self.raycast_distances) < NUM_RAYCASTS or len(self.short_raycast_distances) < 3:
            return
        
        if self.time_alive % 60 == 0:
            self._update_pathfinding()
        
        self.update_movement_history()
        
        if self.use_advanced_ai:
            inputs = []
            
            normalized_distances = [d / self.raycast_length for d in self.raycast_distances]
            inputs.extend(normalized_distances)
            
            normalized_speed = min(1.0, abs(self.speed) / 5.0)
            inputs.append(normalized_speed)
            inputs.append(1.0 if self.intersection_detected else 0.0)
            
            # Enhanced target direction calculation with lookahead
            target_direction = 0.0
            lookahead_direction = 0.0
            
            if self.next_waypoint:
                dx = self.next_waypoint[0] - self.x
                dy = self.next_waypoint[1] - self.y
                target_angle = math.degrees(math.atan2(dx, -dy))
                angle_diff = target_angle - self.angle
                while angle_diff > 180: angle_diff -= 360
                while angle_diff < -180: angle_diff += 360
                target_direction = angle_diff / 180.0
                
                # Look ahead to next waypoint for better path following
                if self.current_path and self.path_index + 1 < len(self.current_path):
                    next_node_id = self.current_path[self.path_index + 1]
                    next_pos = self.pathfinder.node_graph['positions'].get(next_node_id)
                    if next_pos:
                        lookahead_dx = next_pos[0] - self.x
                        lookahead_dy = next_pos[1] - self.y
                        lookahead_angle = math.degrees(math.atan2(lookahead_dx, -lookahead_dy))
                        lookahead_diff = lookahead_angle - self.angle
                        while lookahead_diff > 180: lookahead_diff -= 360
                        while lookahead_diff < -180: lookahead_diff += 360
                        lookahead_direction = lookahead_diff / 180.0
            
            inputs.append(target_direction)
            inputs.append(lookahead_direction)  # New lookahead input
            
            normalized_target_distance = min(1.0, self.distance_to_target / 500.0)
            inputs.append(normalized_target_distance)
            
            path_progress = 0.0
            if self.current_path:
                path_progress = self.path_index / max(1, len(self.current_path) - 1)
            inputs.append(path_progress)
            
            pathfinding_features = self._get_pathfinding_features()
            inputs.extend(pathfinding_features)
            
            input_tensor = torch.tensor(inputs, dtype=torch.float32)
            
            with torch.no_grad():
                outputs = self.ai(input_tensor)
            
            acceleration = outputs[0].item()
            turn_direction = outputs[1].item()
            self.predicted_direction = outputs[2].item() * 45
            self.path_curvature = outputs[3].item()
            
            path_following_weight = torch.sigmoid(outputs[4]).item()
            exploration_bias = outputs[5].item()
            
            # Stronger path following with lookahead
            path_following_weight = min(1.0, path_following_weight * 2.0)
            if self.next_waypoint and path_following_weight > 0.15:
                dx = self.next_waypoint[0] - self.x
                dy = self.next_waypoint[1] - self.y
                target_angle = math.degrees(math.atan2(dx, -dy))
                angle_diff = target_angle - self.angle
                while angle_diff > 180: angle_diff -= 360
                while angle_diff < -180: angle_diff += 360
                
                pathfinding_turn = max(-1.0, min(1.0, angle_diff / 25.0))  # More sensitive turning
                
                # Blend in lookahead direction for smoother curves
                if lookahead_direction != 0.0:
                    pathfinding_turn = (pathfinding_turn * 0.7 + lookahead_direction * 0.3)
                
                turn_direction = (turn_direction * (1 - path_following_weight) + 
                                pathfinding_turn * path_following_weight)
                
                # Speed up when aligned with path
                if abs(angle_diff) < 20:
                    acceleration = max(acceleration, 0.3)
                elif abs(angle_diff) < 45:
                    acceleration = max(acceleration, 0.1)
        
        else:
            # ...existing code...
            inputs = []
            
            normalized_distances = [d / self.raycast_length for d in self.raycast_distances]
            inputs.extend(normalized_distances)
            
            normalized_short_distances = [d / self.short_raycast_length for d in self.short_raycast_distances]
            inputs.extend(normalized_short_distances)
            
            normalized_speed = min(1.0, abs(self.speed) / 5.0)
            inputs.append(normalized_speed)
            
            inputs.append(1.0 if self.intersection_detected else 0.0)
            
            input_tensor = torch.tensor(inputs, dtype=torch.float32)
            
            with torch.no_grad():
                outputs = self.ai(input_tensor)
            
            acceleration = outputs[0].item()
            turn_direction = outputs[1].item()
            self.predicted_direction = outputs[2].item() * 45
            self.path_curvature = outputs[3].item()
        
        # More gentle intersection handling - reduce crazy steering
        if self.intersection_detected and self.last_intersection_decision:
            decision_strength = 0.3  # Much weaker influence to prevent crazy behavior
            if self.last_intersection_decision == "left":
                turn_direction = min(1.0, turn_direction - decision_strength)
            elif self.last_intersection_decision == "right":
                turn_direction = max(-1.0, turn_direction + decision_strength)
            # No special handling for "forward" - let AI decide naturally
        
        # Enhanced obstacle avoidance with better short raycast handling
        if len(self.short_raycast_distances) >= 5:  # Updated for 5 short rays
            obstacles = [d < self.short_raycast_length * 0.5 for d in self.short_raycast_distances]
            far_left, left, front, right, far_right = obstacles
            
            if front:
                acceleration = min(acceleration, -0.8)  # Emergency braking
                # Choose best escape direction
                if not left and not far_left:
                    turn_direction = min(turn_direction, -0.9)
                elif not right and not far_right:
                    turn_direction = max(turn_direction, 0.9)
                elif not left:
                    turn_direction = min(turn_direction, -0.7)
                elif not right:
                    turn_direction = max(turn_direction, 0.7)
                else:
                    acceleration = -1.0  # Full stop if trapped
            
            # Gradual avoidance for side obstacles
            if left and not front:
                turn_direction += 0.4
            if right and not front:
                turn_direction -= 0.4
            if far_left and not left and not front:
                turn_direction += 0.2
            if far_right and not right and not front:
                turn_direction -= 0.2
        
        # Adaptive speed and turn scaling
        intersection_factor = 0.8 if self.intersection_detected else 1.0  # Less aggressive slowdown
        speed_factor = max(0.4, 1.0 - abs(self.speed) / 6.0)  # Slower turning at high speed
        
        speed_scale = 0.6 * intersection_factor
        self.speed += acceleration * speed_scale
        
        # Prevent going backwards
        if self.speed < 0:
            self.speed = 0
        
        # Apply steering smoothing to prevent erratic behavior
        self.steering_history.append(turn_direction)
        if len(self.steering_history) > self.max_steering_history:
            self.steering_history.pop(0)
        
        # Smooth steering based on recent history
        if len(self.steering_history) > 1:
            avg_steering = sum(self.steering_history) / len(self.steering_history)
            turn_direction = (turn_direction * (1 - self.steering_smoothing) + 
                            avg_steering * self.steering_smoothing)
            
        turn_scale = 2.5 * speed_factor * intersection_factor  # Reduced turn scale
        self.angle += turn_direction * turn_scale
        
        self.predicted_direction = turn_direction * 20  # Reduced prediction strength
        self.path_curvature = turn_direction * 0.8  # Reduced curvature influence

    def calculate_fitness(self):
        if self.off_road_time > 0:
            off_road_penalty = (self.off_road_time ** 2) * 3.0  # Increased penalty
            off_road_base_penalty = 150
        else:
            off_road_penalty = 0
            off_road_base_penalty = 0

        # Enhanced road following rewards
        road_bonus = self.time_alive * 0.8 if self.off_road_time == 0 else 0
        distance_bonus = self.distance_traveled * 0.4 if self.off_road_time < 10 else 0
        time_bonus = self.time_alive * 0.08

        avg_speed = self.distance_traveled / self.time_alive if self.time_alive > 0 else 0
        optimal_speed = 2.5
        speed_bonus = max(0, optimal_speed - abs(avg_speed - optimal_speed)) * 20
        
        # Enhanced smoothness rewards
        smoothness_bonus = 0
        if self.movement_smoothness > 0:
            smoothness_bonus = max(0, (math.pi - self.movement_smoothness) * 8)
        
        # Intersection intelligence bonus
        intersection_bonus = 0
        if hasattr(self, 'last_intersection_decision') and self.last_intersection_decision:
            intersection_bonus = 35  # Increased reward for smart decisions
            
            # Bonus for path-aligned intersection decisions
            if self.next_waypoint and hasattr(self, 'last_intersection_decision'):
                dx = self.next_waypoint[0] - self.x
                dy = self.next_waypoint[1] - self.y
                target_angle = math.degrees(math.atan2(dx, -dy))
                angle_diff = target_angle - self.angle
                while angle_diff > 180: angle_diff -= 360
                while angle_diff < -180: angle_diff += 360
                
                # Reward decisions that align with path
                if ((abs(angle_diff) < 30 and self.last_intersection_decision == "forward") or
                    (angle_diff < -30 and self.last_intersection_decision == "left") or
                    (angle_diff > 30 and self.last_intersection_decision == "right")):
                    intersection_bonus += 25

        pathfinding_bonus = 0
        target_proximity_bonus = 0
        
        if hasattr(self, 'end_node') and self.end_node and self.pathfinder:
            target_pos = self.pathfinder.node_graph['positions'].get(self.end_node)
            if target_pos:
                current_distance = math.sqrt((self.x - target_pos[0])**2 + (self.y - target_pos[1])**2)
                
                max_possible_distance = 1200
                proximity_bonus = (max_possible_distance - current_distance) / max_possible_distance * 150
                target_proximity_bonus += proximity_bonus
                
                # Progressive target rewards
                if current_distance < 40:
                    target_proximity_bonus += 300
                    self.reached_target_bonus = 300
                elif current_distance < 80:
                    target_proximity_bonus += 200
                    self.reached_target_bonus = 200
                elif current_distance < 150:
                    target_proximity_bonus += 100
                    self.reached_target_bonus = 100
                elif current_distance < 250:
                    target_proximity_bonus += 50
                    self.reached_target_bonus = 50
        
        # Enhanced pathfinding rewards
        if self.current_path and len(self.current_path) > 0:
            path_progress = self.path_index / max(1, len(self.current_path) - 1)
            pathfinding_bonus += path_progress * 200  # Increased reward
            
            pathfinding_bonus += self.path_following_accuracy * 100
            
            # Bonus for staying on optimal path
            pathfinding_bonus += 40
            
            # Reward for waypoint progression
            if hasattr(self, 'waypoints_reached'):
                pathfinding_bonus += self.waypoints_reached * 15

        # Advanced AI usage bonus
        navigation_intelligence_bonus = 0
        if hasattr(self, 'use_advanced_ai') and self.use_advanced_ai:
            navigation_intelligence_bonus += 50
            
            if self.intersection_detected and self.next_waypoint:
                navigation_intelligence_bonus += 30
                
            # Reward for good turn prediction
            if abs(self.predicted_direction) > 5 and self.intersection_detected:
                navigation_intelligence_bonus += 20

        crash_penalty = 400 if self.crashed else 0
        stationary_penalty = self.stationary_timer * 3
        
        # Enhanced exploration rewards
        exploration_bonus = 0
        if hasattr(self, 'position_history') and len(self.position_history) > 15:
            unique_positions = set()
            for pos in self.position_history[-15:]:
                rounded_pos = (round(pos[0] / 25) * 25, round(pos[1] / 25) * 25)
                unique_positions.add(rounded_pos)
            
            exploration_ratio = len(unique_positions) / 15
            exploration_bonus = exploration_ratio * 40
        
        # Consistency bonus - reward cars that perform well over time
        consistency_bonus = 0
        if self.time_alive > 300:  # Long-lived cars
            consistency_bonus = min(50, (self.time_alive - 300) * 0.1)
        
        # Turn efficiency bonus
        turn_efficiency_bonus = 0
        if hasattr(self, 'movement_smoothness') and self.movement_smoothness < math.pi / 3:
            turn_efficiency_bonus = 25
        
        self.fitness = (
            distance_bonus + time_bonus + road_bonus + speed_bonus + 
            smoothness_bonus + intersection_bonus + pathfinding_bonus +
            target_proximity_bonus + navigation_intelligence_bonus + exploration_bonus +
            consistency_bonus + turn_efficiency_bonus -
            off_road_penalty - off_road_base_penalty - crash_penalty - stationary_penalty
        )
        
        return self.fitness
    
    def clone(self):
        new_car = Car(self.x, self.y, self.color, use_ai=True, road_system=self.road_system, 
                     pathfinder=self.pathfinder)
        if self.use_ai:
            new_car.ai.load_state_dict(self.ai.state_dict())
            new_car.use_advanced_ai = self.use_advanced_ai
        return new_car
    
    def mutate(self, mutation_rate=0.15, mutation_strength=0.3):
        if not self.use_ai:
            return
        
        with torch.no_grad():
            for param in self.ai.parameters():
                layer_mutation_rate = mutation_rate
                if param.dim() > 1:
                    layer_mutation_rate *= 1.2
                
                mutation_mask = torch.rand_like(param) < layer_mutation_rate
                mutation = torch.randn_like(param) * mutation_strength
                param.data += mutation_mask.float() * mutation

    def draw_predicted_path(self, camera):
        if self.current_path and self.pathfinder:
            path_points = []
            for i, node_id in enumerate(self.current_path):
                node_pos = self.pathfinder.node_graph['positions'].get(node_id)
                if node_pos:
                    screen_point = camera.world_to_screen(node_pos[0], node_pos[1])
                    path_points.append(screen_point)
            
            if len(path_points) > 1:
                for i in range(len(path_points) - 1):
                    progress = i / (len(path_points) - 1)
                    red = int(0 + progress * 100)
                    green = int(100 + progress * 155)
                    blue = int(255 - progress * 100)
                    color = (red, green, blue)
                    
                    thickness = 4 if i < self.path_index + 2 else 2
                    pygame.draw.line(screen, color, path_points[i], path_points[i + 1], thickness)
                
                if self.path_index < len(path_points):
                    target_screen = path_points[self.path_index]
                    pygame.draw.circle(screen, (255, 255, 0), target_screen, 8, 3)
                    
                if path_points:
                    final_target = path_points[-1]
                    pygame.draw.circle(screen, (255, 0, 255), final_target, 12, 4)
        
        front_offset = self.height / 2
        start_x = self.x + math.sin(math.radians(self.angle)) * front_offset
        start_y = self.y - math.cos(math.radians(self.angle)) * front_offset
        
        predicted_angle = self.angle + self.predicted_direction
        path_length = 50  
        
        num_points = 10  
        immediate_path_points = []
        
        for i in range(num_points + 1):
            t = i / num_points
            
            current_angle = predicted_angle
            if abs(self.path_curvature) > 0.1:
                curve_amount = self.path_curvature * t * t * 20
                current_angle += curve_amount
            
            distance = path_length * t
            point_x = start_x + math.sin(math.radians(current_angle)) * distance
            point_y = start_y - math.cos(math.radians(current_angle)) * distance
        
            if self.road_system and not self.road_system.is_point_on_road(point_x, point_y):
                distance *= 0.7
                point_x = start_x + math.sin(math.radians(current_angle)) * distance
                point_y = start_y - math.cos(math.radians(current_angle)) * distance
        
        if len(immediate_path_points) >= 2:
            end_point = immediate_path_points[-1]
            second_last = immediate_path_points[-2]
            
            dx = end_point[0] - second_last[0]
            dy = end_point[1] - second_last[1]
            
            if dx != 0 or dy != 0:
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx /= length
                    dy /= length
                    
                    arrow_length = 8  
                    arrow_width = 6   
                    
                    tip_x, tip_y = end_point
                    base_x = tip_x - dx * arrow_length
                    base_y = tip_y - dy * arrow_length
                    
                    left_wing_x = base_x - dy * arrow_width
                    left_wing_y = base_y + dx * arrow_width
                    right_wing_x = base_x + dy * arrow_width
                    right_wing_y = base_y - dx * arrow_width
                    
                    arrow_points = [
                        (tip_x, tip_y),
                        (left_wing_x, left_wing_y),
                        (right_wing_x, right_wing_y)
                    ]
                    pygame.draw.polygon(screen, (255, 100, 0), arrow_points)
                    pygame.draw.polygon(screen, (255, 200, 0), arrow_points, 2)

    def make_smart_intersection_decision(self):
        """Enhanced intersection decision making with pathfinding awareness"""
        available_directions = []
        
        if len(self.raycast_distances) >= 5:
            num_rays = len(self.raycast_distances)
            left_idx = 0
            center_idx = num_rays // 2
            right_idx = num_rays - 1
            
            # More sophisticated direction detection
            quarter = num_rays // 4
            
            # Check multiple rays for each direction
            left_clear = sum(1 for i in range(quarter) if self.raycast_distances[i] > self.raycast_length * 0.6) > 1
            right_clear = sum(1 for i in range(num_rays - quarter, num_rays) if self.raycast_distances[i] > self.raycast_length * 0.6) > 1
            forward_clear = self.raycast_distances[center_idx] > self.raycast_length * 0.5
            
            if left_clear:
                available_directions.append("left")
            if right_clear:
                available_directions.append("right")
            if forward_clear:
                available_directions.append("forward")
        
        # Path-aware decision making
        if available_directions and self.next_waypoint:
            # Calculate direction to next waypoint
            dx = self.next_waypoint[0] - self.x
            dy = self.next_waypoint[1] - self.y
            target_angle = math.degrees(math.atan2(dx, -dy))
            angle_diff = target_angle - self.angle
            while angle_diff > 180: angle_diff -= 360
            while angle_diff < -180: angle_diff += 360
            
            # Choose direction that best aligns with path
            if abs(angle_diff) < 45 and "forward" in available_directions:
                self.last_intersection_decision = "forward"
            elif angle_diff < -45 and "left" in available_directions:
                self.last_intersection_decision = "left"
            elif angle_diff > 45 and "right" in available_directions:
                self.last_intersection_decision = "right"
            else:
                # Fallback to random choice from available directions
                self.last_intersection_decision = random.choice(available_directions)
        elif available_directions:
            # Random choice if no pathfinding info
            self.last_intersection_decision = random.choice(available_directions)
        else:
            self.last_intersection_decision = "forward"
    
    def update_movement_history(self):
        self.position_history.append((self.x, self.y))
        
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        if len(self.position_history) >= 3:
            direction_changes = []
            for i in range(len(self.position_history) - 2):
                p1 = self.position_history[i]
                p2 = self.position_history[i + 1]
                p3 = self.position_history[i + 2]
                
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                if (v1[0]**2 + v1[1]**2) > 0 and (v2[0]**2 + v2[1]**2) > 0:
                    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                    
                    cos_angle = dot_product / (mag1 * mag2)
                    cos_angle = max(-1, min(1, cos_angle))  
                    angle = math.acos(cos_angle)
                    direction_changes.append(angle)
            if direction_changes:
                self.movement_smoothness = sum(direction_changes) / len(direction_changes)
            else:
                self.movement_smoothness = 0

    def _update_pathfinding(self):
        if self.pathfinder and hasattr(self, 'end_node') and self.end_node:
            # Use the shared end_node instead of the nearest node
            target_pos = self.pathfinder.node_graph['positions'].get(self.end_node)
            
            if target_pos:
                self.current_path = self.pathfinder.find_path(self.x, self.y, target_pos[0], target_pos[1])
                self.path_index = 0
                self._update_next_waypoint()
                
                if len(self.current_path) > 0:
                    self.last_node = self.current_path[-1]
    
    def _update_next_waypoint(self):
        if self.current_path and self.path_index < len(self.current_path):
            node_id = self.current_path[self.path_index]
            self.next_waypoint = self.pathfinder.node_graph['positions'].get(node_id)
        else:
            self.next_waypoint = None
    
    def _get_pathfinding_features(self):
        features = [0.0] * 12  
        
        if self.next_waypoint:
            dx = self.next_waypoint[0] - self.x
            dy = self.next_waypoint[1] - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            features[0] = min(1.0, distance / 150.0)  # Shorter normalization distance
            
            target_angle = math.degrees(math.atan2(dx, -dy))
            angle_diff = target_angle - self.angle
            while angle_diff > 180: angle_diff -= 360
            while angle_diff < -180: angle_diff += 360
            features[1] = angle_diff / 180.0  
            
            if self.current_path:
                features[2] = self.path_index / max(1, len(self.current_path) - 1)
            
            features[3] = min(1.0, self.distance_to_target / 400.0)  # Adjusted normalization
            
        features[4] = self.path_following_accuracy
        
        remaining_waypoints = len(self.current_path) - self.path_index if self.current_path else 0
        features[5] = min(1.0, remaining_waypoints / 8.0)  # Adjusted for typical path lengths
        
        optimal_speed = 2.8  # Slightly faster optimal speed
        features[6] = min(1.0, abs(self.speed) / optimal_speed)
        
        # Enhanced open directions detection
        open_directions = 0
        if len(self.raycast_distances) >= NUM_RAYCASTS:
            threshold = self.raycast_length * 0.65  # Slightly higher threshold
            for distance in self.raycast_distances:
                if distance > threshold:
                    open_directions += 1
        features[7] = min(1.0, open_directions / max(1, NUM_RAYCASTS))
        
        features[8] = min(1.0, self.time_alive % 240 / 240.0)  # Longer cycle for stability
        
        if self.next_waypoint:
            path_deviation = self._calculate_path_deviation()
            features[9] = min(1.0, path_deviation / 40.0)  # More sensitive to deviation
        
        if self.path_curvature != 0:
            features[10] = min(1.0, abs(self.path_curvature))
        
        # Enhanced target proximity
        if self.distance_to_target < 60:  # Increased proximity threshold
            features[11] = 1.0
        elif self.distance_to_target < 120:
            features[11] = 0.5
        
        return features
    
    def _calculate_path_deviation(self):
        if not self.next_waypoint:
            return 0
        
        dx = self.next_waypoint[0] - self.x
        dy = self.next_waypoint[1] - self.y
        distance_to_waypoint = math.sqrt(dx*dx + dy*dy)
        
        # Closer waypoint threshold for more responsive navigation
        if distance_to_waypoint < 25:  # Reduced from 30
            self.path_index += 1
            self._update_next_waypoint()
            self.path_following_accuracy = min(1.0, self.path_following_accuracy + 0.15)
            
            # Track waypoint progression
            if not hasattr(self, 'waypoints_reached'):
                self.waypoints_reached = 0
            self.waypoints_reached += 1
            
            return 0
        
        return distance_to_waypoint



def evolve_population(cars, population_size=30, road_system=None, pathfinder=None, shared_path=None, start_node=None, end_node=None):
    for car in cars:
        car.calculate_fitness()
    cars.sort(key=lambda x: x.fitness, reverse=True)
    print(f"Generation evolved! Best fitness: {cars[0].fitness:.2f}, Top 5: {[f'{car.fitness:.1f}' for car in cars[:5]]}")
    
    # More elites for better genetic diversity preservation
    elite_count = max(2, population_size // 4)  # Increased elite preservation
    elites = cars[:elite_count]
    new_cars = []
    
    # Calculate initial road angle based on the first path segment
    initial_angle = 0
    if len(shared_path) > 1:
        first_node_pos = pathfinder.node_graph['positions'][shared_path[0]]
        second_node_pos = pathfinder.node_graph['positions'][shared_path[1]]
        dx = second_node_pos[0] - first_node_pos[0]
        dy = second_node_pos[1] - first_node_pos[1]
        initial_angle = math.degrees(math.atan2(dx, -dy))
    
    # Add elites directly (with small mutations to prevent stagnation)
    for i, elite in enumerate(elites):
        new_car = elite.clone()
        spawn_x, spawn_y = pathfinder.node_graph['positions'][start_node]
        new_car.x = spawn_x
        new_car.y = spawn_y
        new_car.angle = initial_angle
        new_car.speed = 0
        new_car.crashed = False
        new_car.fitness = 0
        new_car.time_alive = 0
        new_car.distance_traveled = 0
        new_car.off_road_time = 0
        new_car.stationary_timer = 0
        new_car.last_position = (spawn_x, spawn_y)
        new_car.frames_since_movement_check = 0
        new_car.current_path = shared_path.copy()
        new_car.path_index = 0
        new_car.next_waypoint = pathfinder.node_graph['positions'][shared_path[0]] if shared_path else None
        new_car.end_node = end_node
        new_car.last_node = start_node
        
        # Very light mutation for elites to prevent exact copies
        if i > 0:  # Don't mutate the absolute best
            new_car.mutate(mutation_rate=0.03, mutation_strength=0.1)
        
        new_cars.append(new_car)
    
    # Fitness-weighted parent selection for remaining population
    total_fitness = sum(max(0, car.fitness) for car in cars[:population_size//2])
    
    while len(new_cars) < population_size:
        # Weighted selection based on fitness
        if total_fitness > 0:
            rand_val = random.uniform(0, total_fitness)
            cumulative = 0
            parent_index = 0
            for i, car in enumerate(cars[:population_size//2]):
                cumulative += max(0, car.fitness)
                if cumulative >= rand_val:
                    parent_index = i
                    break
        else:
            parent_index = random.randint(0, min(len(cars) - 1, population_size // 2 - 1))
        
        parent = cars[parent_index]
        offspring = parent.clone()
        
        # Adaptive mutation based on parent performance and generation diversity
        base_mutation_rate = 0.15
        base_mutation_strength = 0.3
        
        # Performance-based mutation adjustment
        if parent.fitness > cars[0].fitness * 0.85:  # Top performers
            mutation_rate = base_mutation_rate * 0.6
            mutation_strength = base_mutation_strength * 0.7
        elif parent.fitness > cars[0].fitness * 0.5:  # Middle performers
            mutation_rate = base_mutation_rate * 1.0
            mutation_strength = base_mutation_strength * 1.0
        else:  # Poor performers
            mutation_rate = base_mutation_rate * 1.5
            mutation_strength = base_mutation_strength * 1.4
        
        # Add some randomness to prevent local optima
        if random.random() < 0.1:  # 10% chance of high mutation
            mutation_rate *= 2.0
            mutation_strength *= 1.5
        
        offspring.mutate(mutation_rate=mutation_rate, mutation_strength=mutation_strength)
        
        # Reset offspring state
        spawn_x, spawn_y = pathfinder.node_graph['positions'][start_node]
        offspring.x = spawn_x
        offspring.y = spawn_y
        offspring.angle = initial_angle
        offspring.speed = 0
        offspring.crashed = False
        offspring.fitness = 0
        offspring.time_alive = 0
        offspring.distance_traveled = 0
        offspring.off_road_time = 0
        offspring.stationary_timer = 0
        offspring.last_position = (spawn_x, spawn_y)
        offspring.frames_since_movement_check = 0
        offspring.current_path = shared_path.copy()
        offspring.path_index = 0
        offspring.next_waypoint = pathfinder.node_graph['positions'][shared_path[0]] if shared_path else None
        offspring.end_node = end_node
        offspring.last_node = start_node
        offspring.color = random.choice([RED, GREEN, BLUE])
        
        # Reset path-following metrics
        offspring.path_following_accuracy = 0
        if hasattr(offspring, 'waypoints_reached'):
            offspring.waypoints_reached = 0
        
        new_cars.append(offspring)
    
    return new_cars

# --- Setup shared start and end node for all cars ---
print("Loading road system...")
road_system = OSMRoadSystem(center_lat=-36.8825825, center_lon=174.9143453, radius=1000)

# Add a custom method to draw roads without intersection highlighting
def draw_roads_with_intersections(self, screen, camera_x, camera_y, screen_width, screen_height, zoom=1.0, pathfinder=None):
    """Draw roads without intersection highlighting"""
    # Just draw normal roads
    self.draw_roads(screen, camera_x, camera_y, screen_width, screen_height, zoom)


# Monkey patch the method to the road_system instance
import types
road_system.draw_roads_with_intersections = types.MethodType(draw_roads_with_intersections, road_system)

print("Building pathfinding graph...")
pathfinder = AStarPathfinder(road_system)

# Pick a random start node (where cars will spawn) and a random end node
all_nodes = list(pathfinder.node_graph['positions'].keys())
start_node = random.choice(all_nodes)
end_node = random.choice([n for n in all_nodes if n != start_node])

start_pos = pathfinder.node_graph['positions'][start_node]
end_pos = pathfinder.node_graph['positions'][end_node]

shared_path = pathfinder.find_path(start_pos[0], start_pos[1], end_pos[0], end_pos[1])

camera = Camera(0, 0, WIDTH, HEIGHT)
road_bounds = road_system.get_road_bounds()
camera.set_bounds(*road_bounds)

population_size = 50
cars = []

# Calculate initial road angle based on the first path segment
initial_angle = 0
if len(shared_path) > 1:
    first_node_pos = pathfinder.node_graph['positions'][shared_path[0]]
    second_node_pos = pathfinder.node_graph['positions'][shared_path[1]]
    dx = second_node_pos[0] - first_node_pos[0]
    dy = second_node_pos[1] - first_node_pos[1]
    initial_angle = math.degrees(math.atan2(dx, -dy))

for i in range(population_size):
    # All cars spawn at the same start node
    spawn_x, spawn_y = start_pos
    spawn_angle = initial_angle
    color = random.choice([RED, GREEN, BLUE])
    car = Car(spawn_x, spawn_y, color, use_ai=True, road_system=road_system, pathfinder=pathfinder)
    car.angle = spawn_angle  # Set the initial angle
    car.current_path = shared_path.copy()
    car.path_index = 0
    car.next_waypoint = pathfinder.node_graph['positions'][shared_path[0]] if shared_path else None
    car.end_node = end_node
    car.last_node = start_node
    cars.append(car)

evolution_timer = 0
generation = 1
evolution_interval = 25 * FPS  

running = True
while running:
    screen.fill((34, 139, 34))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                print("Resetting to new location...")
                road_system = OSMRoadSystem(
                    center_lat=random.uniform(-40, 50), 
                    center_lon=random.uniform(-120, 120), 
                    radius=1000
                )
                road_bounds = road_system.get_road_bounds()
                camera.set_bounds(*road_bounds)
                 
                pathfinder = AStarPathfinder(road_system)
                
                # Pick a random start node (where cars will spawn) and a random end node
                all_nodes = list(pathfinder.node_graph['positions'].keys())
                start_node = random.choice(all_nodes)
                end_node = random.choice([n for n in all_nodes if n != start_node])

                start_pos = pathfinder.node_graph['positions'][start_node]
                end_pos = pathfinder.node_graph['positions'][end_node]

                shared_path = pathfinder.find_path(start_pos[0], start_pos[1], end_pos[0], end_pos[1])

                for car in cars:
                    if not car.crashed:
                        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
                        car.x, car.y, car.angle = spawn_x, spawn_y, spawn_angle
                        car.road_system = road_system
                        car.pathfinder = pathfinder
                        car.current_path = shared_path.copy()
                        car.path_index = 0
                        car.next_waypoint = pathfinder.node_graph['positions'][shared_path[0]] if shared_path else None
                        car.end_node = end_node
                        car.last_node = start_node
    
    keys = pygame.key.get_pressed()
    
    alive_cars = 0
    best_car = None
    for car in cars:
        if not car.crashed:
            if evolution_timer % 30 == 0:
                car.calculate_fitness()
            if best_car is None or car.fitness > best_car.fitness:
                best_car = car
            alive_cars += 1
    
    if best_car and not camera.manual_mode:
        camera.follow_target(best_car.x, best_car.y)
    camera.update(keys)
    
    road_system.draw_roads_with_intersections(screen, camera.x, camera.y, WIDTH, HEIGHT, camera.zoom, pathfinder)
      # Draw road edges near intersections in yellow
    if best_car:
        for car in cars:
            if car.intersection_detected:
                car.draw_raycasts(camera)
    
    for car in cars:
        if not car.crashed:
            car.move(keys)
            is_best = (car == best_car)
            if is_best or evolution_timer % 3 == 0:
                car.update_raycasts()
            
            car.draw(camera, is_best)
    
    evolution_timer += 1
    if evolution_timer >= evolution_interval or alive_cars == 0:
        cars = evolve_population(cars, population_size, road_system, pathfinder, shared_path, start_node, end_node)
        evolution_timer = 0
        generation += 1
        print(f"Generation {generation} started")
    
    font = pygame.font.Font(None, 36)
    info_text = f"Gen: {generation} | Alive: {alive_cars} | Time: {evolution_timer // FPS}s"
    if best_car:
        info_text += f" | Best Fitness: {best_car.fitness:.1f}"
    
    text_surface = font.render(info_text, True, WHITE)
    text_rect = text_surface.get_rect()
    text_rect.topleft = (10, 10)
    
    pygame.draw.rect(screen, (0, 0, 0, 128), text_rect.inflate(10, 5))
    screen.blit(text_surface, text_rect)
    
    # Display raycast configuration
    raycast_info = f"Raycasts: {NUM_RAYCASTS} rays | Spread: {RAYCAST_SPREAD}°"
    raycast_surface = font.render(raycast_info, True, WHITE)
    raycast_rect = raycast_surface.get_rect()
    raycast_rect.topleft = (10, text_rect.bottom + 10)
    
    pygame.draw.rect(screen, (0, 0, 0, 128), raycast_rect.inflate(10, 5))
    screen.blit(raycast_surface, raycast_rect)
    
    if best_car and evolution_timer % 10 == 0:
        pass

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
