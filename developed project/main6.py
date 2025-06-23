import pygame
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from osm_roads import OSMRoadSystem
from camera import Camera
import heapq
from collections import defaultdict

pygame.init()

WIDTH, HEIGHT = 1920, 1080
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

class CheckpointSystem:
    def __init__(self, pathfinder, num_checkpoints=8, min_checkpoint_distance=300):
        self.pathfinder = pathfinder
        self.num_checkpoints = num_checkpoints
        self.min_checkpoint_distance = min_checkpoint_distance
        self.checkpoints = []
        self.checkpoint_radius = 75
        self._generate_checkpoints()
    
    def _generate_checkpoints(self):
        if not self.pathfinder.node_graph['positions']:
            return
        
        available_nodes = list(self.pathfinder.node_graph['positions'].keys())
        if len(available_nodes) < self.num_checkpoints:
            self.num_checkpoints = len(available_nodes)
        
        self.checkpoints = []
        used_nodes = set()
        
        if available_nodes:
            first_checkpoint = random.choice(available_nodes)
            self.checkpoints.append(first_checkpoint)
            used_nodes.add(first_checkpoint)
        
        for i in range(1, self.num_checkpoints):
            best_candidate = None
            best_distance = 0
            
            for node_id in available_nodes:
                if node_id in used_nodes:
                    continue
                
                min_dist_to_existing = float('inf')
                for existing_checkpoint in self.checkpoints:
                    existing_pos = self.pathfinder.node_graph['positions'][existing_checkpoint]
                    candidate_pos = self.pathfinder.node_graph['positions'][node_id]
                    dist = math.sqrt(
                        (existing_pos[0] - candidate_pos[0])**2 + 
                        (existing_pos[1] - candidate_pos[1])**2
                    )
                    min_dist_to_existing = min(min_dist_to_existing, dist)
                
                if min_dist_to_existing >= self.min_checkpoint_distance:
                    if min_dist_to_existing > best_distance:
                        best_distance = min_dist_to_existing
                        best_candidate = node_id
            
            if best_candidate:
                self.checkpoints.append(best_candidate)
                used_nodes.add(best_candidate)
            elif available_nodes:
                remaining_nodes = [n for n in available_nodes if n not in used_nodes]
                if remaining_nodes:
                    self.checkpoints.append(random.choice(remaining_nodes))
        
        print(f"Generated {len(self.checkpoints)} checkpoints")
        for i, checkpoint in enumerate(self.checkpoints):
            pos = self.pathfinder.node_graph['positions'][checkpoint]
            print(f"  Checkpoint {i+1}: ({pos[0]:.1f}, {pos[1]:.1f})")
    
    def get_checkpoint_position(self, checkpoint_index):
        if 0 <= checkpoint_index < len(self.checkpoints):
            checkpoint_id = self.checkpoints[checkpoint_index]
            return self.pathfinder.node_graph['positions'].get(checkpoint_id)
        return None
    
    def get_next_checkpoint_for_car(self, car):
        if not hasattr(car, 'current_checkpoint_index'):
            car.current_checkpoint_index = 0
        
        current_pos = self.get_checkpoint_position(car.current_checkpoint_index)
        if current_pos:
            distance = math.sqrt((car.x - current_pos[0])**2 + (car.y - current_pos[1])**2)
            if distance < self.checkpoint_radius:
                car.current_checkpoint_index += 1
                car.checkpoint_reached_bonus = car.current_checkpoint_index * 100
                print(f"Car reached checkpoint {car.current_checkpoint_index}!")
        
        next_index = car.current_checkpoint_index % len(self.checkpoints)
        return self.checkpoints[next_index] if self.checkpoints else None
    
    def draw_checkpoints(self, screen, camera, current_checkpoint_index=0):
        for i, checkpoint_id in enumerate(self.checkpoints):
            pos = self.pathfinder.node_graph['positions'].get(checkpoint_id)
            if pos:
                screen_pos = camera.world_to_screen(pos[0], pos[1])
                
                if i < current_checkpoint_index:
                    color = (0, 255, 0)
                    outline_color = (0, 150, 0)
                elif i == current_checkpoint_index:
                    pulse = int(50 + 25 * math.sin(pygame.time.get_ticks() * 0.01))
                    color = (255, 255, 0)
                    outline_color = (255, 150, 0)
                    pygame.draw.circle(screen, outline_color, screen_pos, pulse, 3)
                else:
                    color = (100, 150, 255)
                    outline_color = (50, 100, 200)
                
                pygame.draw.circle(screen, color, screen_pos, 30, 4)
                pygame.draw.circle(screen, outline_color, screen_pos, 20, 2)
                
                font = pygame.font.Font(None, 24)
                text = font.render(str(i + 1), True, (255, 255, 255))
                text_rect = text.get_rect(center=screen_pos)
                screen.blit(text, text_rect)

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
    def __init__(self, x, y, color, use_ai=False, road_system=None, pathfinder=None, checkpoint_system=None):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.color = color
        self.width = 12/1.4
        self.height = 20/1.4
        
        self.raycast_length = 120        
        self.raycast_angles = [-90, -60, -30, 0, 30, 60, 90]
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
        self.intersection_cooldown = 60
        
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
        self.checkpoint_system = checkpoint_system
        self.current_checkpoint_index = 0
        self.target_node = None
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
                self.ai = AdvancedCarAI(input_size=24, hidden_size=256, output_size=6)
            else:
                self.ai = CarAI()
            self.randomize_weights()
            self._update_pathfinding()
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
    
    def move(self, keys):
        if self.use_ai:
            self.ai_move()
        else:
            if keys[pygame.K_UP]:
                self.speed += 0.2
            if keys[pygame.K_DOWN]:
                self.speed -= 0.2
            if keys[pygame.K_LEFT]:
                self.angle -= 5
            if keys[pygame.K_RIGHT]:
                self.angle += 5

        prev_x, prev_y = self.x, self.y
        self.x += math.sin(math.radians(self.angle)) * self.speed
        self.y -= math.cos(math.radians(self.angle)) * self.speed

        if self.use_ai:
            distance = math.sqrt((self.x - prev_x)**2 + (self.y - prev_y)**2)
            self.distance_traveled += distance
            self.time_alive += 1

        self.speed *= 0.95
        
        if self.use_ai and self.time_alive > self.stationary_threshold:
            self.frames_since_movement_check += 1
            if self.frames_since_movement_check >= self.movement_check_interval:
                distance_moved = math.sqrt((self.x - self.last_position[0])**2 + 
                                         (self.y - self.last_position[1])**2)
                
                if distance_moved < self.min_movement_distance:
                    self.stationary_timer += self.movement_check_interval
                    if self.stationary_timer >= 60:
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
        if not self.road_system:
            self.raycast_distances = [self.raycast_length] * len(self.raycast_angles)
            self.short_raycast_distances = [self.short_raycast_length] * len(self.short_raycast_angles)
            return
        
        self.raycast_distances = []
        for angle_offset in self.raycast_angles:
            ray_angle = self.angle + angle_offset
            distance = self.road_system.raycast_to_road_edge(
                self.x, self.y, ray_angle, self.raycast_length
            )
            self.raycast_distances.append(distance)
        
        self.short_raycast_distances = []
        for angle_offset in self.short_raycast_angles:
            ray_angle = self.angle + angle_offset
            distance = self.road_system.raycast_to_road_edge(
                self.x, self.y, ray_angle, self.short_raycast_length
            )
            self.short_raycast_distances.append(distance)
        
        self.detect_intersection()
    
    def detect_intersection(self):
        if len(self.raycast_distances) < 7:
            return
        
        left_opening = self.raycast_distances[0] > self.raycast_length * 0.8  # -90 degrees
        right_opening = self.raycast_distances[6] > self.raycast_length * 0.8  # +90 degrees
        forward_clear = self.raycast_distances[3] > self.raycast_length * 0.6  # 0 degrees
        
        far_left_clear = self.raycast_distances[1] > self.raycast_length * 0.7  # -60 degrees
        far_right_clear = self.raycast_distances[5] > self.raycast_length * 0.7  # +60 degrees
        
        self.frames_since_intersection += 1
        
        if (self.frames_since_intersection > self.intersection_cooldown and
            ((left_opening and right_opening) or 
             (left_opening and forward_clear and far_left_clear) or
             (right_opening and forward_clear and far_right_clear))):
            
            if not self.intersection_detected:
                self.intersection_detected = True
                self.frames_since_intersection = 0
                self.make_intersection_decision()
        else:
            self.intersection_detected = False
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
        if len(self.raycast_distances) < 7 or len(self.short_raycast_distances) < 3:
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
            
            target_direction = 0.0
            if self.next_waypoint:
                dx = self.next_waypoint[0] - self.x
                dy = self.next_waypoint[1] - self.y
                target_angle = math.degrees(math.atan2(dx, -dy))
                angle_diff = target_angle - self.angle
                while angle_diff > 180: angle_diff -= 360
                while angle_diff < -180: angle_diff += 360
                target_direction = angle_diff / 180.0
            inputs.append(target_direction)
            
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
            
            if self.next_waypoint and path_following_weight > 0.3:
                dx = self.next_waypoint[0] - self.x
                dy = self.next_waypoint[1] - self.y
                target_angle = math.degrees(math.atan2(dx, -dy))
                angle_diff = target_angle - self.angle
                while angle_diff > 180: angle_diff -= 360
                while angle_diff < -180: angle_diff += 360
                
                pathfinding_turn = max(-1.0, min(1.0, angle_diff / 30.0))
                turn_direction = (turn_direction * (1 - path_following_weight) + 
                                pathfinding_turn * path_following_weight)
                
                if abs(angle_diff) < 30:
                    acceleration = max(acceleration, 0.2)
        
        else:
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
        
        if self.intersection_detected and self.last_intersection_decision:
            if self.last_intersection_decision == "left":
                turn_direction = min(1.0, turn_direction - 0.7)
            elif self.last_intersection_decision == "right":
                turn_direction = max(-1.0, turn_direction + 0.7)
        
        if len(self.short_raycast_distances) >= 3:
            left_obstacle = self.short_raycast_distances[0] < self.short_raycast_length * 0.5
            front_obstacle = self.short_raycast_distances[1] < self.short_raycast_length * 0.3
            right_obstacle = self.short_raycast_distances[2] < self.short_raycast_length * 0.5
            
            if front_obstacle:
                acceleration = min(acceleration, -0.5)
                if left_obstacle and not right_obstacle:
                    turn_direction = max(turn_direction, 0.8)
                elif right_obstacle and not left_obstacle:
                    turn_direction = min(turn_direction, -0.8)
                elif left_obstacle and right_obstacle:
                    acceleration = -1.0
            
            if left_obstacle and not front_obstacle:
                turn_direction += 0.3
            if right_obstacle and not front_obstacle:
                turn_direction -= 0.3
        
        speed_scale = 0.5 if self.intersection_detected else 0.4
        self.speed += acceleration * speed_scale
        
        turn_scale = 4.0 if abs(self.speed) < 1.0 else 2.5
        self.angle += turn_direction * turn_scale
        
        self.predicted_direction = turn_direction * 20
        self.path_curvature = turn_direction

    def calculate_fitness(self):
        if self.off_road_time > 0:
            off_road_penalty = (self.off_road_time ** 2) * 2.0
            off_road_base_penalty = 100
        else:
            off_road_penalty = 0
            off_road_base_penalty = 0

        road_bonus = self.time_alive * 0.5 if self.off_road_time == 0 else 0
        distance_bonus = self.distance_traveled * 0.3 if self.off_road_time < 10 else 0
        time_bonus = self.time_alive * 0.05

        avg_speed = self.distance_traveled / self.time_alive if self.time_alive > 0 else 0
        speed_bonus = avg_speed * 15
        
        smoothness_bonus = 0
        if self.movement_smoothness > 0:
            smoothness_bonus = max(0, (math.pi - self.movement_smoothness) * 5)
        
        intersection_bonus = 0
        if hasattr(self, 'last_intersection_decision') and self.last_intersection_decision:
            intersection_bonus = 20
        
        checkpoint_bonus = self.checkpoint_reached_bonus
        
        current_checkpoint_bonus = 0
        if self.checkpoint_system and hasattr(self, 'current_checkpoint_index'):
            current_checkpoint_pos = self.checkpoint_system.get_checkpoint_position(self.current_checkpoint_index)
            if current_checkpoint_pos:
                distance_to_checkpoint = math.sqrt(
                    (self.x - current_checkpoint_pos[0])**2 + 
                    (self.y - current_checkpoint_pos[1])**2
                )
                max_distance = 500
                proximity_to_checkpoint = max(0, (max_distance - distance_to_checkpoint) / max_distance)
                current_checkpoint_bonus = proximity_to_checkpoint * 50

        pathfinding_bonus = 0
        target_seeking_bonus = 0
        
        if self.target_node and self.pathfinder:
            target_pos = self.pathfinder.node_graph['positions'].get(self.target_node)
            if target_pos:
                current_distance = math.sqrt((self.x - target_pos[0])**2 + (self.y - target_pos[1])**2)
                
                max_possible_distance = 1000
                proximity_bonus = (max_possible_distance - current_distance) / max_possible_distance * 100
                target_seeking_bonus += proximity_bonus
                
                if current_distance < 50:
                    target_seeking_bonus += 200
                    self.reached_target_bonus = 200
                elif current_distance < 100:
                    target_seeking_bonus += 100
                    self.reached_target_bonus = 100
                elif current_distance < 200:
                    target_seeking_bonus += 50
                    self.reached_target_bonus = 50
        
        if self.current_path and len(self.current_path) > 0:
            path_progress = self.path_index / max(1, len(self.current_path) - 1)
            pathfinding_bonus += path_progress * 150
            
            pathfinding_bonus += self.path_following_accuracy * 75
            
            pathfinding_bonus += 25
        
        navigation_intelligence_bonus = 0
        if hasattr(self, 'use_advanced_ai') and self.use_advanced_ai:
            navigation_intelligence_bonus += 30
            
            if self.intersection_detected and self.next_waypoint:
                navigation_intelligence_bonus += 20
        
        crash_penalty = 300 if self.crashed else 0
        
        stationary_penalty = self.stationary_timer * 2
        
        exploration_bonus = 0
        if hasattr(self, 'position_history') and len(self.position_history) > 10:
            unique_positions = set()
            for pos in self.position_history[-10:]:
                rounded_pos = (round(pos[0] / 20) * 20, round(pos[1] / 20) * 20)
                unique_positions.add(rounded_pos)
            
            exploration_ratio = len(unique_positions) / 10
            exploration_bonus = exploration_ratio * 25
        
        self.fitness = (
            distance_bonus + time_bonus + road_bonus + speed_bonus + 
            smoothness_bonus + intersection_bonus + pathfinding_bonus +
            target_seeking_bonus + navigation_intelligence_bonus + exploration_bonus +
            checkpoint_bonus + current_checkpoint_bonus -
            off_road_penalty - off_road_base_penalty - crash_penalty - stationary_penalty
        )
        
        return self.fitness
    
    def clone(self):
        new_car = Car(self.x, self.y, self.color, use_ai=True, road_system=self.road_system, 
                     pathfinder=self.pathfinder, checkpoint_system=self.checkpoint_system)
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
            
            screen_point = camera.world_to_screen(point_x, point_y)
            immediate_path_points.append(screen_point)
        
        if len(immediate_path_points) > 1:
            for i in range(len(immediate_path_points) - 1):
                progress = i / (len(immediate_path_points) - 1)
                thickness = max(1, int(4 * (1 - progress * 0.7)))  
                
                red = 255
                green = int(255 * (1 - progress * 0.4))
                blue = int(50 * (1 - progress))
                color = (red, green, blue)
                
                pygame.draw.line(screen, color, immediate_path_points[i], immediate_path_points[i + 1], thickness)
        
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

    def make_intersection_decision(self):
        available_directions = []
        
        if len(self.raycast_distances) >= 7:
            if self.raycast_distances[0] > self.raycast_length * 0.6:
                available_directions.append("left")
            
            if self.raycast_distances[6] > self.raycast_length * 0.6:
                available_directions.append("right")
            
            if self.raycast_distances[3] > self.raycast_length * 0.5:
                available_directions.append("forward")
        
        if available_directions:
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
        if self.checkpoint_system and self.pathfinder:
            self.target_node = self.checkpoint_system.get_next_checkpoint_for_car(self)
            
            if self.target_node:
                target_pos = self.pathfinder.node_graph['positions'].get(self.target_node)
                if target_pos:
                    self.current_path = self.pathfinder.find_path(self.x, self.y, target_pos[0], target_pos[1])
                    self.path_index = 0
                    self._update_next_waypoint()
                    
                    self.distance_to_target = math.sqrt(
                        (self.x - target_pos[0])**2 + (self.y - target_pos[1])**2
                    )
    
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
            
            features[0] = min(1.0, distance / 200.0)
            
            target_angle = math.degrees(math.atan2(dx, -dy))
            angle_diff = target_angle - self.angle
            while angle_diff > 180: angle_diff -= 360
            while angle_diff < -180: angle_diff += 360
            features[1] = angle_diff / 180.0  
            
            if self.current_path:
                features[2] = self.path_index / max(1, len(self.current_path) - 1)
            
            features[3] = min(1.0, self.distance_to_target / 500.0)
            
        features[4] = self.path_following_accuracy
        
        remaining_waypoints = len(self.current_path) - self.path_index if self.current_path else 0
        features[5] = min(1.0, remaining_waypoints / 10.0)
        
        optimal_speed = 3.0  
        features[6] = min(1.0, abs(self.speed) / optimal_speed)
        
        open_directions = 0
        if len(self.raycast_distances) >= 7:
            threshold = self.raycast_length * 0.6
            for distance in self.raycast_distances:
                if distance > threshold:
                    open_directions += 1
        features[7] = min(1.0, open_directions / 7.0)
        
        features[8] = min(1.0, self.time_alive % 180 / 180.0)  
        
        if self.next_waypoint:
            path_deviation = self._calculate_path_deviation()
            features[9] = min(1.0, path_deviation / 50.0)
        
        if self.path_curvature != 0:
            features[10] = abs(self.path_curvature)
        
        if self.distance_to_target < 50:
            features[11] = 1.0
        
        return features
    
    def _calculate_path_deviation(self):
        if not self.next_waypoint:
            return 0
        
        dx = self.next_waypoint[0] - self.x
        dy = self.next_waypoint[1] - self.y
        distance_to_waypoint = math.sqrt(dx*dx + dy*dy)
        
        if distance_to_waypoint < 30:
            self.path_index += 1
            self._update_next_waypoint()
            self.path_following_accuracy = min(1.0, self.path_following_accuracy + 0.1)
            return 0
        
        return distance_to_waypoint
def evolve_population(cars, population_size=30, road_system=None, pathfinder=None, checkpoint_system=None):
    for car in cars:
        car.calculate_fitness()
    cars.sort(key=lambda x: x.fitness, reverse=True)
    print(f"Generation evolved! Best fitness: {cars[0].fitness:.2f}")
    
    if cars[0].checkpoint_system and hasattr(cars[0], 'current_checkpoint_index'):
        print(f"Best car checkpoint progress: {cars[0].current_checkpoint_index}/{cars[0].checkpoint_system.num_checkpoints}")
        if hasattr(cars[0], 'checkpoint_reached_bonus'):
            print(f"Checkpoint bonus: {cars[0].checkpoint_reached_bonus}")
    
    elite_count = max(1, population_size // 5)
    elites = cars[:elite_count]
    new_cars = []
    
    for elite in elites:
        new_car = elite.clone()
        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point() if road_system else (0, 0, 0)
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
        new_car.frames_since_movement_check = 0
        new_car.current_path = []
        new_car.path_index = 0
        new_car.next_waypoint = None
        new_car.path_following_accuracy = 0
        new_car.reached_target_bonus = 0
        new_car.current_checkpoint_index = 0
        new_car.checkpoint_reached_bonus = 0
        if hasattr(new_car.ai, 'hidden_state'):
            new_car.ai.hidden_state = None
            new_car.ai.memory_sequence = []
        new_cars.append(new_car)
    while len(new_cars) < population_size:
        parent_index = random.randint(0, min(len(cars) - 1, population_size // 2 - 1))
        parent = cars[parent_index]
        offspring = parent.clone()
        
        base_mutation_rate = 0.12
        base_mutation_strength = 0.25
        
        if parent.fitness > cars[0].fitness * 0.8:
            offspring.mutate(mutation_rate=base_mutation_rate * 0.7, 
                           mutation_strength=base_mutation_strength * 0.8)
        else:
            offspring.mutate(mutation_rate=base_mutation_rate * 1.3, 
                           mutation_strength=base_mutation_strength * 1.2)
        
        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point() if road_system else (0, 0, 0)
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
        offspring.frames_since_movement_check = 0
        offspring.current_path = []
        offspring.path_index = 0
        offspring.next_waypoint = None
        offspring.path_following_accuracy = 0
        offspring.reached_target_bonus = 0
        offspring.current_checkpoint_index = 0
        offspring.checkpoint_reached_bonus = 0
        if hasattr(offspring.ai, 'hidden_state'):
            offspring.ai.hidden_state = None
            offspring.ai.memory_sequence = []
        offspring.color = random.choice([RED, GREEN, BLUE])
        new_cars.append(offspring)
    
    return new_cars


print("Loading road system...")
road_system = OSMRoadSystem(center_lat=-36.8825825, center_lon=174.9143453, radius=1000)

print("Building pathfinding graph...")
pathfinder = AStarPathfinder(road_system)

print("Setting up checkpoint system...")
checkpoint_system = CheckpointSystem(pathfinder, num_checkpoints=10, min_checkpoint_distance=250)

camera = Camera(0, 0, WIDTH, HEIGHT)
road_bounds = road_system.get_road_bounds()
camera.set_bounds(*road_bounds)

population_size = 50
cars = []
for i in range(population_size):
    spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
    color = random.choice([RED, GREEN, BLUE])
    car = Car(spawn_x, spawn_y, color, use_ai=True, road_system=road_system, 
             pathfinder=pathfinder, checkpoint_system=checkpoint_system)
    cars.append(car)

checkpoint_system = CheckpointSystem(pathfinder, num_checkpoints=8, min_checkpoint_distance=300)

evolution_timer = 0
generation = 1
evolution_interval = 8 * FPS  

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
                
                checkpoint_system = CheckpointSystem(pathfinder, num_checkpoints=10, min_checkpoint_distance=250)

                for car in cars:
                    if not car.crashed:
                        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
                        car.x, car.y, car.angle = spawn_x, spawn_y, spawn_angle
                        car.road_system = road_system
                        car.pathfinder = pathfinder
                        car.checkpoint_system = checkpoint_system
                        car.current_checkpoint_index = 0
                        car._update_pathfinding()
    
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
    
    road_system.draw_roads(screen, camera.x, camera.y, WIDTH, HEIGHT, camera.zoom)
      # Draw checkpoints
    if checkpoint_system and best_car:
        checkpoint_system.draw_checkpoints(screen, camera, 
                                          getattr(best_car, 'current_checkpoint_index', 0))
    
    for car in cars:
        if not car.crashed:
            car.move(keys)
            is_best = (car == best_car)
            if is_best or evolution_timer % 3 == 0:
                car.update_raycasts()
            
            car.draw(camera, is_best)
    
    checkpoint_system.draw_checkpoints(screen, camera)
    
    evolution_timer += 1
    if evolution_timer >= evolution_interval or alive_cars == 0:
        cars = evolve_population(cars, population_size, road_system, pathfinder, checkpoint_system)
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
    
    if best_car and evolution_timer % 10 == 0:
        pass

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
