"""
Pathfinding system using A* algorithm
Handles navigation and checkpoint systems
"""
import math
import random
import heapq
import pygame
from collections import defaultdict


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
