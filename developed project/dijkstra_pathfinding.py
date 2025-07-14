"""
Simple Dijkstra's Algorithm Pathfinding for Road Networks
"""
import heapq
import math
import random
from typing import List, Tuple, Optional, Dict, Set

class DijkstraPathfinder:
    def __init__(self, road_system):
        self.road_system = road_system
        self.node_graph = {'positions': {}, 'connections': {}}
        self.build_graph()
    
    def build_graph(self):
        """Build a graph from road segments for pathfinding"""
        print("Building pathfinding graph...")
        
        # Create nodes from road segment endpoints
        node_id = 0
        position_to_node = {}
        connection_threshold = 50  # Distance threshold for connecting nodes
        
        # Add all road segment endpoints as potential nodes
        for segment in self.road_system.road_segments:
            start_pos = segment.start
            end_pos = segment.end
            
            # Round positions to avoid floating point issues
            start_pos = (round(start_pos[0], 1), round(start_pos[1], 1))
            end_pos = (round(end_pos[0], 1), round(end_pos[1], 1))
            
            # Add start node
            if start_pos not in position_to_node:
                position_to_node[start_pos] = node_id
                self.node_graph['positions'][node_id] = start_pos
                self.node_graph['connections'][node_id] = []
                node_id += 1
            
            # Add end node
            if end_pos not in position_to_node:
                position_to_node[end_pos] = node_id
                self.node_graph['positions'][node_id] = end_pos
                self.node_graph['connections'][node_id] = []
                node_id += 1
            
            # Connect the nodes (road segment)
            start_node_id = position_to_node[start_pos]
            end_node_id = position_to_node[end_pos]
            
            if end_node_id not in self.node_graph['connections'][start_node_id]:
                self.node_graph['connections'][start_node_id].append(end_node_id)
            if start_node_id not in self.node_graph['connections'][end_node_id]:
                self.node_graph['connections'][end_node_id].append(start_node_id)
        
        # Connect nearby nodes to create intersections
        nodes = list(self.node_graph['positions'].items())
        for i, (node1_id, pos1) in enumerate(nodes):
            for j, (node2_id, pos2) in enumerate(nodes[i+1:], i+1):
                distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                if distance < connection_threshold and node2_id not in self.node_graph['connections'][node1_id]:
                    self.node_graph['connections'][node1_id].append(node2_id)
                    self.node_graph['connections'][node2_id].append(node1_id)
        
        print(f"Graph built with {len(self.node_graph['positions'])} nodes")
    
    def get_nearest_node(self, x, y):
        """Find the nearest node to a given position"""
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
    
    def get_random_target_node(self):
        """Get a random node that's on a road"""
        if not self.node_graph['positions']:
            return None
        
        # Try to find a node that's actually on a road
        attempts = 0
        while attempts < 20:
            node_id = random.choice(list(self.node_graph['positions'].keys()))
            pos = self.node_graph['positions'][node_id]
            
            # Check if this position is on a road
            if self.road_system.is_point_on_road(pos[0], pos[1]):
                return node_id
            attempts += 1
        
        # Fallback: just return any random node
        return random.choice(list(self.node_graph['positions'].keys()))
    
    def dijkstra_shortest_path(self, start_node, target_node):
        """Find shortest path using Dijkstra's algorithm"""
        if start_node is None or target_node is None or start_node == target_node:
            return []
        
        # Initialize distances and previous nodes
        distances = {node: float('inf') for node in self.node_graph['positions']}
        distances[start_node] = 0
        previous = {}
        visited = set()
        
        # Priority queue: (distance, node)
        pq = [(0, start_node)]
        
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # If we reached the target, reconstruct path
            if current_node == target_node:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = previous.get(current_node)
                return list(reversed(path))
            
            # Check all neighbors
            for neighbor in self.node_graph['connections'].get(current_node, []):
                if neighbor in visited:
                    continue
                
                # Calculate distance to neighbor
                pos1 = self.node_graph['positions'][current_node]
                pos2 = self.node_graph['positions'][neighbor]
                edge_distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                new_distance = distances[current_node] + edge_distance
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_distance, neighbor))
        
        return []  # No path found
    
    def find_path(self, start_x, start_y, target_x, target_y):
        """Find path from start position to target position"""
        start_node = self.get_nearest_node(start_x, start_y)
        target_node = self.get_nearest_node(target_x, target_y)
        
        return self.dijkstra_shortest_path(start_node, target_node)
    
    def path_to_waypoints(self, path):
        """Convert a path of node IDs to a list of (x, y) waypoints"""
        if not path:
            return []
        
        waypoints = []
        for node_id in path:
            pos = self.node_graph['positions'][node_id]
            waypoints.append(pos)
        
        return waypoints
    
    def get_random_destination(self):
        """Get a random destination node and its position"""
        target_node = self.get_random_target_node()
        if target_node is not None:
            pos = self.node_graph['positions'][target_node]
            return target_node, pos
        return None, None
