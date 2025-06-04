"""
OSM Road System for 2D Car AI Simulation
Optimized for performance with spatial indexing and efficient collision detection
"""
import requests
import math
import random
import pygame
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import defaultdict

class RoadSegment:
    def __init__(self, start_point, end_point, width=1250):  # Increased to 1250 for 5x bigger roads
        self.start = start_point
        self.end = end_point
        self.width = width
        self.length = math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        
        # Pre-calculate direction vector for efficiency
        if self.length > 0:
            self.direction = ((end_point[0] - start_point[0]) / self.length, 
                            (end_point[1] - start_point[1]) / self.length)
            self.normal = (-self.direction[1], self.direction[0])  # Perpendicular vector
        else:
            self.direction = (0, 1)
            self.normal = (1, 0)
    
    def get_boundaries(self):
        """Get left and right boundaries of the road segment"""
        half_width = self.width / 2
        
        # Left boundary (offset by normal)
        left_start = (self.start[0] + self.normal[0] * half_width, 
                     self.start[1] + self.normal[1] * half_width)
        left_end = (self.end[0] + self.normal[0] * half_width, 
                   self.end[1] + self.normal[1] * half_width)
        
        # Right boundary (offset by negative normal)
        right_start = (self.start[0] - self.normal[0] * half_width, 
                      self.start[1] - self.normal[1] * half_width)
        right_end = (self.end[0] - self.normal[0] * half_width, 
                    self.end[1] - self.normal[1] * half_width)
        
        return (left_start, left_end), (right_start, right_end)
    
    def point_to_segment_distance(self, point):
        """Calculate shortest distance from point to road segment centerline"""
        px, py = point
        sx, sy = self.start
        ex, ey = self.end
        
        # Vector from start to end
        dx = ex - sx
        dy = ey - sy
        
        if dx == 0 and dy == 0:
            return math.sqrt((px - sx)**2 + (py - sy)**2)
        
        # Project point onto line segment
        t = max(0, min(1, ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)))
        
        # Find closest point on segment
        closest_x = sx + t * dx
        closest_y = sy + t * dy
        
        return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def is_point_on_road(self, point):
        """Check if point is on the road (within road width)"""
        return self.point_to_segment_distance(point) <= self.width / 2

class SpatialGrid:
    """Spatial grid for efficient collision detection and raycasting"""
    
    def __init__(self, bounds, cell_size=100):
        self.bounds = bounds  # (min_x, min_y, max_x, max_y)
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        
        # Calculate grid dimensions
        self.min_x, self.min_y, self.max_x, self.max_y = bounds
        self.cols = int((self.max_x - self.min_x) / cell_size) + 1
        self.rows = int((self.max_y - self.min_y) / cell_size) + 1
    
    def _get_cell_coords(self, x, y):
        """Convert world coordinates to grid cell coordinates"""
        col = int((x - self.min_x) / self.cell_size)
        row = int((y - self.min_y) / self.cell_size)
        return max(0, min(self.cols - 1, col)), max(0, min(self.rows - 1, row))
    
    def add_segment(self, segment):
        """Add road segment to appropriate grid cells"""
        # Get all cells that the segment intersects
        start_col, start_row = self._get_cell_coords(segment.start[0], segment.start[1])
        end_col, end_row = self._get_cell_coords(segment.end[0], segment.end[1])
        
        # Use Bresenham-like algorithm to get all cells the line passes through
        cells = set()
        
        # Simple approach: add all cells in bounding box of segment
        min_col = min(start_col, end_col)
        max_col = max(start_col, end_col)
        min_row = min(start_row, end_row)
        max_row = max(start_row, end_row)
        
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                cells.add((col, row))
        
        # Add segment to all relevant cells
        for cell in cells:
            self.grid[cell].append(segment)
    
    def get_nearby_segments(self, x, y, radius=50):
        """Get all road segments near a point"""
        # Calculate cells to check based on radius
        cells_to_check = set()
        cell_radius = int(radius / self.cell_size) + 1
        
        center_col, center_row = self._get_cell_coords(x, y)
        
        for col in range(center_col - cell_radius, center_col + cell_radius + 1):
            for row in range(center_row - cell_radius, center_row + cell_radius + 1):
                if 0 <= col < self.cols and 0 <= row < self.rows:
                    cells_to_check.add((col, row))
        
        segments = []
        seen = set()
        for cell in cells_to_check:
            for segment in self.grid[cell]:
                if id(segment) not in seen:
                    segments.append(segment)
                    seen.add(id(segment))
        
        return segments

class OSMRoadSystem:
    def __init__(self, center_lat=40.7128, center_lon=-74.0060, radius=1000):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius
        self.road_segments = []
        self.spatial_grid = None
        self.scale_factor = 2.5  # Scale up roads for better car simulation - 5x bigger
        
        # Conversion factors (approximate)
        self.meters_per_deg_lat = 111320
        self.meters_per_deg_lon = 111320 * math.cos(math.radians(center_lat))
        
        # Load roads from OSM
        self.load_roads()
        
        # Create spatial grid for optimization
        if self.road_segments:
            self._create_spatial_grid()
    
    def latlon_to_pixel(self, lat, lon):
        """Convert latitude/longitude to pixel coordinates"""
        # Convert to meters relative to center
        dx = (lon - self.center_lon) * self.meters_per_deg_lon
        dy = (lat - self.center_lat) * self.meters_per_deg_lat
        
        # Scale and convert to screen coordinates
        x = dx * self.scale_factor
        y = -dy * self.scale_factor  # Flip Y axis for screen coordinates
        
        return x, y
    
    def load_roads(self):
        """Load road data from OpenStreetMap"""
        query = f"""
        [out:json][timeout:25];
        (
          way(around:{self.radius},{self.center_lat},{self.center_lon})["highway"~"^(primary|secondary|tertiary|residential|trunk|unclassified)$"];
        );
        (._;>;);
        out body;
        """
        
        url = "http://overpass-api.de/api/interpreter"
        
        try:
            print("Loading roads from OpenStreetMap...")
            response = requests.post(url, data={'data': query}, timeout=30)
            
            if response.status_code != 200:
                print(f"Failed to fetch OSM data: {response.status_code}")
                self._create_fallback_roads()
                return
            
            data = response.json()
            self._parse_osm_data(data)
            
        except Exception as e:
            print(f"Error loading OSM data: {e}")
            self._create_fallback_roads()
    
    def _parse_osm_data(self, data):
        """Parse OSM JSON data into road segments"""
        nodes = {}
        
        # First pass: collect all nodes
        for element in data.get('elements', []):
            if element['type'] == 'node':
                nodes[element['id']] = element
        
        # Second pass: process ways (roads)
        for element in data.get('elements', []):
            if element['type'] == 'way' and 'nodes' in element:
                node_ids = element['nodes']
                
                if len(node_ids) < 2:
                    continue
                
                # Convert nodes to pixel coordinates
                points = []
                for node_id in node_ids:
                    if node_id in nodes:
                        node = nodes[node_id]
                        x, y = self.latlon_to_pixel(node['lat'], node['lon'])
                        points.append((x, y))
                
                if len(points) < 2:
                    continue
                
                # Create road segments between consecutive points
                road_width = self._get_road_width(element.get('tags', {}))
                for i in range(len(points) - 1):
                    segment = RoadSegment(points[i], points[i + 1], road_width)
                    self.road_segments.append(segment)
        
        print(f"Loaded {len(self.road_segments)} road segments from OSM")
    
    def _get_road_width(self, tags):
        """Determine road width based on highway type"""
        highway_type = tags.get('highway', 'residential')
        
        width_map = {
            'motorway': 120,    # Wide highways
            'trunk': 100,       # Major roads
            'primary': 80,      # Primary roads
            'secondary': 65,    # Secondary roads
            'tertiary': 50,     # Tertiary roads
            'residential': 40,  # Residential streets
            'unclassified': 40  # Unclassified roads
        }
        
        return width_map.get(highway_type, 40)  # Default 40
    
    def _create_fallback_roads(self):
        """Create simple road network if OSM data fails"""
        print("Creating fallback road network...")
        
        # Create a simple grid of roads
        grid_size = 200
        road_width = 20
        
        # Horizontal roads
        for y in range(-400, 401, grid_size):
            segment = RoadSegment((-600, y), (600, y), road_width)
            self.road_segments.append(segment)
        
        # Vertical roads
        for x in range(-600, 601, grid_size):
            segment = RoadSegment((x, -400), (x, 400), road_width)
            self.road_segments.append(segment)
        
        print(f"Created {len(self.road_segments)} fallback road segments")
    
    def _create_spatial_grid(self):
        """Create spatial grid for efficient collision detection"""
        if not self.road_segments:
            return
        
        # Calculate bounds
        min_x = min(min(seg.start[0], seg.end[0]) for seg in self.road_segments)
        max_x = max(max(seg.start[0], seg.end[0]) for seg in self.road_segments)
        min_y = min(min(seg.start[1], seg.end[1]) for seg in self.road_segments)
        max_y = max(max(seg.start[1], seg.end[1]) for seg in self.road_segments)
        
        # Add padding
        padding = 100
        bounds = (min_x - padding, min_y - padding, max_x + padding, max_y + padding)
        
        self.spatial_grid = SpatialGrid(bounds, cell_size=100)
        
        # Add all segments to grid
        for segment in self.road_segments:
            self.spatial_grid.add_segment(segment)
    def get_random_spawn_point(self):
        """Get a fixed spawn point on a road (same position for all cars)"""
        if not self.road_segments:
            return 0, 0, 0
        
        # Always use the first road segment for consistent spawning
        segment = self.road_segments[0]
        
        # Use midpoint of the segment for consistent spawning
        x = (segment.start[0] + segment.end[0]) / 2
        y = (segment.start[1] + segment.end[1]) / 2
          # Calculate angle based on road direction
        # Car movement: x += sin(angle) * speed, y -= cos(angle) * speed
        # So angle 0 = moving up (negative Y), angle 90 = moving right (positive X)
        angle = math.degrees(math.atan2(segment.direction[0], -segment.direction[1]))
        
        return x, y, angle
    def is_point_on_road(self, x, y, tolerance=None):
        """Check if a point is on any road (optimized with spatial grid)"""
        if not self.spatial_grid:
            return False
        
        # Use a larger tolerance for raycasting to handle intersections better
        check_radius = tolerance if tolerance is not None else 40
        nearby_segments = self.spatial_grid.get_nearby_segments(x, y, radius=check_radius)
        
        for segment in nearby_segments:
            if segment.is_point_on_road((x, y)):
                return True
        return False
    
    def raycast_to_road_edge(self, start_x, start_y, angle, max_distance=200):
        """Cast a ray and find distance to nearest road edge (improved for intersections)"""
        if not self.spatial_grid:
            return max_distance
            
        # Ray direction - match car movement convention: sin for x, -cos for y
        rad_angle = math.radians(angle)
        dx = math.sin(rad_angle)
        dy = -math.cos(rad_angle)
        
        # Check if starting point is on a road - if not, return short distance
        if not self.is_point_on_road(start_x, start_y):
            return 5  # Very short distance if not on road
        
        # Step along the ray and check if we're still on road
        step_size = 2  # Smaller steps for better accuracy
        current_distance = 0
        
        while current_distance < max_distance:
            current_distance += step_size
            check_x = start_x + dx * current_distance
            check_y = start_y + dy * current_distance
            
            # If we're no longer on any road, we've hit an edge
            if not self.is_point_on_road(check_x, check_y):
                return current_distance
        
        return max_distance
    
    def _ray_line_intersection(self, ray_start, ray_dir, line_start, line_end):
        """Calculate intersection distance between ray and line segment"""
        x1, y1 = ray_start
        dx, dy = ray_dir
        
        x3, y3 = line_start
        x4, y4 = line_end
        
        # Line segment vector
        line_dx = x4 - x3
        line_dy = y4 - y3
        
        # Check if ray and line segment are parallel
        denominator = dx * line_dy - dy * line_dx
        if abs(denominator) < 1e-10:
            return None
        
        # Calculate intersection parameters
        t = ((x3 - x1) * line_dy - (y3 - y1) * line_dx) / denominator
        u = ((x3 - x1) * dy - (y3 - y1) * dx) / denominator
        
        # Check if intersection is within line segment and in ray direction
        if 0 <= u <= 1 and t > 0:
            return t
        
        return None
    def draw_roads(self, screen, camera_x, camera_y, screen_width, screen_height, zoom=1.0):
        """Draw roads relative to camera position with improved rendering"""
        if not self.road_segments:
            return
        
        # Calculate visible area with zoom
        view_width = screen_width / zoom
        view_height = screen_height / zoom
        left = camera_x - view_width // 2
        right = camera_x + view_width // 2
        top = camera_y - view_height // 2
        bottom = camera_y + view_height // 2
        
        # Group segments by approximate position to reduce drawing artifacts
        visible_segments = []
        
        for segment in self.road_segments:
            # Quick bounds check
            seg_left = min(segment.start[0], segment.end[0]) - segment.width
            seg_right = max(segment.start[0], segment.end[0]) + segment.width
            seg_top = min(segment.start[1], segment.end[1]) - segment.width
            seg_bottom = max(segment.start[1], segment.end[1]) + segment.width
            
            if (seg_right >= left and seg_left <= right and 
                seg_bottom >= top and seg_top <= bottom):
                visible_segments.append(segment)
        
        # Draw road segments with better rendering
        for segment in visible_segments:
            # Convert to screen coordinates
            start_screen = (
                int((segment.start[0] - camera_x) * zoom + screen_width // 2),
                int((segment.start[1] - camera_y) * zoom + screen_height // 2)
            )
            end_screen = (
                int((segment.end[0] - camera_x) * zoom + screen_width // 2),
                int((segment.end[1] - camera_y) * zoom + screen_height // 2)
            )
            
            # Calculate road width in screen coordinates
            road_width = max(1, int(segment.width * zoom))
            
            # Draw road background (dark gray)
            if road_width > 0:
                pygame.draw.line(screen, (50, 50, 50), start_screen, end_screen, road_width)
                
                # Draw road center area (lighter gray)
                center_width = max(1, int(road_width * 0.8))
                pygame.draw.line(screen, (80, 80, 80), start_screen, end_screen, center_width)
                
                # Draw center line (white dashed line effect)
                if road_width > 10:
                    line_width = max(1, min(4, road_width // 15))
                    pygame.draw.line(screen, (255, 255, 255), start_screen, end_screen, line_width)
    
    def get_road_bounds(self):
        """Get the bounds of all roads for camera limits"""
        if not self.road_segments:
            return (-500, -500, 500, 500)
        
        min_x = min(min(seg.start[0], seg.end[0]) for seg in self.road_segments)
        max_x = max(max(seg.start[0], seg.end[0]) for seg in self.road_segments)
        min_y = min(min(seg.start[1], seg.end[1]) for seg in self.road_segments)
        max_y = max(max(seg.start[1], seg.end[1]) for seg in self.road_segments)
        
        return min_x, min_y, max_x, max_y
