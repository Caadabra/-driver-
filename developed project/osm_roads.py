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

# --- Real-world scaling constants (approximate) ---
LANE_WIDTH_M = 3.5           # Standard lane width
LINE_WIDTH_M = 0.12          # Typical paint line width
YELLOW_LINE_WIDTH_M = 0.15
DASH_LENGTH_M = 3.0          # Center dash length
GAP_LENGTH_M = 9.0           # Center gap length
ARROW_LENGTH_M = 2.5         # Lane arrow length
ARROW_WIDTH_M = 1.2
STOP_LINE_WIDTH_M = 0.30
INTERSECTION_RADIUS_M = 4.0  # Visual radius for intersection pad

class RoadSegment:
    def __init__(self, start_point, end_point, width=40, lane_count: int = 2,
                 oneway: bool = False, lanes_forward: int | None = None, lanes_backward: int | None = None,
                 lane_turns: Optional[List[List[str]]] = None):
        self.start = start_point
        self.end = end_point
        self.width = width
        self.length = math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        # Lane meta
        self.lane_count = max(1, lane_count)
        self.oneway = oneway
        if oneway:
            self.lanes_forward = self.lane_count
            self.lanes_backward = 0
        else:
            if lanes_forward is not None or lanes_backward is not None:
                self.lanes_forward = lanes_forward or 0
                self.lanes_backward = lanes_backward or 0
                if (self.lanes_forward + self.lanes_backward) == 0:
                    self.lanes_forward = self.lane_count // 2
                    self.lanes_backward = self.lane_count - self.lanes_forward
            else:
                self.lanes_forward = self.lane_count // 2
                self.lanes_backward = self.lane_count - self.lanes_forward

        # Pre-calculate direction vector for efficiency
        if self.length > 0:
            self.direction = ((end_point[0] - start_point[0]) / self.length,
                              (end_point[1] - start_point[1]) / self.length)
            self.normal = (-self.direction[1], self.direction[0])  # Perpendicular vector
        else:
            self.direction = (0, 1)
            self.normal = (1, 0)
        # Lane turns & markings
        self.lane_turns = lane_turns or []  # list per lane of direction strings (left/right/through)
        self.lane_markings = self._compute_lane_markings()

    def _compute_lane_markings(self):
        markings = []
        if self.lane_count <= 1:
            return markings
        lane_width = self.width / self.lane_count

        def add_mark(offset_from_center, color, dashed=True, line_scale=0.08):
            markings.append({
                'offset': offset_from_center,
                'color': color,
                'dashed': dashed,
                'width': max(1, int(self.width * line_scale))
            })

        if self.oneway:
            for i in range(1, self.lane_count):
                offset = -self.width / 2 + i * lane_width
                add_mark(offset, (255, 255, 255), dashed=True)
        else:
            if self.lanes_forward and self.lanes_backward:
                dashed_center = True if (self.lanes_forward == 1 and self.lanes_backward == 1) else False
                add_mark(0, (255, 235, 59), dashed=dashed_center, line_scale=0.1)  # Yellow center
                boundary_offsets = [(-self.width / 2) + i * lane_width for i in range(1, self.lane_count)]
                for off in boundary_offsets:
                    if abs(off) < lane_width * 0.25:  # Skip center already added
                        continue
                    add_mark(off, (255, 255, 255), dashed=True)
            else:
                for i in range(1, self.lane_count):
                    offset = -self.width / 2 + i * lane_width
                    add_mark(offset, (255, 255, 255), dashed=True)
        return markings
    
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
        
        # Simple approach: add all cells in bounding box of segment
        min_col = min(start_col, end_col)
        max_col = max(start_col, end_col)
        min_row = min(start_row, end_row)
        max_row = max(start_row, end_row)
        
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                self.grid[(col, row)].append(segment)
    
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
    def get_normalized_center_deviation(self, x, y):
        """
        Returns normalized deviation from road center:
        -1 = far left edge, 0 = center, +1 = far right edge (relative to nearest segment).
        If not on a road, returns 0.
        """
        nearest_seg = None
        nearest_dist2 = float('inf')
        nearest_center = None
        for seg in self.road_segments:
            sx, sy = seg.start; ex, ey = seg.end
            dx = ex - sx; dy = ey - sy
            seg_len2 = dx*dx + dy*dy
            if seg_len2 <= 1e-6:
                continue
            t = ((x - sx)*dx + (y - sy)*dy)/seg_len2
            t = max(0.0, min(1.0, t))
            px = sx + dx * t; py = sy + dy * t
            d2 = (px - x)**2 + (py - y)**2
            if d2 < nearest_dist2:
                nearest_dist2 = d2
                nearest_seg = seg
                nearest_center = (px, py)
        if nearest_seg is None or nearest_center is None:
            return 0.0
        # Compute deviation: signed distance from center, normalized by half road width
        px, py = nearest_center
        seg_dx = nearest_seg.end[0] - nearest_seg.start[0]
        seg_dy = nearest_seg.end[1] - nearest_seg.start[1]
        seg_len = math.hypot(seg_dx, seg_dy)
        if seg_len < 1e-6:
            return 0.0
        # Perpendicular vector to segment
        perp_dx = -seg_dy / seg_len
        perp_dy = seg_dx / seg_len
        # Signed distance: positive = right of center, negative = left
        deviation = (x - px) * perp_dx + (y - py) * perp_dy
        half_width = max(1.0, nearest_seg.width / 2.0)
        norm_dev = max(-1.0, min(1.0, deviation / half_width))
        return norm_dev
    def __init__(self, center_lat=-36.902395416035674, center_lon=174.9444570937648, radius=1000):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius
        self.road_segments = []
        self.spatial_grid = None
        self.scale_factor = 2.5  # Scale up roads for better car simulation
        
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
        nodes: Dict[int, dict] = {}
        node_usage: Dict[int, int] = defaultdict(int)
        stop_nodes: Dict[int, dict] = {}

        # First pass: collect nodes and special points
        for element in data.get('elements', []):
            if element.get('type') == 'node':
                nid = element['id']
                nodes[nid] = element
                tags = element.get('tags', {}) or {}
                if tags.get('highway') in ('stop', 'give_way'):
                    stop_nodes[nid] = element

        # Second pass: ways
        for element in data.get('elements', []):
            if element.get('type') != 'way':
                continue
            node_ids = element.get('nodes', [])
            if len(node_ids) < 2:
                continue
            tags = element.get('tags', {}) or {}
            for nid in node_ids:
                node_usage[nid] += 1
            # Convert nodes
            points: List[Tuple[float, float]] = []
            for nid in node_ids:
                nd = nodes.get(nid)
                if nd:
                    x, y = self.latlon_to_pixel(nd['lat'], nd['lon'])
                    points.append((x, y))
            if len(points) < 2:
                continue
            road_width = self._get_road_width(tags)
            lane_count = 2
            lanes_forward = None
            lanes_backward = None
            oneway = tags.get('oneway') in ('yes', 'true', '1') or tags.get('junction') == 'roundabout'
            lane_turns: List[List[str]] = []
            turn_lanes_tag = tags.get('turn:lanes') or tags.get('turn:lanes:forward')
            if turn_lanes_tag:
                try:
                    for spec in turn_lanes_tag.split('|'):
                        dirs = [d.strip() for d in spec.split(';') if d.strip()]
                        lane_turns.append(dirs if dirs else ['through'])
                except Exception:
                    lane_turns = []
            if 'lanes' in tags:
                try:
                    lane_count = max(1, int(tags['lanes']))
                except Exception:
                    pass
            if 'lanes:forward' in tags:
                try:
                    lanes_forward = max(0, int(tags['lanes:forward']))
                except Exception:
                    pass
            if 'lanes:backward' in tags:
                try:
                    lanes_backward = max(0, int(tags['lanes:backward']))
                except Exception:
                    pass
            for i in range(len(points) - 1):
                seg = RoadSegment(points[i], points[i+1], road_width,
                                   lane_count=lane_count, oneway=oneway,
                                   lanes_forward=lanes_forward, lanes_backward=lanes_backward,
                                   lane_turns=lane_turns if lane_turns else None)
                self.road_segments.append(seg)

        # Post-process: intersections & stops
        self.intersections = []
        self.roundabouts = []
        self.stop_node_positions = []
        for nid, count in node_usage.items():
            if count >= 3 and nid in nodes:
                nd = nodes[nid]
                x, y = self.latlon_to_pixel(nd['lat'], nd['lon'])
                self.intersections.append((x, y))
        
        # Identify roundabouts from OSM data
        for element in data.get('elements', []):
            if element.get('type') == 'way':
                tags = element.get('tags', {}) or {}
                if tags.get('junction') == 'roundabout':
                    node_ids = element.get('nodes', [])
                    if len(node_ids) >= 4:  # Valid roundabout
                        # Calculate center point
                        points = []
                        for nid in node_ids:
                            nd = nodes.get(nid)
                            if nd:
                                x, y = self.latlon_to_pixel(nd['lat'], nd['lon'])
                                points.append((x, y))
                        if len(points) >= 4:
                            # Calculate centroid
                            cx = sum(p[0] for p in points) / len(points)
                            cy = sum(p[1] for p in points) / len(points)
                            # Calculate average radius
                            radius = sum(math.hypot(p[0] - cx, p[1] - cy) for p in points) / len(points)
                            self.roundabouts.append((cx, cy, radius))
        
        for nid, nd in stop_nodes.items():
            x, y = self.latlon_to_pixel(nd['lat'], nd['lon'])
            self.stop_node_positions.append((x, y))
        print(f"Loaded {len(self.road_segments)} road segments, {len(self.roundabouts)} roundabouts from OSM")

    def snap_point_to_road_center(self, x: float, y: float) -> tuple[float, float]:
        """Return nearest point on any road segment centerline (start->end) to (x,y)."""
        best_pt = (x, y)
        best_d2 = float('inf')
        for seg in self.road_segments:
            sx, sy = seg.start; ex, ey = seg.end
            dx = ex - sx; dy = ey - sy
            seg_len2 = dx*dx + dy*dy
            if seg_len2 <= 1e-6:
                continue
            t = ((x - sx)*dx + (y - sy)*dy)/seg_len2
            t = max(0.0, min(1.0, t))
            px = sx + dx * t; py = sy + dy * t
            d2 = (px - x)**2 + (py - y)**2
            if d2 < best_d2:
                best_d2 = d2
                best_pt = (px, py)
        return best_pt
    
    def _get_road_width(self, tags):
        """Determine road width based on highway type"""
        highway_type = tags.get('highway', 'residential')
        width_map = {
            'motorway': 138,
            'trunk': 115,
            'primary': 92,
            'secondary': 75,
            'tertiary': 58,
            'residential': 46,
            'unclassified': 46
        }
        explicit_width = tags.get('width')
        if explicit_width:
            try:
                meters = float(str(explicit_width).split(';')[0])
                return max(24, int(meters * self.scale_factor))
            except Exception:
                pass
        if 'lanes' in tags:
            try:
                lanes = int(tags['lanes'])
                est = int(lanes * 3.5 * self.scale_factor)  # 3.5m typical lane
                return max(width_map.get(highway_type, 46), est)
            except Exception:
                pass
        return width_map.get(highway_type, 46)
    
    def _create_fallback_roads(self):
        """Create simple road network if OSM data fails"""
        print("Creating fallback road network...")
        
        # Create a simple grid of roads
        grid_size = 200
        road_width = 46  # widened
        
        # Horizontal roads
        for y in range(-400, 401, grid_size):
            segment = RoadSegment((-600, y), (600, y), road_width, lane_count=2)
            self.road_segments.append(segment)
        
        # Vertical roads
        for x in range(-600, 601, grid_size):
            segment = RoadSegment((x, -400), (x, 400), road_width, lane_count=2)
            self.road_segments.append(segment)
        
        # Initialize intersection and roundabout lists for fallback
        self.intersections = []
        self.roundabouts = []
        self.stop_node_positions = []
        
        # Create intersections at grid crossings
        for x in range(-600, 601, grid_size):
            for y in range(-400, 401, grid_size):
                self.intersections.append((x, y))
        
        print(f"Created {len(self.road_segments)} fallback road segments with {len(self.intersections)} intersections")
    
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
    
    def draw_roads(self, screen, camera_x, camera_y, screen_width, screen_height, zoom=1.0):
        """Draw roads with realistic rendering, proper blending, and no overlaps"""
        if not self.road_segments:
            return

        view_width = screen_width / zoom
        view_height = screen_height / zoom
        left = camera_x - view_width // 2
        right = camera_x + view_width // 2
        top = camera_y - view_height // 2
        bottom = camera_y + view_height // 2

        visible_segments = []
        for segment in self.road_segments:
            seg_left = min(segment.start[0], segment.end[0]) - segment.width
            seg_right = max(segment.start[0], segment.end[0]) + segment.width
            seg_top = min(segment.start[1], segment.end[1]) - segment.width
            seg_bottom = max(segment.start[1], segment.end[1]) + segment.width
            if (seg_right >= left and seg_left <= right and seg_bottom >= top and seg_top <= bottom):
                visible_segments.append(segment)

        # First pass: Draw road bases (asphalt) with shoulders
        for segment in visible_segments:
            self._draw_road_base(screen, segment, camera_x, camera_y, screen_width, screen_height, zoom)
        
        # Second pass: Draw roundabouts and intersections
        self._draw_intersections_and_roundabouts(screen, camera_x, camera_y, screen_width, screen_height, zoom)
        
        # Third pass: Draw lane markings and details
        for segment in visible_segments:
            self._draw_road_markings(screen, segment, camera_x, camera_y, screen_width, screen_height, zoom)

    def _draw_road_base(self, screen, segment, camera_x, camera_y, screen_width, screen_height, zoom):
        """Draw the base asphalt layer with shoulders"""
        start_screen = (
            int((segment.start[0] - camera_x) * zoom + screen_width // 2),
            int((segment.start[1] - camera_y) * zoom + screen_height // 2)
        )
        end_screen = (
            int((segment.end[0] - camera_x) * zoom + screen_width // 2),
            int((segment.end[1] - camera_y) * zoom + screen_height // 2)
        )
        
        road_width = max(1, int(segment.width * zoom))
        shoulder_width = max(1, int((segment.width + 16) * zoom))  # Add shoulder
        
        # Draw shoulder (dirt/grass edge)
        pygame.draw.line(screen, (90, 120, 70), start_screen, end_screen, shoulder_width + 8)
        
        # Draw road base (dark asphalt)
        pygame.draw.line(screen, (45, 45, 50), start_screen, end_screen, road_width + 4)
        
        # Draw main road surface (lighter asphalt)
        pygame.draw.line(screen, (65, 65, 70), start_screen, end_screen, road_width)

    def _draw_road_markings(self, screen, segment, camera_x, camera_y, screen_width, screen_height, zoom):
        """Draw lane markings, centerlines, and directional arrows"""
        start_screen = (
            int((segment.start[0] - camera_x) * zoom + screen_width // 2),
            int((segment.start[1] - camera_y) * zoom + screen_height // 2)
        )
        end_screen = (
            int((segment.end[0] - camera_x) * zoom + screen_width // 2),
            int((segment.end[1] - camera_y) * zoom + screen_height // 2)
        )
        
        line_width_px = max(1, int(LINE_WIDTH_M * segment.width * 0.02 * zoom))
        yellow_line_width_px = max(1, int(YELLOW_LINE_WIDTH_M * segment.width * 0.025 * zoom))
        
        # Draw edge lines (white solid)
        self._draw_edge_lines(screen, segment, start_screen, end_screen, zoom, line_width_px)
        
        # Draw center line and lane dividers
        self._draw_center_and_lane_lines(screen, segment, start_screen, end_screen, zoom, line_width_px, yellow_line_width_px)
        
        # Draw directional arrows if road is long enough
        if segment.length * zoom > 80:
            self._draw_directional_arrows(screen, segment, start_screen, end_screen, zoom)

    def _draw_edge_lines(self, screen, segment, start_screen, end_screen, zoom, line_width):
        """Draw white edge lines on road shoulders"""
        road_width = segment.width * zoom
        half_width = road_width / 2 - 4  # Inset from edge
        
        # Calculate perpendicular offset
        dx = end_screen[0] - start_screen[0]
        dy = end_screen[1] - start_screen[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
        
        perp_x = -dy / length * half_width
        perp_y = dx / length * half_width
        
        # Left edge line
        left_start = (start_screen[0] + perp_x, start_screen[1] + perp_y)
        left_end = (end_screen[0] + perp_x, end_screen[1] + perp_y)
        pygame.draw.line(screen, (255, 255, 255), left_start, left_end, line_width)
        
        # Right edge line
        right_start = (start_screen[0] - perp_x, start_screen[1] - perp_y)
        right_end = (end_screen[0] - perp_x, end_screen[1] - perp_y)
        pygame.draw.line(screen, (255, 255, 255), right_start, right_end, line_width)

    def _draw_center_and_lane_lines(self, screen, segment, start_screen, end_screen, zoom, line_width, yellow_line_width):
        """Draw center lines and lane dividers"""
        if segment.lane_count <= 1:
            return
            
        road_width = segment.width * zoom
        lane_width = road_width / segment.lane_count
        
        # Calculate perpendicular vector
        dx = end_screen[0] - start_screen[0]
        dy = end_screen[1] - start_screen[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
        
        perp_x = -dy / length
        perp_y = dx / length
        
        # Draw center line (yellow) for two-way roads
        if not segment.oneway and segment.lanes_forward > 0 and segment.lanes_backward > 0:
            center_offset = 0
            center_start = (start_screen[0] + perp_x * center_offset, start_screen[1] + perp_y * center_offset)
            center_end = (end_screen[0] + perp_x * center_offset, end_screen[1] + perp_y * center_offset)
            
            # Solid yellow for no passing zones, dashed for passing allowed
            if segment.lanes_forward == 1 and segment.lanes_backward == 1:
                self._draw_dashed_line(screen, (255, 235, 59), center_start, center_end, yellow_line_width)
            else:
                pygame.draw.line(screen, (255, 235, 59), center_start, center_end, yellow_line_width)
        
        # Draw lane dividers (white dashed)
        for i in range(1, segment.lane_count):
            offset = -road_width / 2 + i * lane_width
            # Skip center line area for two-way roads
            if not segment.oneway and abs(offset) < lane_width * 0.3:
                continue
                
            line_start = (start_screen[0] + perp_x * offset, start_screen[1] + perp_y * offset)
            line_end = (end_screen[0] + perp_x * offset, end_screen[1] + perp_y * offset)
            self._draw_dashed_line(screen, (255, 255, 255), line_start, line_end, line_width)

    def _draw_dashed_line(self, surface, color, start_pos, end_pos, width=1, dash_length=None, gap_length=None):
        """Draw a dashed line with realistic proportions"""
        if dash_length is None:
            dash_length = max(8, int(DASH_LENGTH_M * 2))
        if gap_length is None:
            gap_length = max(8, int(GAP_LENGTH_M * 2))
            
        x1, y1 = start_pos
        x2, y2 = end_pos
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length <= 0:
            return
            
        pattern_len = dash_length + gap_length
        dist = 0.0
        while dist < length:
            seg_end = min(dist + dash_length, length)
            sx = x1 + dx * (dist / length)
            sy = y1 + dy * (dist / length)
            ex = x1 + dx * (seg_end / length)
            ey = y1 + dy * (seg_end / length)
            pygame.draw.line(surface, color, (int(sx), int(sy)), (int(ex), int(ey)), width)
            dist += pattern_len

    def _draw_directional_arrows(self, screen, segment, start_screen, end_screen, zoom):
        """Draw directional arrows on lanes"""
        if not hasattr(segment, 'lane_turns') or not segment.lane_turns:
            return
            
        road_width = segment.width * zoom
        lane_width = road_width / segment.lane_count
        arrow_length = max(12, int(20 * zoom))
        arrow_width = max(8, int(12 * zoom))
        
        # Calculate road direction
        dx = end_screen[0] - start_screen[0]
        dy = end_screen[1] - start_screen[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
            
        road_angle = math.atan2(dy, dx)
        perp_x = -dy / length
        perp_y = dx / length
        
        # Draw arrows at 1/3 and 2/3 along the road
        for t in [0.33, 0.66]:
            arrow_x = start_screen[0] + dx * t
            arrow_y = start_screen[1] + dy * t
            
            for lane_idx in range(min(len(segment.lane_turns), segment.lane_count)):
                lane_offset = -road_width / 2 + (lane_idx + 0.5) * lane_width
                ax = arrow_x + perp_x * lane_offset
                ay = arrow_y + perp_y * lane_offset
                
                # Determine arrow direction
                directions = segment.lane_turns[lane_idx] if segment.lane_turns[lane_idx] else ['through']
                for direction in directions[:1]:  # Only show primary direction
                    if direction.startswith('left'):
                        arrow_angle = road_angle - math.pi / 2
                    elif direction.startswith('right'):
                        arrow_angle = road_angle + math.pi / 2
                    else:  # through, straight, etc.
                        arrow_angle = road_angle
                    
                    self._draw_arrow(screen, (ax, ay), arrow_angle, arrow_length, arrow_width)

    def _draw_arrow(self, screen, center, angle, length, width):
        """Draw a single directional arrow"""
        cx, cy = center
        
        # Arrow tip
        tip_x = cx + math.cos(angle) * length
        tip_y = cy + math.sin(angle) * length
        
        # Arrow base corners
        base_angle1 = angle + math.pi - 0.5
        base_angle2 = angle + math.pi + 0.5
        base_length = length * 0.7
        
        base1_x = cx + math.cos(base_angle1) * base_length
        base1_y = cy + math.sin(base_angle1) * base_length
        base2_x = cx + math.cos(base_angle2) * base_length
        base2_y = cy + math.sin(base_angle2) * base_length
        
        # Arrow shaft
        shaft_length = length * 0.4
        shaft_x = cx - math.cos(angle) * shaft_length
        shaft_y = cy - math.sin(angle) * shaft_length
        
        points = [
            (int(tip_x), int(tip_y)),
            (int(base1_x), int(base1_y)),
            (int(shaft_x), int(shaft_y)),
            (int(base2_x), int(base2_y))
        ]
        
        pygame.draw.polygon(screen, (255, 255, 255), points)

    def _draw_intersections_and_roundabouts(self, screen, camera_x, camera_y, screen_width, screen_height, zoom):
        """Draw intersection blending and roundabouts"""
        # Draw detected roundabouts first
        if hasattr(self, 'roundabouts'):
            for cx, cy, radius in self.roundabouts:
                # Check if roundabout is visible
                view_width = screen_width / zoom
                view_height = screen_height / zoom
                left = camera_x - view_width // 2
                right = camera_x + view_width // 2
                top = camera_y - view_height // 2
                bottom = camera_y + view_height // 2
                
                if (cx < left-radius or cx > right+radius or cy < top-radius or cy > bottom+radius):
                    continue
                    
                sx = int((cx - camera_x) * zoom + screen_width // 2)
                sy = int((cy - camera_y) * zoom + screen_height // 2)
                radius_px = max(8, int(radius * zoom))
                
                self._draw_roundabout(screen, sx, sy, radius_px)
        
        # Draw regular intersections
        if not hasattr(self, 'intersections'):
            return
            
        radius_px = max(8, int(INTERSECTION_RADIUS_M * self.scale_factor * zoom))
        
        for ix, iy in self.intersections:
            # Skip if this is part of a roundabout
            is_roundabout = False
            if hasattr(self, 'roundabouts'):
                for cx, cy, r in self.roundabouts:
                    if math.hypot(ix - cx, iy - cy) < r + 20:  # 20 pixel tolerance
                        is_roundabout = True
                        break
            
            if is_roundabout:
                continue
                
            # Check if intersection is visible
            view_width = screen_width / zoom
            view_height = screen_height / zoom
            left = camera_x - view_width // 2
            right = camera_x + view_width // 2
            top = camera_y - view_height // 2
            bottom = camera_y + view_height // 2
            
            if (ix < left-100 or ix > right+100 or iy < top-100 or iy > bottom+100):
                continue
                
            sx = int((ix - camera_x) * zoom + screen_width // 2)
            sy = int((iy - camera_y) * zoom + screen_height // 2)
            
            # Check if this should be a roundabout (4+ connecting roads)
            connecting_roads = self._count_connecting_roads(ix, iy)
            
            if connecting_roads >= 4 and radius_px > 15:
                self._draw_roundabout(screen, sx, sy, radius_px)
            else:
                self._draw_intersection(screen, sx, sy, radius_px)

    def _count_connecting_roads(self, x, y, threshold=50):
        """Count roads connecting to an intersection point"""
        count = 0
        for segment in self.road_segments:
            start_dist = math.hypot(segment.start[0] - x, segment.start[1] - y)
            end_dist = math.hypot(segment.end[0] - x, segment.end[1] - y)
            if start_dist < threshold or end_dist < threshold:
                count += 1
        return count

    def _draw_roundabout(self, screen, cx, cy, radius):
        """Draw a proper roundabout"""
        # Outer circle (grass/landscape)
        pygame.draw.circle(screen, (80, 120, 60), (cx, cy), int(radius * 1.8))
        
        # Road surface (circular)
        pygame.draw.circle(screen, (65, 65, 70), (cx, cy), radius)
        
        # Inner circle (island)
        inner_radius = max(4, int(radius * 0.4))
        pygame.draw.circle(screen, (90, 140, 70), (cx, cy), inner_radius)
        
        # White circle markings
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), radius, 2)
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), inner_radius, 2)

    def _draw_intersection(self, screen, cx, cy, radius):
        """Draw a standard intersection with blended edges"""
        # Smooth circular blend for intersection
        pygame.draw.circle(screen, (70, 70, 75), (cx, cy), radius)
        
        # Crosswalk markings if intersection is large enough
        if radius > 12:
            self._draw_crosswalk_markings(screen, cx, cy, radius)

    def _draw_crosswalk_markings(self, screen, cx, cy, radius):
        """Draw crosswalk stripes at intersection"""
        stripe_width = 3
        stripe_spacing = 6
        crosswalk_width = radius * 1.2
        
        # Draw crosswalks in four directions
        for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
            start_x = cx + math.cos(angle) * radius * 0.8
            start_y = cy + math.sin(angle) * radius * 0.8
            end_x = cx + math.cos(angle) * radius * 1.4
            end_y = cy + math.sin(angle) * radius * 1.4
            
            # Draw zebra stripes
            perp_angle = angle + math.pi/2
            for i in range(-3, 4):
                offset = i * stripe_spacing
                stripe_start_x = start_x + math.cos(perp_angle) * offset
                stripe_start_y = start_y + math.sin(perp_angle) * offset
                stripe_end_x = end_x + math.cos(perp_angle) * offset
                stripe_end_y = end_y + math.sin(perp_angle) * offset
                
                pygame.draw.line(screen, (255, 255, 255), 
                               (int(stripe_start_x), int(stripe_start_y)), 
                               (int(stripe_end_x), int(stripe_end_y)), stripe_width)

        def draw_dashed_line(surface, color, start_pos, end_pos, width=1, dash_length=12, gap_length=12):
            x1, y1 = start_pos
            x2, y2 = end_pos
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length <= 0:
                return
            pattern_len = dash_length + gap_length
            dist = 0.0
            while dist < length:
                seg_end = min(dist + dash_length, length)
                sx = x1 + dx * (dist / length)
                sy = y1 + dy * (dist / length)
                ex = x1 + dx * (seg_end / length)
                ey = y1 + dy * (seg_end / length)
                pygame.draw.line(surface, color, (int(sx), int(sy)), (int(ex), int(ey)), width)
                dist += pattern_len

    def get_road_bounds(self):
        """Get the bounds of all roads for camera limits"""
        if not self.road_segments:
            return (-500, -500, 500, 500)
        
        min_x = min(min(seg.start[0], seg.end[0]) for seg in self.road_segments)
        max_x = max(max(seg.start[0], seg.end[0]) for seg in self.road_segments)
        min_y = min(min(seg.start[1], seg.end[1]) for seg in self.road_segments)
        max_y = max(max(seg.start[1], seg.end[1]) for seg in self.road_segments)
        
        return min_x, min_y, max_x, max_y
