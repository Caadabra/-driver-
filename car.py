"""

 ██████  █████  ██████     ██████  ██    ██ 
██      ██   ██ ██   ██    ██   ██  ██  ██  
██      ███████ ██████     ██████    ████   
██      ██   ██ ██   ██    ██         ██    
 ██████ ██   ██ ██   ██ ██ ██         ██    
                                            
                                            
 
Car class and related functionality for the AI driving simulation.
Contains the main Car class with AI behavior, pathfinding, and physics.
"""

import pygame
import math
import torch
from collections import deque

from ai_models import SimpleCarAI
from constants import *


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
            # AI diagnostics
            self.ai_param_count = sum(p.numel() for p in self.ai.parameters())
            # Capture first linear layer weight stats
            try:
                first_layer = [m for m in self.ai.network if isinstance(m, torch.nn.Linear)][0]
                w = first_layer.weight.detach().cpu().numpy()
                self.ai_w_min = float(w.min())
                self.ai_w_max = float(w.max())
                self.ai_w_mean = float(w.mean())
            except Exception:
                self.ai_w_min = self.ai_w_max = self.ai_w_mean = 0.0
            self.ai_accel_history = deque(maxlen=180)  # 3 seconds @60fps
            self.ai_steer_history = deque(maxlen=180)
            self.ai_accel_var = 0.0
            self.ai_steer_var = 0.0
            self.ai_accel_mean = 0.0
            self.ai_steer_mean = 0.0
            self.ai_stability_index = 0.0
            # Thinking stream log (cognitive trace)
            self.thinking_log = deque(maxlen=300)

        # Control visualization
        self.ai_acceleration = 0
        self.ai_steering = 0

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

    # ----- Waypoint utilities (lane offset, resample, prune) -----
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

    def _estimate_segment_width_at(self, x: float, y: float, search_radius: float = 80.0) -> float:
        """Estimate local road width (in px) near a point by querying nearest segment."""
        if not getattr(self, 'road_system', None) or not getattr(self.road_system, 'spatial_grid', None):
            return 40.0  # sensible default
        try:
            nearby = self.road_system.spatial_grid.get_nearby_segments(x, y, radius=search_radius)
            best_seg = None
            best_d2 = float('inf')
            for seg in nearby:
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
                    best_seg = seg
            if best_seg is not None and getattr(best_seg, 'width', None):
                return float(best_seg.width)
        except Exception:
            pass
        return 40.0

    def _offset_waypoints_for_left_hand(self, lane_ratio: float = 0.25):
        """Offset path_waypoints to left side of travel to obey left-hand traffic.

        lane_ratio is fraction of local road width to offset from the centerline.
        """
        pts = self.path_waypoints
        if not pts or len(pts) < 2:
            return
        new_pts = []
        n = len(pts)
        for i in range(n):
            if i < n - 1:
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
            else:
                x0, y0 = pts[i - 1]
                x1, y1 = pts[i]
            vx = x1 - x0
            vy = y1 - y0
            length = math.hypot(vx, vy)
            if length < 1e-6:
                new_pts.append(pts[i])
                continue
            nx = -vy / length
            ny = vx / length
            mx = (x0 + x1) * 0.5
            my = (y0 + y1) * 0.5
            road_w = self._estimate_segment_width_at(mx, my)
            offset = max(6.0, min(road_w * lane_ratio, max(8.0, road_w * 0.4)))
            ox = nx * offset
            oy = ny * offset
            new_pts.append((pts[i][0] + ox, pts[i][1] + oy))
        self.path_waypoints = new_pts
        self._rebuild_path_geometry()

    def _resample_waypoints_evenly(self, spacing=None):
        """Resample current path_waypoints so checkpoints are evenly spaced.
        If spacing not provided, compute adaptively from total path length."""
        if not self.path_waypoints or len(self.path_waypoints) < 2:
            return
        pts = self.path_waypoints
        dists = [0.0]
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i-1][0]
            dy = pts[i][1] - pts[i-1][1]
            dists.append(dists[-1] + math.hypot(dx, dy))
        total = dists[-1]
        if spacing is None:
            spacing = compute_adaptive_waypoint_spacing(total)
        if total < spacing * 0.5:
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

    def _prune_backward_waypoints(self):
        """Remove leading waypoints that lie behind the current heading.

        Prevent immediate U-turns caused by initial waypoint behind the car.
        """
        if not self.path_waypoints or len(self.path_waypoints) < 2:
            return
        heading_rad = math.radians(self.angle)
        fwd = (math.sin(heading_rad), -math.cos(heading_rad))
        pruned = False
        while len(self.path_waypoints) > 1:
            wx, wy = self.path_waypoints[0]
            dx = wx - self.x; dy = wy - self.y
            dist = math.hypot(dx, dy)
            if dist < 5:
                self.path_waypoints.pop(0)
                pruned = True
                continue
            dot = dx * fwd[0] + dy * fwd[1]
            if dot < 0:
                self.path_waypoints.pop(0)
                pruned = True
            else:
                break
        if pruned:
            self._rebuild_path_geometry()
            self.current_waypoint_index = 0
            if self.path_waypoints:
                self.target_x, self.target_y = self.path_waypoints[0]

    def _in_roundabout_zone(self) -> bool:
        """Return True if the car is within or near a roundabout ring to tighten checkpoint logic."""
        if not self.road_system or not hasattr(self.road_system, 'roundabouts'):
            return False
        for (cx, cy, r) in getattr(self.road_system, 'roundabouts', []):
            # Treat within ring plus small margin as roundabout zone
            # Use outer radius with margin; inner island excluded by off-road check anyway
            if math.hypot(self.x - cx, self.y - cy) <= (r + 40):
                return True
        return False

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

    def _estimate_segment_width_at(self, x: float, y: float, search_radius: float = 80.0) -> float:
        """Estimate local road width (in px) near a point by querying nearest segment."""
        if not getattr(self, 'road_system', None) or not getattr(self.road_system, 'spatial_grid', None):
            return 40.0  # sensible default
        try:
            nearby = self.road_system.spatial_grid.get_nearby_segments(x, y, radius=search_radius)
            best_seg = None
            best_d2 = float('inf')
            for seg in nearby:
                # Distance to centerline squared
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
                    best_seg = seg
            if best_seg is not None and getattr(best_seg, 'width', None):
                return float(best_seg.width)
        except Exception:
            pass
        return 40.0

    def _offset_waypoints_for_left_hand(self, lane_ratio: float = 0.25):
        """Offset path_waypoints to left side of travel to obey left-hand traffic.

        lane_ratio is fraction of local road width to offset from the centerline.
        """
        pts = self.path_waypoints
        if not pts or len(pts) < 2:
            return
        new_pts = []
        n = len(pts)
        for i in range(n):
            # Use segment direction based on neighbor to define "left"
            if i < n - 1:
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
            else:
                # For last point, reuse previous segment direction
                x0, y0 = pts[i - 1]
                x1, y1 = pts[i]
            vx = x1 - x0
            vy = y1 - y0
            length = math.hypot(vx, vy)
            if length < 1e-6:
                new_pts.append(pts[i])
                continue
            # Perpendicular pointing to the left of travel
            nx = -vy / length
            ny = vx / length
            # Estimate local road width near this waypoint (midpoint)
            mx = (x0 + x1) * 0.5
            my = (y0 + y1) * 0.5
            road_w = self._estimate_segment_width_at(mx, my)
            # Clamp offset to a reasonable range
            offset = max(6.0, min(road_w * lane_ratio, max(8.0, road_w * 0.4)))
            ox = nx * offset
            oy = ny * offset
            new_pts.append((pts[i][0] + ox, pts[i][1] + oy))
        self.path_waypoints = new_pts
        self._rebuild_path_geometry()
    
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
                        # Keep to the left lane
                        self._offset_waypoints_for_left_hand()
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
        # Keep to the left lane
        self._offset_waypoints_for_left_hand()
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
            
            # Ensure exactly 11 inputs
            while len(inputs) < 11:
                inputs.append(0.0)
            inputs = inputs[:11]
            
            # Get AI decision for this simulated state
            input_tensor = torch.tensor(inputs, dtype=torch.float32)
            with torch.no_grad():
                outputs = self.ai(input_tensor)
            
            # Extract AI decisions
            acceleration = outputs[0].item()
            steering = outputs[1].item()
            
            # Apply AI decisions to simulated car state
            if acceleration > 0:
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
        screen = pygame.display.get_surface()
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
    
    def draw_route(self, camera, is_best=False):
        """Draw the pathfinding route waypoints and current target"""
        if not self.path_waypoints:
            return
        # Pathfinding lines are intentionally hidden (green/yellow segments removed)
    
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
            def _pressed(k):
                try:
                    return bool(keys[k])
                except Exception:
                    # keys may be a dict; fall back to .get
                    try:
                        return bool(keys.get(k, False))
                    except Exception:
                        return False

            if _pressed(pygame.K_UP):
                self.speed += 0.2
            if _pressed(pygame.K_DOWN):  # reverse / brake
                if self.can_reverse:
                    self.speed -= 0.15  # allow gentle reverse acceleration
                else:
                    self.speed -= 0.3
            # Only allow turning if car has some speed
            if abs(self.speed) > 0.1:
                if _pressed(pygame.K_LEFT):
                    self.angle -= 5
                if _pressed(pygame.K_RIGHT):
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
        # Tighten checkpoint radius when inside a roundabout zone to enforce correct circulation
        reach_dist = 18 if self._in_roundabout_zone() else self.waypoint_reach_distance
        if dist <= reach_dist:
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
        screen = pygame.display.get_surface()
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
        # Update diagnostics histories
        if self.use_ai:
            self.ai_accel_history.append(acceleration)
            self.ai_steer_history.append(steering)
            if len(self.ai_accel_history) > 5:
                # Compute running stats cheaply
                ah = list(self.ai_accel_history)
                sh = list(self.ai_steer_history)
                import statistics as _stats
                try:
                    self.ai_accel_mean = _stats.fmean(ah)
                    self.ai_steer_mean = _stats.fmean(sh)
                    self.ai_accel_var = _stats.pvariance(ah)
                    self.ai_steer_var = _stats.pvariance(sh)
                except Exception:
                    pass
                # Stability index (lower variance & moderate magnitude preferred)
                total_var = self.ai_accel_var + self.ai_steer_var
                self.ai_stability_index = max(0.0, 1.0 - min(1.0, total_var * 3.0))
                # Thinking log entry (neutral technical record)
                try:
                    if hasattr(self, 'thinking_log'):
                        min_clear = 0.0
                        if self.raycast_distances:
                            min_clear = min(self.raycast_distances)
                        curvature = 0.0
                        if self.path_waypoints and self.current_waypoint_index < len(self.path_waypoints)-2:
                            i = self.current_waypoint_index
                            a1x,a1y = self.path_waypoints[i]; a2x,a2y = self.path_waypoints[i+1]; a3x,a3y = self.path_waypoints[i+2]
                            v1x=a2x-a1x; v1y=a2y-a1y; v2x=a3x-a2x; v2y=a3y-a2y
                            l1=math.hypot(v1x,v1y); l2=math.hypot(v2x,v2y)
                            if l1>1e-3 and l2>1e-3:
                                h1=math.degrees(math.atan2(v1x,-v1y)); h2=math.degrees(math.atan2(v2x,-v2y))
                                diff=h2-h1
                                while diff>180: diff-=360
                                while diff<-180: diff+=360
                                curvature=abs(diff)
                        lane_dev = getattr(self, 'lateral_deviation', 0.0)
                        progress_pct = 0.0
                        if self.path_waypoints:
                            progress_pct = 100.0 * self.current_waypoint_index / max(1, len(self.path_waypoints)-1)
                        # Derive action label (neutral)
                        if curvature > 55:
                            act = 'curve_prep'
                        elif lane_dev > 6:
                            act = 'lane_adjust'
                        elif acceleration > 0.25:
                            act = 'accelerate'
                        elif acceleration < -0.25:
                            act = 'decelerate'
                        elif abs(steering) > 0.35:
                            act = 'steer_refine'
                        else:
                            act = 'maintain'
                        entry = (f"t={self.time_alive:05d} v={self.speed:4.2f} a={acceleration:+.2f} s={steering:+.2f} "
                                 f"clr={min_clear:5.1f} curv={curvature:4 .1f} dev={lane_dev:4.1f} prog={progress_pct:5.1f}% act={act}")
                        self.thinking_log.append(entry)
                except Exception:
                    pass
        # Apply acceleration
        if acceleration > 0.01:
            self.speed += acceleration * 0.25
        elif acceleration < -0.01:
            self.speed += acceleration * 0.25  # allow reverse if needed
        # Apply steering
        self.angle += steering * 10.0  # scale steering output
        
        # Clamp speeds (prevent casual negative speeds)
        if self.speed > self.max_forward_speed: self.speed = self.max_forward_speed
        if self.speed < -self.max_reverse_speed: self.speed = -self.max_reverse_speed

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