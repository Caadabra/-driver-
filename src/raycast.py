import math

class Raycast:
    def cast(self, vehicle, road, num_rays=5, max_dist=100):
        # Define a field-of-view in radians (e.g. 90 degrees)
        fov = math.pi / 2
        start_angle = vehicle.angle - fov / 2
        ray_angles = [start_angle + i * (fov / (num_rays - 1)) for i in range(num_rays)]
        distances = []

        for angle in ray_angles:
            ray_origin = (vehicle.x, vehicle.y)
            ray_direction = (math.cos(angle), math.sin(angle))
            closest_dist = max_dist

            # Iterate over all road boundary segments
            for seg in road.boundaries:
                # Each segment is a tuple: ((x1,y1), (x2,y2))
                pt1, pt2 = seg
                intersect = self.ray_segment_intersect(ray_origin, ray_direction, pt1, pt2)
                if intersect is not None and intersect < closest_dist:
                    closest_dist = intersect

            distances.append(closest_dist)
        return distances

    def ray_segment_intersect(self, ray_origin, ray_direction, seg_start, seg_end):
        # Unpack points
        (x1, y1) = ray_origin
        (dx, dy) = ray_direction
        (x3, y3) = seg_start
        (x4, y4) = seg_end

        # Vector for segment
        seg_dx = x4 - x3
        seg_dy = y4 - y3

        # Calculate denominator (if zero, lines are parallel)
        denom = dx * seg_dy - dy * seg_dx
        if denom == 0:
            return None

        # Calculate parameters t for the ray and u for the segment
        t = ((x3 - x1) * seg_dy - (y3 - y1) * seg_dx) / denom
        u = ((x3 - x1) * dy - (y3 - y1) * dx) / denom

        # Check if there is an intersection:
        # t must be non-negative (in front of the ray) and u between 0 and 1 (within the segment)
        if t >= 0 and 0 <= u <= 1:
            return t  # distance along the ray
        return None
