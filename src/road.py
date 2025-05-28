"""
Road generation and handling from OSM data
"""
import math
import time
import random
import requests
import pygame
from panda3d.core import (Point3, Vec3, GeomVertexFormat, 
                          GeomVertexData, GeomVertexWriter, Geom, 
                          GeomTriangles, GeomNode, GeomTristrips, 
                          TransparencyAttrib, GeomLines)

class RoadSystem:
    def __init__(self, render, noise_texture, car_width):
        self.render = render
        self.noise_texture = noise_texture
        self.road_width = car_width * 5 * 2.0
        self.user_lat = -36.899811100536766
        self.user_lon = 174.94692277461417
        
        self.road_segments = []
        self.spawn_point = None
        
        # Container for road elements
        self.roads_node = None
        self.last_road_side = None

        # Pygame setup for road generation
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Road Generation Simulation")
        self.clock = pygame.time.Clock()

    @property
    def lane_width(self):
        """Calculate and return the lane width based on road width."""
        return self.road_width / 2.0  # Assuming two lanes

    def is_on_road(self, pos):
        """Check if a position is on a road"""
        for a, b, half_width in self.road_segments:
            if self.point_line_distance(pos, a, b) <= half_width:
                return True
        return False

    def get_road_side(self, vehicle_pos, hysteresis=0.5):
        """
        Returns 'left', 'right', or 'center' depending on which side of the road the vehicle is.
        Uses hysteresis to avoid rapid toggling or desync.
        Prints a message if the car is not on the road, and recalibrates when it comes back on.
        """
        if not self.is_on_road(vehicle_pos):
            if self.last_road_side is not None:
                print("Car is not on the road.")
                self.last_road_side = None
            return None  # Not on any road

        segment, road_center, road_direction = self.get_closest_road_segment(vehicle_pos)
        if segment is None:
            if self.last_road_side is not None:
                print("Car is not on the road.")
                self.last_road_side = None
            return None

        # If last_road_side was None (just got back on road), recalibrate
        if self.last_road_side is None:
            # Determine initial side
            to_vehicle = vehicle_pos - road_center
            road_dir_2d = road_direction.getXy()
            to_vehicle_2d = to_vehicle.getXy()
            cross = road_dir_2d.getX() * to_vehicle_2d.getY() - road_dir_2d.getY() * to_vehicle_2d.getX()
            if cross > 0.5:
                side = 'left'
            elif cross < -0.5:
                side = 'right'
            else:
                side = 'center'
            print(f"Vehicle recalibrated to the {side} side of the road.")
            self.last_road_side = side
            return side

        to_vehicle = vehicle_pos - road_center
        road_dir_2d = road_direction.getXy()
        to_vehicle_2d = to_vehicle.getXy()
        cross = road_dir_2d.getX() * to_vehicle_2d.getY() - road_dir_2d.getY() * to_vehicle_2d.getX()

        # Hysteresis thresholds
        center_thresh = 0.1
        side_thresh = hysteresis

        # Determine side with hysteresis
        prev_side = self.last_road_side

        if prev_side == 'left':
            if cross < -side_thresh:
                side = 'right'
            elif abs(cross) < center_thresh:
                side = 'center'
            else:
                side = 'left'
        elif prev_side == 'right':
            if cross > side_thresh:
                side = 'left'
            elif abs(cross) < center_thresh:
                side = 'center'
            else:
                side = 'right'
        else:  # prev_side is None or 'center'
            if cross > side_thresh:
                side = 'left'
            elif cross < -side_thresh:
                side = 'right'
            else:
                side = 'center'

        if side != self.last_road_side:
            print(f"Vehicle moved to the {side} side of the road.")
            self.last_road_side = side

        return side

    def get_closest_road_segment(self, pos):
        """
        Finds the closest road segment to the given position.
        Returns (segment, closest_point, direction_vector).
        segment: (a, b, half_width)
        closest_point: Point3 on the segment
        direction_vector: Vec3 from a to b (normalized)
        """
        min_dist = float('inf')
        closest_segment = None
        closest_point = None
        direction = None
        for a, b, half_width in self.road_segments:
            # Find closest point on segment ab to pos
            ap = Vec3(pos.getX()-a.getX(), pos.getY()-a.getY(), 0)
            ab = Vec3(b.getX()-a.getX(), b.getY()-a.getY(), 0)
            ab_length = ab.length()
            if ab_length == 0:
                continue
            t = ap.dot(ab) / (ab_length**2)
            if t < 0:
                nearest = a
            elif t > 1:
                nearest = b
            else:
                nearest = a + ab * t
            dist = (pos - nearest).length()
            if dist < min_dist:
                min_dist = dist
                closest_segment = (a, b, half_width)
                closest_point = nearest
                direction = ab.normalized()
        if closest_segment is not None:
            return closest_segment, closest_point, direction
        else:
            return None, None, None

        
    def latlon_to_point(self, lat, lon):
        """Convert latitude/longitude to 3D points"""
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * math.cos(math.radians(self.user_lat))
        dx = (lon - self.user_lon) * meters_per_deg_lon
        dy = (lat - self.user_lat) * meters_per_deg_lat
        return Point3(dx, dy, 0)
        
    def load_roads_from_osm(self, lat, lon, radius=500, park_car_model=None):
        """Load road data from OpenStreetMap. Optionally generate parked cars using park_car_model.
           park_car_model should be a NodePath that can be copied (e.g., loaded via loader.loadModel)."""
        query = f"""
        [out:json];
        (
          way(around:{radius},{lat},{lon})["highway"];
        );
        (._;>;);
        out body;
        """
        url = "http://overpass-api.de/api/interpreter"
        retries = 3
        data = None
        for attempt in range(retries):
            try:
                response = requests.post(url, data={'data': query}, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    break
                else:
                    print(f"Attempt {attempt+1}: Received status {response.status_code}. Retrying...")
            except Exception as e:
                print(f"Attempt {attempt+1}: Error retrieving OSM data: {e}")
            time.sleep(1)

        if not data or "elements" not in data:
            print("Failed to retrieve OSM data after retries.")
            return self.render.attachNewNode("osm_roads_empty")

        nodes = {}
        for element in data["elements"]:
            if element["type"] == "node":
                nodes[element["id"]] = element

        self.roads_node = self.render.attachNewNode("osm_roads")
        self.roads_node.setZ(0)
        self.roads_node.setTexture(self.noise_texture, 1)

        drawn = 0
        first_road_center = None

        # Define disallowed road types (no parked cars will be spawned on these)
        disallowed_types = ["motorway", "trunk", "roundabout", "primary"]

        for element in data["elements"]:
            if element["type"] == "way" and "nodes" in element:
                pts = []
                for nid in element["nodes"]:
                    if nid in nodes:
                        node = nodes[nid]
                        pt = self.latlon_to_point(node["lat"], node["lon"])
                        pts.append(pt)
                if len(pts) < 2:
                    continue

                lr_points = []
                for i, p in enumerate(pts):
                    if i == 0:
                        direction = (pts[i+1] - p)
                    elif i == len(pts) - 1:
                        direction = (p - pts[i-1])
                    else:
                        direction = (pts[i+1] - p) + (p - pts[i-1])
                    direction.normalize()
                    perp = Vec3(-direction.getY(), direction.getX(), 0)
                    perp.normalize()
                    left = p + perp * (self.road_width/2.0)
                    right = p - perp * (self.road_width/2.0)
                    lr_points.append((left, right))

                # Compute center of this road
                first_pt = lr_points[0]
                center = (first_pt[0] + first_pt[1]) * 0.5

                if not first_road_center:
                    first_road_center = center

                center_line = [(l + r)*0.5 for l, r in lr_points]
                for i in range(len(center_line)-1):
                    self.road_segments.append((center_line[i], center_line[i+1], self.road_width/2.0))

                # Draw road polygon (existing code)
                vertex_format = GeomVertexFormat.getV3n3()
                vdata = GeomVertexData("road", vertex_format, Geom.UHStatic)
                vertex_writer = GeomVertexWriter(vdata, 'vertex')
                normal_writer = GeomVertexWriter(vdata, 'normal')
                for left, right in lr_points:
                    vertex_writer.addData3f(left)
                    normal_writer.addData3f(0, 0, 1)
                    vertex_writer.addData3f(right)
                    normal_writer.addData3f(0, 0, 1)
                prim = GeomTristrips(Geom.UHStatic)
                vcount = len(lr_points) * 2
                for idx in range(vcount):
                    prim.addVertex(idx)
                prim.closePrimitive()
                geom = Geom(vdata)
                geom.addPrimitive(prim)
                geom_node = GeomNode("road_geom")
                geom_node.addGeom(geom)
                    dash_prim = GeomLines(Geom.UHStatic)
                    dash_length = self.road_width * 0.3   # Length of each dash
                    gap_length = dash_length              # Gap between dashes
                    z_offset = 0.02  # Small offset to avoid z-fighting, but keep flat
                    for i in range(len(center_line) - 1):
                        seg_start = center_line[i]
                        seg_end = center_line[i+1]
                        seg_vec = seg_end - seg_start
                        seg_len = seg_vec.length()
                        if seg_len == 0:
                            continue
                        seg_vec.normalize()
                        travelled = 0
                        while travelled < seg_len:
                            dash_start = seg_start + seg_vec * travelled
                            proposed_dash_end = dash_start + seg_vec * dash_length
                            if (travelled + dash_length) > seg_len:
                                dash_end = seg_end
                            else:
                                dash_end = proposed_dash_end
                            dash_start_flat = Point3(dash_start.getX(), dash_start.getY(), z_offset)
                            dash_end_flat = Point3(dash_end.getX(), dash_end.getY(), z_offset)
                            index = vwriter_lines.getWriteRow()
                            vwriter_lines.addData3f(dash_start_flat)
                            vwriter_lines.addData3f(dash_end_flat)
                            dash_prim.addVertices(index, index + 1)
                            travelled += dash_length + gap_length
                    geom_lines = Geom(vdata_lines)
                    geom_lines.addPrimitive(dash_prim)
                    dash_node = self.roads_node.attachNewNode(GeomNode("dashed_center_line"))
                    dash_node.node().addGeom(geom_lines)
                    dash_node.setColor(1, 1, 1, 1)
                    dash_node.setRenderModeThickness(4)
                    dash_node.setTransparency(TransparencyAttrib.MAlpha)
                # --- End improved dashed center line code ---

                # --- Generate parked cars on the side of eligible roads ---
                road_type = element.get("tags", {}).get("highway", "")
                if road_type not in disallowed_types and len(center_line) >= 2:
                    # If no parked car model is provided, use a box as the parked car.
                    if not park_car_model:
                        park_car_model = self.createbox()
                    # Increase the number of parked cars: more likely to generate cars.
                    min_cars = max(2, len(center_line) // 2)
                    max_cars = max(min_cars, len(center_line))
                    num_cars = random.randint(min_cars, max_cars)
                    for _ in range(num_cars):
                        # Pick a random segment along the center line.
                        seg_index = random.randint(0, len(center_line) - 2)
                        start = center_line[seg_index]
                        end = center_line[seg_index + 1]
                        # Interpolate a position along the segment.
                        f = random.random()
                        pos = start + (end - start) * f
                        # Compute the segment's direction and a perpendicular.
                        direction = end - start
                        if direction.length() == 0:
                            continue
                        direction.normalize()
                        perp = Vec3(-direction.getY(), direction.getX(), 0)
                        # Choose left or right side randomly.
                        side = random.choice([1, -1])
                        # Offset distance: beyond the road edge.
                        offset_distance = (self.road_width / 2.0) + 2
                        parked_pos = pos + perp * side * offset_distance
                        # Compute the heading so the car aligns with the road.
                        heading = math.degrees(math.atan2(direction.getY(), direction.getX()))
                        # Create an instance of the parked car (box).
                        parked_car = park_car_model.copyTo(self.roads_node)
                        parked_car.setPos(parked_pos)
                        parked_car.setH(heading)
                # --- End parked cars generation ---

                drawn += 1

        print(f"Drawn {drawn} roads from OSM.")
        if first_road_center:
            self.spawn_point = first_road_center



    def point_line_distance(self, pt, a, b):
        """Calculate distance from point to line segment"""
        ap = Vec3(pt.getX()-a.getX(), pt.getY()-a.getY(), 0)
        ab = Vec3(b.getX()-a.getX(), b.getY()-a.getY(), 0)
        ab_length = ab.length()
        if ab_length == 0:
            return ap.length()
        t = ap.dot(ab) / (ab_length**2)
        if t < 0:
            nearest = a
        elif t > 1:
            nearest = b
        else:
            nearest = a + ab * t
        return (pt - nearest).length()

    def createbox(self):
        """Create a simple box NodePath as a placeholder car."""
        from panda3d.core import GeomVertexFormat, GeomVertexData, GeomVertexWriter, Geom, GeomTriangles, GeomNode, NodePath
        format = GeomVertexFormat.getV3()
        vdata = GeomVertexData('box', format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, 'vertex')
        # 8 corners of a box (centered at 0,0,0)
        size = 2.0
        h = size * 0.5
        # bottom
        vertex.addData3f(-h, -h, 0)
        vertex.addData3f(h, -h, 0)
        vertex.addData3f(h, h, 0)
        vertex.addData3f(-h, h, 0)
        # top
        vertex.addData3f(-h, -h, size)
        vertex.addData3f(h, -h, size)
        vertex.addData3f(h, h, size)
        vertex.addData3f(-h, h, size)
        triangles = GeomTriangles(Geom.UHStatic)
        # bottom face
        triangles.addVertices(0, 1, 2)
        triangles.addVertices(2, 3, 0)
        # top face
        triangles.addVertices(4, 5, 6)
        triangles.addVertices(6, 7, 4)
        # sides
        triangles.addVertices(0, 1, 5)
        triangles.addVertices(5, 4, 0)
        triangles.addVertices(1, 2, 6)
        triangles.addVertices(6, 5, 1)
        triangles.addVertices(2, 3, 7)
        triangles.addVertices(7, 6, 2)
        triangles.addVertices(3, 0, 4)
        triangles.addVertices(4, 7, 3)
        geom = Geom(vdata)
        geom.addPrimitive(triangles)
        node = GeomNode('box')
        node.addGeom(geom)
        np = NodePath(node)
        np.setColor(1, 0, 0, 1)
        return np

    def lane_offset(self, pos):
        """Returns the signed distance from the given position to the centerline of the closest road segment."""
        segment, center, _ = self.get_closest_road_segment(pos)
        if center is None:
            return 0.0
        # Signed offset: left is negative, right is positive
        to_vehicle = pos - center
        road_dir = (segment[1] - segment[0]).normalized() if segment else None
        if road_dir is not None:
            perp = road_dir.cross(Vec3(0, 0, 1))
            offset = to_vehicle.dot(perp)
            return offset
        return 0.0
