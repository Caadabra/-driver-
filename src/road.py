"""
Road generation and handling from OSM data
"""
import math
import time
import requests
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
        
    def latlon_to_point(self, lat, lon):
        """Convert latitude/longitude to 3D points"""
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * math.cos(math.radians(self.user_lat))
        dx = (lon - self.user_lon) * meters_per_deg_lon
        dy = (lat - self.user_lat) * meters_per_deg_lat
        return Point3(dx, dy, 0)
        
    def load_roads_from_osm(self, lat, lon, radius=500):
        """Load road data from OpenStreetMap"""
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
                    first_pt = lr_points[0]
                    center = (first_pt[0] + first_pt[1]) * 0.5
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
                road_np = self.roads_node.attachNewNode(geom_node)
                road_np.setColor(0.2, 0.2, 0.2, 1)
                road_np.setTexture(self.noise_texture, 1)

                # --- Begin dashed center line code ---
                if len(center_line) >= 2:
                    vertex_format_lines = GeomVertexFormat.getV3()
                    vdata_lines = GeomVertexData("dashes", vertex_format_lines, Geom.UHStatic)
                    vwriter_lines = GeomVertexWriter(vdata_lines, 'vertex')
                    dash_prim = GeomLines(Geom.UHStatic)
                    dash_length = self.road_width * 0.3   # Length of each dash
                    gap_length = dash_length              # Gap between dashes
                    # Process each segment of the center line.
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
                            # Clamp dash_end so it doesn't overshoot seg_end.
                            if (travelled + dash_length) > seg_len:
                                dash_end = seg_end
                            else:
                                dash_end = proposed_dash_end
                            index = vwriter_lines.getWriteRow()
                            vwriter_lines.addData3f(dash_start)
                            vwriter_lines.addData3f(dash_end)
                            dash_prim.addVertices(index, index + 1)
                            travelled += dash_length + gap_length
                    geom_lines = Geom(vdata_lines)
                    geom_lines.addPrimitive(dash_prim)
                    dash_node = self.roads_node.attachNewNode(GeomNode("dashed_center_line"))
                    dash_node.node().addGeom(geom_lines)
                    dash_node.setColor(1, 1, 1, 1)  # White dashed line
                # --- End dashed center line code ---

                drawn += 1


        print(f"Drawn {drawn} roads from OSM.")
        if first_road_center:
            self.spawn_point = first_road_center 
            
        return self.roads_node

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

    def is_on_road(self, pos):
        """Check if a position is on a road"""
        for a, b, half_width in self.road_segments:
            if self.point_line_distance(pos, a, b) <= half_width:
                return True
        return False
