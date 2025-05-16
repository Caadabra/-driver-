"""
Road generation and handling from OSM data
"""
import math
import time
import requests
from panda3d.core import (Point3, Vec3, NodePath, GeomVertexFormat, 
                          GeomVertexData, GeomVertexWriter, Geom, 
                          GeomTriangles, GeomNode, GeomTristrips, 
                          CollisionNode, CollisionBox, TransparencyAttrib)

class RoadSystem:
    def __init__(self, render, noise_texture, car_width):
        self.render = render
        self.noise_texture = noise_texture
        self.road_width = car_width * 5 * 2.0
        self.curb_thickness = 0.5
        self.curb_height = 0.5
        self.user_lat = -36.899811100536766
        self.user_lon = 174.94692277461417
        
        self.road_segments = []
        self.drawn_curb_keys = set()  # For deduplicating curb walls
        self.spawn_point = None
        
        # Containers for road elements
        self.roads_node = None
        self.curb_node = None
        self.curb_collisions = None
        
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

        self.curb_node = self.render.attachNewNode("road_curbs")
        self.curb_node.setZ(0)
        self.curb_node.setTexture(self.noise_texture, 1)

        self.curb_collisions = self.render.attachNewNode("curb_collisions")

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
                curb_left_pts = []
                curb_right_pts = []
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
                    curb_left = left + perp * self.curb_thickness
                    curb_right = right - perp * self.curb_thickness
                    curb_left_pts.append(curb_left)
                    curb_right_pts.append(curb_right)

                if not first_road_center:
                    first_pt = lr_points[0]
                    center = (first_pt[0] + first_pt[1]) * 0.5
                    first_road_center = center

                center_line = [(l+r)*0.5 for l, r in lr_points]
                for i in range(len(center_line)-1):
                    self.road_segments.append((center_line[i], center_line[i+1], self.road_width/2.0))

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

                self.make_curb_wall(
                    [pt for pt, _ in lr_points], 
                    curb_left_pts, 
                    self.curb_height)
                self.make_curb_wall(
                    [pt for _, pt in lr_points], 
                    curb_right_pts, 
                    self.curb_height, 
                    flip=True)

                drawn += 1

        print(f"Drawn {drawn} roads from OSM.")
        if first_road_center:
            self.spawn_point = first_road_center + Vec3(0,0,0)
            
        return self.roads_node

    def make_curb_wall(self, inner_pts, outer_pts, curb_height, flip=False):
        """Create a curb wall with collision detection"""
        vertex_format = GeomVertexFormat.getV3n3()
        vdata = GeomVertexData("curb", vertex_format, Geom.UHStatic)
        vw = GeomVertexWriter(vdata, 'vertex')
        nw = GeomVertexWriter(vdata, 'normal')
        prim = GeomTriangles(Geom.UHStatic)
        vert_count = 0

        # Construct the curb wall using inner_pts for the base and outer_pts for the top.
        for i in range(len(inner_pts)-1):
            bl = inner_pts[i]
            br = inner_pts[i+1]
            # Use outer_pts to calculate the top vertices with added curb_height.
            tl = Point3(outer_pts[i].getX(), outer_pts[i].getY(), outer_pts[i].getZ() + curb_height)
            tr = Point3(outer_pts[i+1].getX(), outer_pts[i+1].getY(), outer_pts[i+1].getZ() + curb_height)

            edge = Vec3(br - bl)
            vertical = Vec3(0, 0, 1)
            face_normal = edge.cross(vertical)
            face_normal.normalize()
            if flip:
                face_normal *= -1

            for pt in [bl, br, tr, tl]:
                vw.addData3f(pt)
                nw.addData3f(face_normal)

            if not flip:
                prim.addVertices(vert_count, vert_count+1, vert_count+2)
                prim.addVertices(vert_count, vert_count+2, vert_count+3)
            else:
                prim.addVertices(vert_count, vert_count+2, vert_count+1)
                prim.addVertices(vert_count, vert_count+3, vert_count+2)
            vert_count += 4

            mid = (bl + br + tl + tr) * 0.25
            length = (br - bl).length()
            half_x = length/2.0
            half_y = 0.25
            half_z = curb_height/2.0
            cb = CollisionBox(mid, half_x, half_y, half_z)
            coll_node = CollisionNode("curb_segment")
            coll_node.addSolid(cb)
            self.curb_collisions.attachNewNode(coll_node)

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        geom_node = GeomNode("curb_geom")
        geom_node.addGeom(geom)
        curb_np = self.curb_node.attachNewNode(geom_node)
        curb_np.setTexture(self.noise_texture, 1)
        curb_np.setTransparency(TransparencyAttrib.MAlpha)
        
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
