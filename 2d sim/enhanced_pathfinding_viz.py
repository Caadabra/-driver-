import requests
import heapq
import math
import tkinter as tk
from tkinter import messagebox, ttk
import json
import time
import threading
from threading import Lock

class AStarPathfinder:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.canvas_width = 1920
        self.canvas_height = 1080
        self.data_lock = Lock()
        
        # Road type speed limits (km/h) and weights
        self.road_speeds = {
            'motorway': 120,
            'motorway_link': 80,
            'trunk': 100,
            'trunk_link': 70,
            'primary': 80,
            'primary_link': 60,
            'secondary': 60,
            'secondary_link': 50,
            'tertiary': 50,
            'tertiary_link': 40,
            'residential': 30,
            'service': 20,
            'unclassified': 40,
            'living_street': 20,
            'default': 30
        }
        
    def fetch_osm_data(self, bbox):
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json][timeout:60];
        (
          way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|residential|service|unclassified|living_street)$"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
        );
        (._;>;);
        out;
        """
        
        try:
            response = requests.get(overpass_url, params={'data': overpass_query}, timeout=60)
            return response.json()
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def process_osm_data(self, data):
        with self.data_lock:
            self.nodes = {}
            self.edges = {}
            
            if not data:
                return
            
            # Process nodes
            for element in data['elements']:
                if element['type'] == 'node':
                    self.nodes[element['id']] = {
                        'lat': element['lat'],
                        'lon': element['lon']
                    }
            
            # Process ways
            for element in data['elements']:
                if element['type'] == 'way':
                    node_ids = element['nodes']
                    road_type = element.get('tags', {}).get('highway', 'default')
                    speed_limit = self.road_speeds.get(road_type, self.road_speeds['default'])
                    
                    for i in range(len(node_ids) - 1):
                        node1, node2 = node_ids[i], node_ids[i + 1]
                        if node1 in self.nodes and node2 in self.nodes:
                            distance = self.haversine_distance(
                                self.nodes[node1]['lat'], self.nodes[node1]['lon'],
                                self.nodes[node2]['lat'], self.nodes[node2]['lon']
                            )
                            
                            # Calculate travel time in seconds
                            travel_time = (distance / 1000) / speed_limit * 3600
                            
                            if node1 not in self.edges:
                                self.edges[node1] = []
                            if node2 not in self.edges:
                                self.edges[node2] = []
                            
                            self.edges[node1].append((node2, travel_time, road_type, distance))
                            self.edges[node2].append((node1, travel_time, road_type, distance))
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def find_nearest_node(self, lat, lon):
        min_distance = float('inf')
        nearest_node = None
        
        with self.data_lock:
            for node_id, node_data in self.nodes.items():
                distance = self.haversine_distance(lat, lon, node_data['lat'], node_data['lon'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = node_id
        
        return nearest_node
    
    def a_star(self, start, goal, excluded_edges=None, weight_factor=1.0, visualize_callback=None, delay=0.01):
        if excluded_edges is None:
            excluded_edges = set()
        
        with self.data_lock:
            open_set = [(0, start)]
            came_from = {}
            g_score = {start: 0}
            f_score = {start: self.heuristic(start, goal)}
            closed_set = set()
            nodes_explored = 0
            
            while open_set:
                current = heapq.heappop(open_set)[1]
                nodes_explored += 1
                
                closed_set.add(current)
                
                # Visualization callback
                if visualize_callback and nodes_explored % 5 == 0:  # Update every 5 nodes for performance
                    visualize_callback(current, list(closed_set), [node for _, node in open_set], came_from)
                    if delay > 0:
                        time.sleep(delay)
                
                if current == goal:
                    path = []
                    while current in came_from:
                        path.append(current)
                        current = came_from[current]
                    path.append(start)
                    
                    # Final visualization callback
                    if visualize_callback:
                        final_path = path[::-1]
                        visualize_callback(goal, list(closed_set), [], came_from, final_path)
                    
                    return path[::-1]
                
                if current in self.edges:
                    for neighbor, travel_time, road_type, distance in self.edges[current]:
                        edge = (min(current, neighbor), max(current, neighbor))
                        if edge in excluded_edges:
                            continue
                        
                        # Apply weight factor for path variation
                        weighted_travel_time = travel_time * weight_factor
                        tentative_g_score = g_score[current] + weighted_travel_time
                        
                        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def heuristic(self, node1, node2):
        distance = self.haversine_distance(
            self.nodes[node1]['lat'], self.nodes[node1]['lon'],
            self.nodes[node2]['lat'], self.nodes[node2]['lon']
        )
        return (distance / 1000) / 60 * 3600  # Assume 60 km/h average
    
    def find_multiple_paths(self, start, goal, num_paths=5, visualize_callback=None):
        paths = []
        excluded_edges = set()
        
        # Enhanced path generation with different strategies
        strategies = [
            {'weight_factor': 1.0, 'name': 'Optimal'},
            {'weight_factor': 1.2, 'name': 'Slightly longer'},
            {'weight_factor': 0.8, 'name': 'Speed focused'},
            {'weight_factor': 1.5, 'name': 'Alternative route'},
            {'weight_factor': 1.1, 'name': 'Balanced'},
            {'weight_factor': 0.9, 'name': 'Fast route'},
            {'weight_factor': 1.8, 'name': 'Scenic route'},
            {'weight_factor': 0.7, 'name': 'Highway preferred'},
            {'weight_factor': 2.0, 'name': 'Local roads'},
            {'weight_factor': 1.3, 'name': 'Mixed route'}
        ]
        
        # First path - optimal with visualization
        path = self.a_star(start, goal, visualize_callback=visualize_callback if len(paths) == 0 else None)
        if path:
            paths.append(path)
            # Exclude some edges from the optimal path for diversity
            exclude_ratio = 0.3  # Exclude 30% of edges
            edges_to_exclude = max(1, int(len(path) * exclude_ratio))
            for j in range(0, min(edges_to_exclude, len(path) - 1)):
                edge = (min(path[j], path[j+1]), max(path[j], path[j+1]))
                excluded_edges.add(edge)
        
        # Find alternative paths with different strategies
        for i in range(1, min(num_paths, len(strategies))):
            strategy = strategies[i % len(strategies)]
            
            # Create a temporary excluded set with varying exclusion patterns
            temp_excluded = excluded_edges.copy()
            
            # Add some randomness to exclusion for more diverse paths
            if len(paths) > 0:
                for existing_path in paths:
                    # Exclude every nth edge based on strategy
                    step = max(2, len(existing_path) // (i + 2))
                    for j in range(0, len(existing_path) - 1, step):
                        edge = (min(existing_path[j], existing_path[j+1]), max(existing_path[j], existing_path[j+1]))
                        temp_excluded.add(edge)
            
            path = self.a_star(start, goal, temp_excluded, strategy['weight_factor'])
            if path and path not in paths:  # Avoid duplicate paths
                paths.append(path)
                
                # Add some edges from this path to exclusion set
                exclude_step = max(3, len(path) // 4)
                for j in range(0, len(path) - 1, exclude_step):
                    edge = (min(path[j], path[j+1]), max(path[j], path[j+1]))
                    excluded_edges.add(edge)
            else:
                # If no new path found, try with reduced exclusions
                reduced_excluded = set()
                if excluded_edges:
                    # Keep only half of the excluded edges
                    excluded_list = list(excluded_edges)
                    reduced_excluded = set(excluded_list[:len(excluded_list)//2])
                
                path = self.a_star(start, goal, reduced_excluded, strategy['weight_factor'])
                if path and path not in paths:
                    paths.append(path)
        
        return paths

class MapGUI:
    def __init__(self):
        self.pathfinder = AStarPathfinder()
        self.root = tk.Tk()
        self.root.title("asdlkasndlkasd")
        self.root.geometry("1920x1080")

        # City presets (you can modify these)
        self.city_presets = {
            "Berlin Center": [52.45, 13.35, 52.55, 13.45],
            "Manhattan": [40.75, -73.99, 40.78, -73.95],
            "London Center": [51.49, -0.15, 51.52, -0.10],
            "Paris Center": [48.84, 2.32, 48.87, 2.37],
            "Custom Area": [52.5, 13.3, 52.52, 13.32]  # Default
        }
        
        self.setup_ui()
        
        self.bbox = self.city_presets["Berlin Center"]
        self.click_count = 0
        self.start_pos = None
        self.goal_pos = None
        self.current_paths = []
        self.is_loading = False
        self.visualization_mode = False
        
    def setup_ui(self):
        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_panel = tk.Frame(main_container)
        control_panel.pack(fill=tk.X, pady=(0, 10))
        
        # City selection
        city_frame = tk.Frame(control_panel)
        city_frame.pack(side=tk.LEFT, padx=(0, 40))
        
        tk.Label(city_frame, text="Select City:").pack(side=tk.LEFT)
        self.city_var = tk.StringVar(value="Berlin Center")
        city_combo = ttk.Combobox(city_frame, textvariable=self.city_var, 
                                 values=list(self.city_presets.keys()), width=15)
        city_combo.pack(side=tk.LEFT, padx=(5, 0))
        city_combo.bind('<<ComboboxSelected>>', self.on_city_change)
        
        # Control buttons
        button_frame = tk.Frame(control_panel)
        button_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Button(button_frame, text="Load Map", command=self.load_map_data, 
                 bg='lightblue').pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Clear", command=self.clear_canvas, 
                 bg='lightcoral').pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Find More Paths", command=self.find_more_paths, 
                 bg='lightgreen').pack(side=tk.LEFT, padx=2)
        
        # Visualization controls
        viz_frame = tk.Frame(control_panel)
        viz_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        self.viz_var = tk.BooleanVar(value=True)
        tk.Checkbutton(viz_frame, text="Visualize Algorithm", variable=self.viz_var).pack(side=tk.LEFT)
        
        tk.Label(viz_frame, text="Speed:").pack(side=tk.LEFT, padx=(10, 0))
        self.speed_var = tk.DoubleVar(value=0.01)
        speed_scale = tk.Scale(viz_frame, from_=0.001, to=0.1, resolution=0.001, 
                              orient=tk.HORIZONTAL, variable=self.speed_var, length=100)
        speed_scale.pack(side=tk.LEFT)
        
        # Path controls
        path_frame = tk.Frame(control_panel)
        path_frame.pack(side=tk.LEFT)
        
        tk.Label(path_frame, text="Paths:").pack(side=tk.LEFT)
        self.num_paths_var = tk.StringVar(value="8")
        paths_spin = tk.Spinbox(path_frame, from_=3, to=15, width=5, 
                               textvariable=self.num_paths_var)
        paths_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # Status
        self.status_var = tk.StringVar(value="Ready - Select a city and load map")
        status_label = tk.Label(control_panel, textvariable=self.status_var, 
                               fg='blue', font=('Arial', 10))
        status_label.pack(side=tk.RIGHT)
        
        # Canvas with scrollbars
        canvas_frame = tk.Frame(main_container)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, width=1400, height=1000, bg='white')
        
        # Scrollbars
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Path optimization panel
        opt_frame = tk.LabelFrame(main_container, text="Real-time Path Optimization", padx=5, pady=5)
        opt_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Sliders for different optimization parameters
        self.speed_factor = tk.DoubleVar(value=1.0)
        self.distance_factor = tk.DoubleVar(value=1.0)
        self.road_preference = tk.DoubleVar(value=1.0)
        
        slider_frame = tk.Frame(opt_frame)
        slider_frame.pack(fill=tk.X)
        
        # Speed preference
        tk.Label(slider_frame, text="Speed Priority:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        speed_scale = tk.Scale(slider_frame, from_=0.5, to=2.0, resolution=0.1, 
                              orient=tk.HORIZONTAL, variable=self.speed_factor, 
                              command=self.on_parameter_change, length=200)
        speed_scale.grid(row=0, column=1, padx=(0, 20))
        
        # Distance preference
        tk.Label(slider_frame, text="Distance Priority:").grid(row=0, column=2, sticky='w', padx=(0, 10))
        distance_scale = tk.Scale(slider_frame, from_=0.5, to=2.0, resolution=0.1, 
                                 orient=tk.HORIZONTAL, variable=self.distance_factor, 
                                 command=self.on_parameter_change, length=200)
        distance_scale.grid(row=0, column=3, padx=(0, 20))
        
        # Road type preference
        tk.Label(slider_frame, text="Highway Preference:").grid(row=0, column=4, sticky='w', padx=(0, 10))
        road_scale = tk.Scale(slider_frame, from_=0.5, to=2.0, resolution=0.1, 
                             orient=tk.HORIZONTAL, variable=self.road_preference, 
                             command=self.on_parameter_change, length=200)
        road_scale.grid(row=0, column=5)
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        
    def visualization_callback(self, current_node, closed_set, open_set, came_from, final_path=None):
        """Callback function for real-time visualization of the A* algorithm"""
        def update_visualization():
            # Clear previous visualization
            self.canvas.delete("search_viz")
            
            # Draw closed nodes (explored)
            for node_id in closed_set[-50:]:  # Show last 50 to avoid clutter
                if node_id in self.pathfinder.nodes:
                    node = self.pathfinder.nodes[node_id]
                    x, y = self.lat_lon_to_canvas(node['lat'], node['lon'])
                    self.canvas.create_oval(x-2, y-2, x+2, y+2, 
                                          fill='red', outline='darkred', tags="search_viz")
            
            # Draw open nodes (frontier)
            for node_id in open_set[-20:]:  # Show last 20 to avoid clutter
                if node_id in self.pathfinder.nodes:
                    node = self.pathfinder.nodes[node_id]
                    x, y = self.lat_lon_to_canvas(node['lat'], node['lon'])
                    self.canvas.create_oval(x-2, y-2, x+2, y+2, 
                                          fill='yellow', outline='orange', tags="search_viz")
            
            # Draw current node
            if current_node in self.pathfinder.nodes:
                node = self.pathfinder.nodes[current_node]
                x, y = self.lat_lon_to_canvas(node['lat'], node['lon'])
                self.canvas.create_oval(x-4, y-4, x+4, y+4, 
                                      fill='blue', outline='darkblue', width=2, tags="search_viz")
            
            # Draw final path if provided
            if final_path:
                self.canvas.delete("search_viz")  # Clear search visualization
                for i in range(len(final_path) - 1):
                    node1 = self.pathfinder.nodes[final_path[i]]
                    node2 = self.pathfinder.nodes[final_path[i + 1]]
                    x1, y1 = self.lat_lon_to_canvas(node1['lat'], node1['lon'])
                    x2, y2 = self.lat_lon_to_canvas(node2['lat'], node2['lon'])
                    self.canvas.create_line(x1, y1, x2, y2, fill='purple', width=5, tags="search_viz")
            
            self.root.update()
        
        # Schedule the update in the main thread
        self.root.after(0, update_visualization)
        
    def on_city_change(self, event=None):
        selected_city = self.city_var.get()
        if selected_city in self.city_presets:
            self.bbox = self.city_presets[selected_city]
            self.status_var.set(f"Selected {selected_city} - Click 'Load Map' to load data")
    
    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def on_parameter_change(self, value=None):
        if self.start_pos and self.goal_pos and not self.is_loading:
            # Recalculate paths with new parameters
            self.update_paths_with_parameters()
    
    def update_paths_with_parameters(self):
        # This will be called when sliders change
        threading.Thread(target=self.recalculate_paths, daemon=True).start()
    
    def recalculate_paths(self):
        if not (self.start_pos and self.goal_pos):
            return
        
        start_node = self.pathfinder.find_nearest_node(self.start_pos[0], self.start_pos[1])
        goal_node = self.pathfinder.find_nearest_node(self.goal_pos[0], self.goal_pos[1])
        
        if start_node and goal_node:
            # Modify road speeds based on preferences
            original_speeds = self.pathfinder.road_speeds.copy()
            
            # Apply road preference
            highway_types = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link']
            for road_type in highway_types:
                if road_type in self.pathfinder.road_speeds:
                    self.pathfinder.road_speeds[road_type] *= self.road_preference.get()
            
            num_paths = int(self.num_paths_var.get())
            paths = self.pathfinder.find_multiple_paths(start_node, goal_node, num_paths)
            
            # Restore original speeds
            self.pathfinder.road_speeds = original_speeds
            
            self.root.after(0, lambda: self.draw_paths(paths))
    
    def load_map_data(self):
        if self.is_loading:
            return
        
        self.is_loading = True
        self.status_var.set("Loading map data... This may take a moment for large areas")
        self.root.update()
        
        def load_in_thread():
            try:
                data = self.pathfinder.fetch_osm_data(self.bbox)
                self.pathfinder.process_osm_data(data)
                self.root.after(0, self.on_data_loaded)
            except Exception as e:
                self.root.after(0, lambda: self.on_load_error(str(e)))
        
        threading.Thread(target=load_in_thread, daemon=True).start()
    
    def on_data_loaded(self):
        self.draw_map()
        self.is_loading = False
        node_count = len(self.pathfinder.nodes)
        edge_count = sum(len(edges) for edges in self.pathfinder.edges.values()) // 2
        self.status_var.set(f"Map loaded: {node_count} nodes, {edge_count} roads. Click to set start and end points.")
    
    def on_load_error(self, error):
        self.is_loading = False
        self.status_var.set(f"Error loading map: {error}")
        messagebox.showerror("Loading Error", f"Failed to load map data: {error}")
        
    def lat_lon_to_canvas(self, lat, lon):
        x = ((lon - self.bbox[1]) / (self.bbox[3] - self.bbox[1])) * self.pathfinder.canvas_width
        y = ((self.bbox[2] - lat) / (self.bbox[2] - self.bbox[0])) * self.pathfinder.canvas_height
        return int(x), int(y)
    
    def canvas_to_lat_lon(self, x, y):
        lon = self.bbox[1] + (x / self.pathfinder.canvas_width) * (self.bbox[3] - self.bbox[1])
        lat = self.bbox[2] - (y / self.pathfinder.canvas_height) * (self.bbox[2] - self.bbox[0])
        return lat, lon
    
    def get_road_color(self, road_type):
        colors = {
            'motorway': '#FF0000',      # Red
            'motorway_link': '#FF3333', # Light red
            'trunk': '#FF6600',         # Orange
            'trunk_link': '#FF8833',    # Light orange
            'primary': '#FFCC00',       # Yellow
            'primary_link': '#FFDD33',  # Light yellow
            'secondary': '#66FF66',     # Light green
            'secondary_link': '#88FF88',# Very light green
            'tertiary': '#66CCFF',      # Light blue
            'tertiary_link': '#88DDFF', # Very light blue
            'residential': '#CCCCCC',   # Light gray
            'service': '#999999',       # Gray
            'unclassified': '#DDDDDD',  # Very light gray
            'living_street': '#FFAAFF', # Pink
            'default': '#AAAAAA'        # Default gray
        }
        return colors.get(road_type, colors['default'])
    
    def draw_map(self):
        self.canvas.delete("road")
        
        with self.pathfinder.data_lock:
            for node_id, neighbors in self.pathfinder.edges.items():
                if node_id in self.pathfinder.nodes:
                    node1 = self.pathfinder.nodes[node_id]
                    x1, y1 = self.lat_lon_to_canvas(node1['lat'], node1['lon'])
                    
                    for neighbor_id, travel_time, road_type, distance in neighbors:
                        if neighbor_id > node_id and neighbor_id in self.pathfinder.nodes:  # Avoid drawing twice
                            node2 = self.pathfinder.nodes[neighbor_id]
                            x2, y2 = self.lat_lon_to_canvas(node2['lat'], node2['lon'])
                            color = self.get_road_color(road_type)
                            width = 3 if 'motorway' in road_type else 2 if 'trunk' in road_type else 1
                            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, tags="road")
    
    def clear_canvas(self):
        self.canvas.delete("path")
        self.canvas.delete("start")
        self.canvas.delete("goal")
        self.canvas.delete("search_viz")
        self.click_count = 0
        self.start_pos = None
        self.goal_pos = None
        self.current_paths = []
        self.status_var.set("Cleared. Click to set new start and end points.")
    
    def on_click(self, event):
        if self.is_loading:
            return
        
        if self.click_count == 0:
            self.canvas.delete("path")
            self.canvas.delete("start")
            self.canvas.delete("goal")
            self.canvas.delete("search_viz")
        
        lat, lon = self.canvas_to_lat_lon(event.x, event.y)
        
        if self.click_count == 0:
            self.start_pos = (lat, lon)
            self.canvas.create_oval(event.x-6, event.y-6, event.x+6, event.y+6, 
                                  fill='green', outline='black', width=2, tags="start")
            self.click_count = 1
            self.status_var.set("Start point set. Click to set destination.")
        elif self.click_count == 1:
            self.goal_pos = (lat, lon)
            self.canvas.create_oval(event.x-6, event.y-6, event.x+6, event.y+6, 
                                  fill='red', outline='black', width=2, tags="goal")
            
            self.status_var.set("Calculating multiple paths...")
            self.root.update()
            threading.Thread(target=self.find_and_draw_paths, daemon=True).start()
            self.click_count = 0
    
    def find_and_draw_paths(self):
        start_node = self.pathfinder.find_nearest_node(self.start_pos[0], self.start_pos[1])
        goal_node = self.pathfinder.find_nearest_node(self.goal_pos[0], self.goal_pos[1])
        
        if start_node and goal_node:
            num_paths = int(self.num_paths_var.get())
            
            # Use visualization callback if enabled
            viz_callback = self.visualization_callback if self.viz_var.get() else None
            
            paths = self.pathfinder.find_multiple_paths(start_node, goal_node, num_paths, viz_callback)
            self.current_paths = paths
            self.root.after(0, lambda: self.draw_paths(paths))
        else:
            self.root.after(0, lambda: self.status_var.set("Error: Could not find nodes near clicked points"))
    
    def draw_paths(self, paths):
        self.canvas.delete("path")
        colors = ['#0000FF', '#FF8800', '#8800FF', '#00FF88', "#9A2A66", 
                 '#88FF00', '#0088FF', '#FF0088', '#8888FF', '#FF88FF',
                 '#00FFFF', '#FF00FF', '#FFFF00', '#80FF80', '#FF8080']
        
        if paths:
            for i, path in enumerate(paths):
                if path:
                    color = colors[i % len(colors)]
                    width = 5 if i == 0 else 3
                    
                    for j in range(len(path) - 1):
                        node1 = self.pathfinder.nodes[path[j]]
                        node2 = self.pathfinder.nodes[path[j + 1]]
                        x1, y1 = self.lat_lon_to_canvas(node1['lat'], node1['lon'])
                        x2, y2 = self.lat_lon_to_canvas(node2['lat'], node2['lon'])
                        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, tags="path")
            
            self.status_var.set(f"Found {len(paths)} alternative paths with enhanced strategies. Red dots = explored nodes.")
        else:
            self.status_var.set("No paths found between selected points.")
    
    def find_more_paths(self):
        if self.start_pos and self.goal_pos and not self.is_loading:
            self.status_var.set("Finding additional paths with enhanced strategies...")
            # Increase the number of paths temporarily
            current_num = int(self.num_paths_var.get())
            self.num_paths_var.set(str(min(15, current_num + 3)))
            threading.Thread(target=self.find_and_draw_paths, daemon=True).start()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MapGUI()
    app.run()
