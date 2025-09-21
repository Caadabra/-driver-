import sys
import os
import json
import math
import random
import threading
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget
from PyQt6.QtCore import pyqtSignal, QUrl, QProcess, QTimer, pyqtSlot
import hashlib
import base64
import secrets
from pathlib import Path
import re
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel

# Import OSM roads and pathfinding
from osm_roads import OSMRoadSystem
from dijkstra_pathfinding import DijkstraPathfinder


    


class FluidWebView(QWebEngineView):
    # Internal signals for app wiring (avoid clashing with JS-exposed slot names)
    ui_clicked = pyqtSignal()
    ui_search_submitted = pyqtSignal(str)
    ui_route_selected = pyqtSignal(int)
    routes_found = pyqtSignal(list)
    zoomFactorChanged = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: black;")
        
        # PIN storage path (user home, not in project folder)
        self._pin_file = Path.home() / ".driver_pin.json"
        
        # Initialize OSM road system and pathfinding
        self.road_system = None
        self.pathfinder = None
        self.routes = []
        self.start_point = None
        self.end_point = None
        
        # Initialize routes
        self.initialize_routes()
        
        # Setup web channel for JS-Python communication
        self.channel = QWebChannel()
        self.page().setWebChannel(self.channel)
        self.channel.registerObject("backend", self)
        
        # Create HTML content with embedded fluid simulation
        html_content = self.create_html_content()
        
        # Load the HTML content
        self.setHtml(html_content)
        
        # Connect page loaded signal
        self.loadFinished.connect(self.on_load_finished)
        
        # Initialize map system
        self.initialize_map_system()

    # Legacy embedded sim process removed

    # ===== PIN MANAGEMENT (exposed to JS via WebChannel) =====
    def _load_pin_record(self):
        try:
            if self._pin_file.exists():
                with open(self._pin_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Failed to read PIN file: {e}")
        return None

    def _save_pin_record(self, record: dict):
        try:
            with open(self._pin_file, 'w', encoding='utf-8') as f:
                json.dump(record, f)
            return True
        except Exception as e:
            print(f"Failed to write PIN file: {e}")
            return False

    def _hash_pin(self, pin: str, salt: bytes | None = None):
        if salt is None:
            salt = secrets.token_bytes(16)
        # PBKDF2-HMAC-SHA256
        dk = hashlib.pbkdf2_hmac('sha256', pin.encode('utf-8'), salt, 200_000)
        return base64.b64encode(salt).decode('ascii'), base64.b64encode(dk).decode('ascii')

    @pyqtSlot(result=bool)
    def has_pin(self) -> bool:
        """Return True if a check string (PIN/key) is already set."""
        rec = self._load_pin_record()
        if not rec:
            return False
        # Support both plaintext ('pin') and legacy hashed ('salt'+'hash') records
        return bool(rec.get('pin')) or bool(rec.get('salt') and rec.get('hash'))

    @pyqtSlot(str, result=bool)
    def set_pin(self, pin: str) -> bool:
        """Set a new plain check string if none exists yet; returns False if already set or invalid."""
        if not pin or len(pin) < 4 or len(pin) > 12:
            return False
        # Only allow letters and digits
        if not re.fullmatch(r"[A-Za-z0-9]{4,12}", pin):
            return False
        if self.has_pin():
            return False
        # Store plaintext check string per spec
        record = {"pin": pin, "ver": "plain1"}
        return self._save_pin_record(record)

    @pyqtSlot(str, result=bool)
    def verify_pin(self, pin: str) -> bool:
        """Verify entered check string against stored record (plaintext preferred; legacy hash supported)."""
        if not pin:
            return False
        # Enforce same length to avoid accidental short/long strings
        if len(pin) < 4 or len(pin) > 12:
            return False
        if not re.fullmatch(r"[A-Za-z0-9]{4,12}", pin):
            return False
        rec = self._load_pin_record()
        if not rec:
            return False
        # Prefer plaintext verification
        if 'pin' in rec:
            try:
                return hashlib.compare_digest(rec.get('pin', ''), pin)
            except Exception:
                return rec.get('pin', '') == pin
        # Fallback to legacy hashed verification
        if 'salt' in rec and 'hash' in rec:
            try:
                salt = base64.b64decode(rec['salt'])
                expected = base64.b64decode(rec['hash'])
                _, computed_b64 = self._hash_pin(pin, salt)
                computed = base64.b64decode(computed_b64)
                return hashlib.compare_digest(computed, expected)
            except Exception as e:
                print(f"PIN verify error: {e}")
                return False
        return False
        
    def initialize_routes(self):
        """Initialize empty routes list"""
        self.routes = []
        print("Initialized empty routes list")
    
    def create_html_content(self):
        """Create HTML content by loading from external index.html file"""

        # Read the index.html file
        html_path = os.path.join(os.path.dirname(__file__), "index.html")
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_template = f.read()
        except FileNotFoundError:
            html_template = """
            <!DOCTYPE html>
            <html><head><title>Error</title></head>
            <body><h1>Error: index.html not found</h1></body>
            </html>
            """

        # Read the fluid.js file
        fluid_js_path = os.path.join(os.path.dirname(__file__), "fluid.js")
        try:
            with open(fluid_js_path, 'r', encoding='utf-8') as f:
                fluid_js_content = f.read()
        except FileNotFoundError:
            fluid_js_content = "console.error('fluid.js not found');"
        
        # Inject the fluid.js content safely; placeholder lives in a JS comment to avoid static parsing errors
        html_content = html_template.replace('/* %%FLUID_JS%% */', fluid_js_content)

        # Ensure Qt WebChannel JS is available to the page
        # Insert <script src="qrc:///qtwebchannel/qwebchannel.js"></script> before </head>
        if '</head>' in html_content:
            html_content = html_content.replace(
                '</head>',
                '\n<script src="qrc:///qtwebchannel/qwebchannel.js"></script>\n</head>'
            )

        return html_content

    def initialize_map_system(self):
        """Initialize the OSM road system and pathfinder"""
        print("Initializing OSM road system...")
        try:
            self.road_system = OSMRoadSystem()
            if self.road_system:
                road_count = len(self.road_system.roads) if hasattr(self.road_system, 'roads') else 0
                print(f"Loaded {road_count} road segments")
            
            self.pathfinder = DijkstraPathfinder(self.road_system)
            if self.pathfinder:
                node_count = len(self.pathfinder.graph) if hasattr(self.pathfinder, 'graph') else 0
                print(f"Built graph with {node_count} nodes")
            
            print("Map system initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing map system: {str(e)}")
            return False
    
    def find_points_route(self, start_x, start_y, end_x, end_y):
        """Find multiple routes between two user-selected points"""
        if self.pathfinder is None or self.road_system is None:
            print("Pathfinder or road system not initialized")
            return []
            
        print(f"Finding routes between points ({start_x}, {start_y}) and ({end_x}, {end_y})")
            
        # Snap the points to the nearest road
        self.start_point = self.road_system.snap_point_to_road_center(start_x, start_y)
        self.end_point = self.road_system.snap_point_to_road_center(end_x, end_y)
        
        print(f"Snapped to road points: {self.start_point} and {self.end_point}")
            
        # Check that a path exists between these points
        path = self.pathfinder.find_path(
            self.start_point[0], self.start_point[1],
            self.end_point[0], self.end_point[1]
        )
        
        if not path or len(path) < 3:
            print("No valid path found between selected points")
            return []
            
        print(f"Found path with {len(path)} waypoints")
        
        # Now generate three different route variants
        return self._generate_route_variants()
    
    def find_routes(self, start_location, end_location):
        """Find multiple routes between two points with different characteristics"""
        if self.pathfinder is None:
            print("Pathfinder not initialized")
            return []
            
        # For simplicity, we're using a preset start location
        if start_location.lower() == "current location":
            # Use a fixed starting point near the center
            start_x, start_y, _ = self.road_system.get_random_spawn_point()
            self.start_point = (start_x, start_y)
        else:
            # Pick a random point
            node_id, pos = self.pathfinder.get_random_destination()
            if pos:
                self.start_point = pos
            else:
                self.start_point = (0, 0)
        
        # Generate end point (destination)
        # In a real app, this would be geocoded
        # Using a point that ensures we get a good route
        max_attempts = 20
        attempts = 0
        valid_path = False
        
        while attempts < max_attempts and not valid_path:
            node_id, pos = self.pathfinder.get_random_destination()
            if pos:
                # Make sure it's far enough away
                dist = math.sqrt((pos[0] - self.start_point[0])**2 + 
                                 (pos[1] - self.start_point[1])**2)
                if dist > 500:  # Ensure destination is at least 500 pixels away
                    self.end_point = pos
                    # Check that a path exists
                    path = self.pathfinder.find_path(
                        self.start_point[0], self.start_point[1],
                        self.end_point[0], self.end_point[1]
                    )
                    if len(path) > 5:  # Path has reasonable length
                        valid_path = True
            attempts += 1
        
        if not valid_path:
            print("Could not find valid destination")
            self.end_point = (500, 500)  # Fallback
        
        # Generate route variants using the helper method
        return self._generate_route_variants()
        
    def _generate_route_variants(self):
        """Generate multiple route variants between the current start and end points"""
        # Now generate three different routes
        self.routes = []
        
        # 1. Fastest route - Shortest path from start to end
        fastest_path = self.pathfinder.find_path(
            self.start_point[0], self.start_point[1],
            self.end_point[0], self.end_point[1]
        )
        fastest_waypoints = self.pathfinder.path_to_waypoints(fastest_path)
        fastest_distance = self.calculate_path_length(fastest_waypoints)
        fastest_time = self.estimate_travel_time(fastest_distance, 60)  # 60 km/h avg speed
        
        # 2. Scenic route - Add some randomness to the path
        scenic_waypoints = self.generate_scenic_route(fastest_waypoints)
        scenic_distance = self.calculate_path_length(scenic_waypoints)
        scenic_time = self.estimate_travel_time(scenic_distance, 50)  # 50 km/h avg speed
        
        # 3. Eco route - Find a middle ground
        eco_waypoints = self.generate_eco_route(fastest_waypoints, scenic_waypoints)
        eco_distance = self.calculate_path_length(eco_waypoints)
        eco_time = self.estimate_travel_time(eco_distance, 55)  # 55 km/h avg speed
        
        # Store route data
        self.routes = [
            {
                "type": "Fastest Route",
                "time": fastest_time,
                "distance": fastest_distance,
                "waypoints": fastest_waypoints,
                "description": "Optimal route",
                "conditions": "Current traffic"
            },
            {
                "type": "Scenic Route",
                "time": scenic_time,
                "distance": scenic_distance,
                "waypoints": scenic_waypoints,
                "description": "Alternative route",
                "conditions": "Different path"
            },
            {
                "type": "Eco Route",
                "time": eco_time,
                "distance": eco_distance,
                "waypoints": eco_waypoints,
                "description": "Best fuel efficiency",
                "conditions": "Fewer stops"
            }
        ]
        
        # Convert routes to JSON for JavaScript
        routes_json = []
        for i, route in enumerate(self.routes):
            routes_json.append({
                "index": i,
                "type": route["type"],
                "time": f"{route['time']} min",
                "distance": f"{route['distance']:.1f} km",
                "description": route["description"],
                "conditions": route["conditions"],
                "waypoints": [[p[0], p[1]] for p in route["waypoints"]],
                "start": [self.start_point[0], self.start_point[1]],
                "end": [self.end_point[0], self.end_point[1]]
            })
            
        return routes_json
        
    def calculate_path_length(self, waypoints):
        """Calculate the total length of a path in km"""
        if not waypoints or len(waypoints) < 2:
            return 0
            
        total_length = 0
        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            segment_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_length += segment_length
            
        # Convert pixels to km (approximate)
        return total_length * 0.01  # 1 pixel ≈ 10 meters
        
    def estimate_travel_time(self, distance_km, avg_speed_kmh):
        """Estimate travel time in minutes given distance and average speed"""
        if avg_speed_kmh <= 0:
            return 0
        hours = distance_km / avg_speed_kmh
        minutes = int(hours * 60)
        return max(1, minutes)  # Ensure at least 1 minute
        
    def generate_scenic_route(self, base_waypoints):
        """Generate a more scenic route by adding detours"""
        if not base_waypoints or len(base_waypoints) < 3:
            return base_waypoints
            
        # Create a copy of the base route
        scenic = base_waypoints.copy()
        
        # Add detours by offsetting some waypoints
        detour_count = min(3, len(scenic) // 3)
        for _ in range(detour_count):
            # Pick a point in the middle of the route (not start or end)
            index = random.randint(1, len(scenic) - 2)
            
            # Create a detour by adding a new point near the selected one
            orig_point = scenic[index]
            offset = 200  # pixels
            angle = random.uniform(0, 2 * math.pi)
            
            # Calculate new point with offset
            new_x = orig_point[0] + offset * math.cos(angle)
            new_y = orig_point[1] + offset * math.sin(angle)
            
            # Find nearest valid road point
            valid_point = self.road_system.snap_point_to_road_center(new_x, new_y)
            
            # Insert new point(s)
            scenic.insert(index + 1, valid_point)
            
        return scenic
        
    def generate_eco_route(self, fastest, scenic):
        """Generate eco-friendly route by combining fastest and scenic"""
        if len(fastest) < 3 or len(scenic) < 3:
            return fastest
            
        # Start with fastest route
        eco = fastest.copy()
        
        # Find a segment to optimize
        segment_idx = len(eco) // 3
        
        # Replace this segment with a more efficient path
        # In a real app, you'd consider elevation, traffic lights, etc.
        eco[segment_idx] = self.road_system.snap_point_to_road_center(
            (eco[segment_idx][0] + eco[segment_idx+1][0]) / 2,
            (eco[segment_idx][1] + eco[segment_idx+1][1]) / 2
        )
        
        return eco
    
    def on_load_finished(self, ok):
        """Called when the web page has finished loading"""
        if ok:
            print("Page loaded successfully")
            try:
                # Make backend object available to JavaScript
                self.page().setWebChannel(self.channel)
                
                # Print out available methods for debugging
                methods = [method for method in dir(self) if not method.startswith('_') and callable(getattr(self, method))]
                print(f"Available Python methods: {', '.join(methods)}")
                
                # Inject Python bridge for communication with proper WebChannel setup
                self.page().runJavaScript(
                    """
                    (function() {
                        function setupChannel() {
                            return new Promise(function(resolve) {
                                if (window.backend && window.backend.__isReady) {
                                    return resolve();
                                }
                                if (typeof QWebChannel === 'function' && window.qt && qt.webChannelTransport) {
                                    new QWebChannel(qt.webChannelTransport, function(channel) {
                                        window.backend = channel.objects.backend || {};
                                        window.backend.__isReady = true;
                                        resolve();
                                    });
                                } else {
                                    console.warn('QWebChannel not available yet; proceeding with stub backend');
                                    window.backend = window.backend || {};
                                    resolve();
                                }
                            });
                        }
                        setupChannel().then(function() {
                            try {
                                window.pyqtBridge = {
                                    clicked: function() {
                                        try { if (window.backend && typeof window.backend.clicked === 'function') { window.backend.clicked(); } else { console.error('clicked not available'); } } catch(err) { console.error('clicked error:', err); }
                                    },
                                    searchSubmitted: function(query) {
                                        try { if (window.backend && typeof window.backend.search_submitted === 'function') { window.backend.search_submitted(query); } else { console.error('search_submitted not available'); } } catch(err) { console.error('searchSubmitted error:', err); }
                                    },
                                    routeSelected: function(routeIndex) {
                                        try { if (window.backend && typeof window.backend.route_selected === 'function') { window.backend.route_selected(routeIndex); } else { console.error('route_selected not available'); } } catch(err) { console.error('routeSelected error:', err); }
                                    },
                                    setSelectedRoute: function(routeData) {
                                        try { if (window.backend && typeof window.backend.set_selected_route === 'function') { window.backend.set_selected_route(routeData); } else { console.error('set_selected_route not available'); } } catch(err) { console.error('setSelectedRoute error:', err); }
                                    },
                                    navigationStarted: function() {
                                        try { if (window.backend && typeof window.backend.navigation_started === 'function') { window.backend.navigation_started(); } else { console.error('navigation_started not available'); } } catch(err) { console.error('navigationStarted error:', err); }
                                    },
                                    getRoutes: function(startLocation, endLocation) {
                                        try { if (window.backend && typeof window.backend.find_routes === 'function') { return window.backend.find_routes(startLocation, endLocation); } else { console.error('find_routes not available'); return []; } } catch(err) { console.error('getRoutes error:', err); return []; }
                                    },
                                    findPointsRoute: function(startX, startY, endX, endY) {
                                        try { if (window.backend && typeof window.backend.find_points_route === 'function') { return window.backend.find_points_route(startX, startY, endX, endY); } else { console.error('find_points_route not available'); return []; } } catch(err) { console.error('findPointsRoute error:', err); return []; }
                                    }
                                };
                                document.dispatchEvent(new Event('pyqtBridgeReady'));
                                console.log('pyqtBridge ready');
                            } catch (err) {
                                console.error('Error setting up pyqtBridge:', err);
                            }
                        });
                    })();
                    """
                )
                
                print("WebChannel and JavaScript bridge setup complete")
                
                
            except Exception as e:
                print(f"Error in on_load_finished: {str(e)}")
        else:
            print("Page did not load successfully")

    # Called from JS via WebChannel when user presses Start
    @pyqtSlot()
    def navigation_started(self):
        try:
            print("Navigation start requested from UI")
            # Get reference to the main app and start navigation (switch to navigation view)
            app = QApplication.instance()
            if isinstance(app, DriverApp):
                app.start_navigation()
            else:
                print("Error: Could not find DriverApp instance")
        except Exception as e:
            print(f"navigation_started error: {e}")

    # Optional slots to reduce console errors from JS bridge
    @pyqtSlot()
    def clicked(self):
        try:
            self.ui_clicked.emit()
        except Exception:
            pass

    @pyqtSlot(str)
    def search_submitted(self, query: str):
        try:
            print(f"Search submitted: {query}")
            self.ui_search_submitted.emit(query)
        except Exception:
            pass

    @pyqtSlot(int)
    def route_selected(self, idx: int):
        try:
            print(f"Route selected method called with index: {idx}")
            print(f"Available routes: {len(self.routes) if hasattr(self, 'routes') and self.routes else 0}")
            if hasattr(self, 'routes') and self.routes:
                for i, route in enumerate(self.routes):
                    print(f"  Route {i}: {route.get('type', 'Unknown')} - {route.get('description', 'No description')}")
            self.ui_route_selected.emit(idx)
        except Exception as e:
            print(f"Error in route_selected: {e}")
            pass
            
    def set_selected_route(self, route_data):
        """Set the selected route data from JavaScript (accepts index or route object)."""
        try:
            print(f"JS called set_selected_route with: {route_data!r}")
            # If JS passed an index (number or numeric string), resolve it
            if isinstance(route_data, (int, float)) or (isinstance(route_data, str) and route_data.isdigit()):
                idx = int(route_data)
                if 0 <= idx < len(self.routes):
                    self.selected_route = self.routes[idx]
                    print(f"Selected route index {idx}: {self.selected_route.get('type','Unknown')}")
                else:
                    print(f"Route index out of range: {idx}")
                    self.selected_route = None
            elif isinstance(route_data, dict):
                # JS passed the full route object
                self.selected_route = route_data
                print(f"Selected route object stored: {self.selected_route.get('type','Unknown')}")
            else:
                print("set_selected_route received unsupported data type")
                self.selected_route = None

            # Notify internal listeners: emit ui_route_selected with index if known, otherwise -1
            if self.selected_route:
                try:
                    idx = next((i for i, r in enumerate(self.routes) if r.get('waypoints') == self.selected_route.get('waypoints')), None)
                    self.ui_route_selected.emit(int(idx) if idx is not None else -1)
                except Exception:
                    self.ui_route_selected.emit(-1)
            else:
                # Emit a negative index to indicate no valid selection
                self.ui_route_selected.emit(-1)
        except Exception as e:
            print(f"Error in set_selected_route: {e}")
            
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        # Do not auto-emit clicks to avoid unintended navigation starts
        super().mousePressEvent(event)


class DriverApp(QApplication):
    def __init__(self):
        super().__init__(sys.argv)
        
        # Create main window
        self.main_window = QWidget()
        self.main_window.setWindowTitle("Driver - Navigation System")
        self.main_window.setStyleSheet("background-color: black;")
        
        # Navigation state
        self.navigation_active = False
        self.selected_route = None
        self.main_py_process = None  # Process for visual3d_server.py when running
        
        # Create layout with stacked widget for navigation
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

    # Central stacked widget to switch between web UI and navigation view
        self.central_stack = QStackedWidget()

        # Page 0: Web UI (startup/route selection)
        self.fluid_view = FluidWebView()
        self.fluid_view.ui_route_selected.connect(self.on_route_selected)
        self.fluid_view.ui_search_submitted.connect(self.on_search_submitted)
        self.central_stack.addWidget(self.fluid_view)

    # Page 1: Navigation view with 3D web simulation
        self.navigation_view = self.create_navigation_view()
        self.central_stack.addWidget(self.navigation_view)

        layout.addWidget(self.central_stack)
        self.main_window.setLayout(layout)
            
        # Show fullscreen
        self.main_window.showFullScreen()
        
    def create_navigation_view(self):
        """Create the navigation view with 3D web simulation"""
        nav_container = QWidget()
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)
        
        # Web view for 3D simulation (main driving view)
        self.web_view = QWebEngineView()
        self.web_view.setObjectName("webView3D")
        nav_layout.addWidget(self.web_view, 1)  # Take most space
        
        nav_container.setLayout(nav_layout)
        return nav_container
    
    def prepare_main_py_background(self):
        """Prepare visual3d_server.py in background thread but don't start it yet"""
        def prepare_simulation():
            try:
                print("Preparing visual3d_server.py simulation in background...")
                # Verify visual3d_server.py exists and is accessible
                script_path = os.path.join(os.path.dirname(__file__), 'visual3d_server.py')
                if os.path.exists(script_path):
                    print(f"✅ visual3d_server.py found at: {script_path}")
                    self.main_py_ready = True
                else:
                    print(f"❌ visual3d_server.py not found at: {script_path}")
                    self.main_py_ready = False
            except Exception as e:
                print(f"Error preparing visual3d_server.py: {e}")
                self.main_py_ready = False
                
        # Run preparation in background thread
        self.main_py_ready = False
        prep_thread = threading.Thread(target=prepare_simulation, daemon=True)
        prep_thread.start()
    
    def on_start_clicked(self):
        """Handle the start action - launch 3D visualization in map area"""
        print("Starting 3D visualization in map area...")
        # Launch the 3D server and replace the map area
        self.launch_3d_in_map_area()
        
    def launch_3d_in_map_area(self):
        """Launch visual3d_server.py and inject into map area instead of switching views"""
        try:
            print("Launch 3D in map area started! Injecting into map area...")
            
            # Check if process is already running
            if hasattr(self, 'main_3d_process') and self.main_3d_process and self.main_3d_process.state() == QProcess.ProcessState.Running:
                print("3D visualization server already running, injecting into map area...")
                self.inject_3d_into_map_simple()
                return
            
            # Launch visual3d_server.py process
            script_path = os.path.join(os.path.dirname(__file__), 'visual3d_server.py')
            if not os.path.exists(script_path):
                print(f"❌ visual3d_server.py not found at: {script_path}")
                return
                
            self.main_3d_process = QProcess(self)
            # Ensure working directory is project folder so relative files (e.g., 3dsim.html, assets) resolve
            workdir = os.path.dirname(__file__)
            self.main_3d_process.setWorkingDirectory(workdir)
            print(f"3D process working directory set to: {workdir}")
            self.main_3d_process.finished.connect(self.on_3d_process_finished)
            # Pipe outputs for diagnostics (e.g., WebSocket bind errors)
            self.main_3d_process.readyReadStandardOutput.connect(lambda: self._log_proc_out(self.main_3d_process, prefix='[3D] '))
            self.main_3d_process.readyReadStandardError.connect(lambda: self._log_proc_err(self.main_3d_process, prefix='[3D ERR] '))
            self.main_3d_process.start(sys.executable, [script_path])
            
            print(f"✅ Launching visual3d_server.py from: {script_path}")
            
            # Wait a moment for server to start, then inject (retry a couple of times)
            def _try_inject(attempt=1):
                print(f"Attempting 3D iframe injection (attempt {attempt})...")
                try:
                    self.inject_3d_into_map_simple()
                finally:
                    if attempt < 3:
                        QTimer.singleShot(1500, lambda: _try_inject(attempt + 1))
            QTimer.singleShot(3000, _try_inject)
            
        except Exception as e:
            print(f"Error launching 3D visualization: {e}")
    
    def inject_3d_into_map_simple(self):
        """Simple iframe injection with better targeting"""
        try:
            print("Injecting 3D visualization with simple approach...")
            
            # Much simpler JavaScript that directly replaces content
            js_code = """
            (function() {
                console.log('Starting simple 3D injection...');
                
                // Remove any existing iframe first
                var existingIframe = document.getElementById('visualization3D');
                if (existingIframe) {
                    existingIframe.remove();
                }
                
                // Find ALL possible map containers
                var targets = [
                    document.querySelector('#mapView'),
                    document.querySelector('.map-container'),
                    document.querySelector('#leafletMap'),
                    document.querySelector('.leaflet-container')
                ];
                
                var targetContainer = null;
                for (var i = 0; i < targets.length; i++) {
                    if (targets[i] && targets[i].offsetWidth > 100 && targets[i].offsetHeight > 100) {
                        targetContainer = targets[i];
                        console.log('Found suitable container:', targetContainer.id || targetContainer.className);
                        break;
                    }
                }
                
                if (!targetContainer) {
                    console.error('No suitable container found');
                    return;
                }
                
                // Clear the container and add iframe
                targetContainer.innerHTML = '';
                targetContainer.style.position = 'relative';
                targetContainer.style.overflow = 'hidden';
                
                var iframe = document.createElement('iframe');
                iframe.src = 'http://localhost:5000';
                iframe.id = 'visualization3D';
                iframe.style.cssText = `
                    width: 100% !important;
                    height: 100% !important;
                    min-height: 500px !important;
                    border: none !important;
                    display: block !important;
                    position: absolute !important;
                    top: 0 !important;
                    left: 0 !important;
                    background: #f0f0f0 !important;
                `;
                
                targetContainer.appendChild(iframe);
                console.log('✅ Simple iframe injection completed');
                
                // Debug iframe after load
                iframe.onload = function() {
                    console.log('Iframe loaded. Size:', iframe.offsetWidth, 'x', iframe.offsetHeight);
                };
                
            })();
            """
            
            # Execute JavaScript
            if hasattr(self, 'fluid_view') and self.fluid_view:
                self.fluid_view.page().runJavaScript(js_code)
                print("✅ Simple 3D injection script executed")
            else:
                print("❌ Fluid view not available")
                
        except Exception as e:
            print(f"Error in simple injection: {e}")
            
    def on_3d_process_finished(self, exit_code):
        """Handle when 3D process finishes"""
        print(f"3D visualization process finished with code: {exit_code}")
        
    def on_search_submitted(self, query):
        """Handle search submission from the web UI"""
        print(f"Search submitted: {query}")
        # Find routes using the query as destination
        routes = self.fluid_view.find_routes("current location", query)
        print(f"Available routes: {len(routes)}")
        # Emit routes_found to update the UI
        self.fluid_view.routes_found.emit(routes)
        
    def on_route_selected(self, route_index):
        """Handle route selection from the web UI"""
        print(f"DriverApp.on_route_selected called with index: {route_index}")
        # selected_route is already set by set_selected_route
        if self.selected_route:
            print(f"Selected route: {self.selected_route.get('type', 'Unknown')} - {self.selected_route.get('description', 'No description')}")
            print(f"Route details: Time={self.selected_route.get('duration')}, Distance={self.selected_route.get('distance')}")
            # Start navigation immediately when route is selected
            self.start_navigation()
        else:
            print("No route selected yet")
            
    def start_navigation(self):
        """Start navigation with 3D web simulation"""
        print(f"start_navigation called. Selected route: {self.selected_route is not None}")
        print(f"navigation_active: {getattr(self, 'navigation_active', False)}")
        
        # Prevent multiple simultaneous navigation starts
        if getattr(self, 'navigation_active', False):
            print("Navigation already active, ignoring duplicate start request")
            return
            
        if not self.selected_route:
            print("ERROR: No route selected for navigation!")
            print("User must select a route from the options before starting navigation.")
            
            # For testing, let's automatically select the first route
            if hasattr(self.fluid_view, 'routes') and self.fluid_view.routes:
                print("Auto-selecting first route for testing...")
                self.selected_route = self.fluid_view.routes[0]
                print(f"Auto-selected route: {self.selected_route['type']}")
            else:
                print("No routes available to auto-select")
                return
            
        print("🚀 Starting 3D web navigation simulation...")
        self.navigation_active = True
        
        # Switch to navigation view (3D web simulation)
        self.central_stack.setCurrentIndex(1)
        
    # Start visual3d_server.py simulation
        self.start_3d_simulation()
        
    def launch_main_py_window(self):
        """Launch visual3d_server.py simulation in a separate window"""
        try:
            print("Launching visual3d_server.py in separate window...")
            
            # Stop any existing visual3d_server.py process
            if self.main_py_process and self.main_py_process.state() == QProcess.ProcessState.Running:
                print("Stopping existing visual3d_server.py process...")
                self.main_py_process.kill()
                self.main_py_process.waitForFinished(3000)
            
            # Create new process
            self.main_py_process = QProcess(self)
            
            # Connect signals for feedback
            self.main_py_process.started.connect(lambda: print("✅ visual3d_server.py started successfully!"))
            self.main_py_process.finished.connect(self.on_main_py_finished)
            self.main_py_process.errorOccurred.connect(lambda error: print(f"visual3d_server.py error: {error}"))
            
            # Prepare command
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), 'visual3d_server.py')
            
            # Run visual3d_server.py with auto mode (destination driving)
            args = [script_path, '--auto', '--mode', '1']  # Mode 1 = Destination mode
            
            print(f"Starting: {python_exe} {' '.join(args)}")
            
            # Start the process
            self.main_py_process.start(python_exe, args)
            
            if not self.main_py_process.waitForStarted(5000):
                print("❌ Failed to start visual3d_server.py")
            else:
                print("🎮 visual3d_server.py simulation launched in separate window!")
                print("ℹ️  The main GUI will stay open for route selection and controls")
                
        except Exception as e:
            print(f"Error launching visual3d_server.py window: {e}")
            
    def on_3d_started_for_map(self):
        """Called when 3D visualization starts - inject into map area"""
        print("✅ 3D visualization server started! Injecting into map area...")
        # Give the server a moment to start up, then inject the iframe
        QTimer.singleShot(2000, self.inject_3d_into_map_simple)
    
    def inject_3d_into_map(self):
        """Inject 3D visualization iframe into the map area of the fluid view"""
        try:
            print("Injecting 3D visualization into map area...")
            
            # JavaScript to replace the map area with an iframe
            js_code = """
            (function() {
                console.log('Injecting 3D visualization iframe into map area...');
                
                # ...existing code...
            })();
            """            # Execute the JavaScript in the web view
            if hasattr(self, 'fluid_view') and self.fluid_view:
                self.fluid_view.page().runJavaScript(js_code, lambda result: print("JavaScript injection completed"))
                print("✅ 3D visualization injected into map area")
            else:
                print("❌ Fluid view not found")
                
        except Exception as e:
            print(f"Error injecting 3D visualization: {e}")

            
    def start_3d_simulation(self):
        """Launch visual3d_server.py simulation and load the web interface"""
        try:
            print("Launching visual3d_server.py 3D simulation...")
            
            # Stop any existing visual3d_server.py process
            if hasattr(self, 'main_3d_process') and self.main_3d_process and self.main_3d_process.state() == QProcess.ProcessState.Running:
                print("Stopping existing visual3d_server.py process...")
                self.main_3d_process.kill()
                self.main_3d_process.waitForFinished(3000)
            
            # Create new process for visual3d_server.py
            self.main_3d_process = QProcess(self)
            # Ensure working directory is project folder for relative resources
            workdir = os.path.dirname(__file__)
            self.main_3d_process.setWorkingDirectory(workdir)
            print(f"3D process working directory set to: {workdir}")
            
            # Connect signals for feedback
            self.main_3d_process.started.connect(self.on_3d_simulation_started)
            self.main_3d_process.finished.connect(self.on_3d_simulation_finished)
            self.main_3d_process.errorOccurred.connect(lambda error: print(f"visual3d_server.py error: {error}"))
            self.main_3d_process.readyReadStandardOutput.connect(lambda: self._log_proc_out(self.main_3d_process, prefix='[3D] '))
            self.main_3d_process.readyReadStandardError.connect(lambda: self._log_proc_err(self.main_3d_process, prefix='[3D ERR] '))
            
            # Prepare command
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), 'visual3d_server.py')
            
            print(f"Starting: {python_exe} {script_path}")
            
            # Start the process
            self.main_3d_process.start(python_exe, [script_path])
            
            if not self.main_3d_process.waitForStarted(5000):
                raise Exception("visual3d_server.py failed to start within 5 seconds")
            
        except Exception as e:
            print(f"Error launching visual3d_server.py: {e}")

    def _log_proc_out(self, proc: QProcess, prefix: str = ''):
        try:
            data = bytes(proc.readAllStandardOutput()).decode(errors='ignore')
            for line in data.splitlines():
                if line.strip():
                    print(prefix + line)
        except Exception:
            pass

    def _log_proc_err(self, proc: QProcess, prefix: str = ''):
        try:
            data = bytes(proc.readAllStandardError()).decode(errors='ignore')
            for line in data.splitlines():
                if line.strip():
                    print(prefix + line)
        except Exception:
            pass
    
    def on_3d_simulation_started(self):
        """Called when visual3d_server.py starts successfully"""
        print("✅ visual3d_server.py started successfully!")
        # Give the server a moment to start up, then load the webpage
        QTimer.singleShot(2000, self.load_3d_webpage)
    
    def load_3d_webpage(self):
        """Load the 3D simulation webpage in the navigation view"""
        print("Loading 3D simulation webpage...")
        if hasattr(self, 'web_view'):
            self.web_view.load(QUrl("http://localhost:5000"))
    
    def on_3d_simulation_finished(self, exit_code, exit_status):
        """Called when visual3d_server.py simulation finishes"""
        print(f"visual3d_server.py simulation finished: code={exit_code}, status={exit_status}")
        self.navigation_active = False
        if exit_code != 0:
            print("⚠️  visual3d_server.py simulation ended with error code")
        else:
            print("✅ visual3d_server.py simulation completed successfully")
            
    def on_main_py_finished(self, exit_code, exit_status):
        """Called when visual3d_server.py simulation finishes"""
        print(f"visual3d_server.py simulation finished: code={exit_code}, status={exit_status}")
        self.navigation_active = False
        if exit_code != 0:
            print("⚠️  visual3d_server.py simulation ended with error code")
        else:
            print("✅ visual3d_server.py simulation completed successfully")
    
    def generate_turn_segments(self, waypoints):
        """Generate turn-by-turn segments from waypoints"""
        if not waypoints or len(waypoints) < 2:
            return [
                {"direction": "↑", "instruction": "Start your journey", "distance": "0.1 km", "street": "from current location"}
            ]
        
        segments = []
        total_segments = min(6, len(waypoints) - 1)  # Limit to 6 segments for display
        
        directions = ["↑", "→", "↑", "←", "↗", "↖"]
        instructions = [
            "Start your journey",
            "Turn right", 
            "Continue straight",
            "Turn left",
            "Bear right",
            "Turn slight left"
        ]
        streets = [
            "from current location",
            "onto Main St",
            "on Main St", 
            "toward destination",
            "onto Highway",
            "to destination"
        ]
        
        for i in range(total_segments):
            if i < len(waypoints) - 1:
                # Calculate distance to next waypoint
                current_wp = waypoints[i]
                next_wp = waypoints[i + 1]
                distance = math.sqrt((next_wp[0] - current_wp[0])**2 + (next_wp[1] - current_wp[1])**2)
                distance_km = distance * 0.01  # Convert to km (approximate)
                
                segment = {
                    "direction": directions[i % len(directions)],
                    "instruction": instructions[i % len(instructions)],
                    "distance": f"{distance_km:.1f} km",
                    "street": streets[i % len(streets)]
                }
                segments.append(segment)
        
        return segments
        
    def on_navigation_cancelled(self):
        """Handle navigation cancellation"""
        print("Navigation cancelled")
        self.navigation_active = False
        self.selected_route = None
        
        # Stop 3D simulation if running
        if hasattr(self, 'main_3d_process') and self.main_3d_process and self.main_3d_process.state() == QProcess.ProcessState.Running:
            try:
                self.main_3d_process.kill()
                self.main_3d_process.waitForFinished(3000)
                print("3D simulation stopped")
            except Exception as e:
                print(f"Error stopping 3D simulation: {e}")
        
        # Return to route selection (web UI)
        self.central_stack.setCurrentIndex(0)
        
    def cleanup_processes(self):
        """Clean up any running processes"""
        # Clean up 3D simulation
        if hasattr(self, 'main_3d_process') and self.main_3d_process and self.main_3d_process.state() == QProcess.ProcessState.Running:
            try:
                self.main_3d_process.kill()
                self.main_3d_process.waitForFinished(3000)
            except Exception as e:
                print(f"Error cleaning up 3D simulation: {e}")
        
        # Clean up any other visual3d_server.py processes
        if self.main_py_process and self.main_py_process.state() == QProcess.ProcessState.Running:
            try:
                print("Cleaning up visual3d_server.py process...")
                self.main_py_process.kill()
                self.main_py_process.waitForFinished(3000)
            except Exception as e:
                print(f"Error cleaning up visual3d_server.py process: {e}")
        
    def run(self):
        """Run the application"""
        try:
            return self.exec()
        finally:
            # Clean up processes when app exits
            self.cleanup_processes()


if __name__ == "__main__":
    app = DriverApp()
    sys.exit(app.run())
"""
3D Visualization of Car AI Simulation using Three.js
This program creates a 3D environment that visualizes the car simulation data
"""

import json
import math
import threading
import time
from typing import List
import websockets
from flask import Flask, render_template_string
import random
from constants import *
from car import Car
from osm_roads import OSMRoadSystem
from dijkstra_pathfinding import DijkstraPathfinder

app = Flask(__name__)

# Global variables to store simulation state
simulation_core = None
simulation_running = False
websocket_clients = set()
emergency_stop_active = False  # Global emergency stop state

class HeadlessSimulationCore:
    """Headless version of simulation core that runs without pygame display"""
    
    def __init__(self, width=1280, height=720, mode=0, auto_mode=False, initial_scale=1.0):
        """Initialize headless simulation without pygame display"""
        self.width = width
        self.height = height
        self.mode = mode
        self.auto_mode = auto_mode
        
        # Initialize game systems without pygame
        self.road_system = OSMRoadSystem()
        
        # Create a simple camera class for headless mode
        class HeadlessCamera:
            def __init__(self, x=0, y=0, width=1280, height=720):
                self.x = x
                self.y = y
                self.width = width
                self.height = height
                self.zoom = 1.0
                self.manual_mode = False
            
            def follow_target(self, x, y):
                self.x = x
                self.y = y
            
            def update(self, keys=None):
                pass
        
        self.camera = HeadlessCamera(0, 0, width, height)
        self.camera.zoom = initial_scale
        self.pathfinder = DijkstraPathfinder(self.road_system)
        
        # Simulation state
        self.running = True
        self.cars = []
        self.generation = 0
        self.evolution_timer = 0
        self.evolution_interval = 40 * 60  # 40 seconds at 60 FPS
        
        # Path validation state
        self.paths_validated = False
        self.valid_paths_count = 0
        self.required_paths_count = 0
        
        # Performance tracking
        from collections import deque
        self.fitness_history = deque(maxlen=100)
        self.generation_history = deque(maxlen=100)
        self.saved_cars_history = deque(maxlen=100)
        
        # Mode-specific state
        self.destination_mode_start = None
        self.destination_mode_end = None
        self.destination_mode_car = None
        self.destination_mode_state = "waiting"
        
        self.initialize_mode()
    
    def initialize_mode(self):
        """Initialize the simulation based on the selected mode (headless version)"""
        self.road_system.load_roads()
        print(f"Road system loaded with {len(self.road_system.road_segments)} segments")
        
        if self.mode == 0:  # Training mode
            self.initialize_training_mode()
        elif self.mode == 1:  # Destination mode
            self.initialize_destination_mode()
            
        # Validate paths after initialization
        if not self.validate_all_paths():
            print("WARNING: Not all cars have valid paths. Simulation will not start until paths are fixed.")
    
    def initialize_training_mode(self):
        """Initialize training mode with population"""
        from evolution import load_population, create_car_from_data, assign_shared_destination
        import random
        
        # Load existing population or create new one
        population_data, loaded_generation = load_population()
        
        if population_data:
            print(f"Loading existing population from generation {loaded_generation}")
            self.generation = loaded_generation + 1
            self.cars = []
            
            # Recreate cars from saved data
            for car_data in population_data['cars']:
                if len(self.cars) >= POPULATION_SIZE:
                    break
                car = create_car_from_data(car_data, self.road_system, self.pathfinder)
                self.cars.append(car)
            
            # Fill remaining slots if needed
            while len(self.cars) < POPULATION_SIZE:
                spawn_x, spawn_y, spawn_angle = self.road_system.get_random_spawn_point()
                car = Car(spawn_x, spawn_y, random.choice([RED, GREEN, BLUE]), 
                         use_ai=True, road_system=self.road_system, pathfinder=self.pathfinder)
                car.angle = spawn_angle
                car.generate_individual_destination()
                if car.path_waypoints:
                    car.orient_to_first_waypoint()
                self.cars.append(car)
        else:
            print("Creating new population for training...")
            self.cars = []
            for _ in range(POPULATION_SIZE):
                spawn_x, spawn_y, spawn_angle = self.road_system.get_random_spawn_point()
                car = Car(spawn_x, spawn_y, random.choice([RED, GREEN, BLUE]), 
                         use_ai=True, road_system=self.road_system, pathfinder=self.pathfinder)
                car.angle = spawn_angle
                car.generate_individual_destination()
                if car.path_waypoints:
                    car.orient_to_first_waypoint()
                self.cars.append(car)
        
        # Assign a shared destination for fair training
        assign_shared_destination(self.cars, self.pathfinder)
        print(f"Training mode: Created population of {len(self.cars)} cars")
        
        # Set camera to follow center of action
        if self.cars:
            center_x = sum(car.x for car in self.cars) / len(self.cars)
            center_y = sum(car.y for car in self.cars) / len(self.cars)
            self.camera.follow_target(center_x, center_y)
    
    def initialize_destination_mode(self):
        """Initialize destination mode with enhanced path validation"""
        print("Initializing destination mode with path validation...")
        
        max_attempts = 5
        cars_created = 0
        target_cars = 1  # For destination mode, we usually want 1 car
        
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts} to create cars with valid paths...")
            
            # Try to create a car with a valid route
            car = self.create_valid_route_car(max_attempts=10)
            if car and hasattr(car, 'path_waypoints') and car.path_waypoints and len(car.path_waypoints) > 1:
                self.cars = [car]
                self.destination_mode_car = car
                self.destination_mode_state = "driving"
                self.camera.follow_target(car.x, car.y)
                cars_created = 1
                print(f"Created car with {len(car.path_waypoints)} waypoints")
                break
            else:
                print(f"Failed to create car with valid route on attempt {attempt + 1}")
        
        if cars_created == 0:
            print("CRITICAL: Failed to create any cars with valid paths after all attempts")
            print("Creating fallback car with simplified path to allow simulation to start")
            from car import Car
            
            # Create a fallback car with a simple straight-line path
            fallback_car = Car(640, 360, 0)
            
            # Create a simple fallback path (straight line)
            start_x, start_y = 640, 360
            end_x, end_y = start_x + 200, start_y  # Simple straight path
            
            fallback_car.path_waypoints = [
                (start_x, start_y),
                (start_x + 50, start_y),
                (start_x + 100, start_y),
                (start_x + 150, start_y),
                (end_x, end_y)
            ]
            fallback_car.destination = (end_x, end_y)
            
            self.cars = [fallback_car]
            self.destination_mode_car = fallback_car
            self.destination_mode_state = "driving"
            cars_created = 1
            print(f"Created fallback car with simple {len(fallback_car.path_waypoints)} waypoint path")
            
        print(f"Destination mode initialized with {len(self.cars)} car(s)")
    
    def create_valid_route_car(self, max_attempts=10):
        """Create a car with a valid route using enhanced pathfinding validation"""
        print(f"Attempting to create car with valid route (max {max_attempts} attempts)...")
        
        for attempt in range(max_attempts):
            try:
                # Get available roads
                roads = list(self.road_system.roads.values()) if hasattr(self.road_system, 'roads') else []
                
                # If no roads in self.road_system.roads, try road_segments
                if len(roads) < 2:
                    roads = self.road_system.road_segments if hasattr(self.road_system, 'road_segments') else []
                
                if len(roads) < 2:
                    print(f"ERROR: Not enough roads for pathfinding (found {len(roads)})")
                    print("Checking if road system was properly initialized...")
                    
                    # Force road system reload
                    try:
                        self.road_system.load_roads()
                        roads = self.road_system.road_segments if hasattr(self.road_system, 'road_segments') else []
                        print(f"After reload: found {len(roads)} road segments")
                    except Exception as e:
                        print(f"Road reload failed: {e}")
                    
                    if len(roads) < 2:
                        print("Still no roads available, will use fallback after all attempts")
                        continue
                
                # Select random start and end roads with minimum distance
                attempts_for_distance = 20
                start_road, end_road = None, None
                
                for dist_attempt in range(attempts_for_distance):
                    potential_start = random.choice(roads)
                    potential_end = random.choice([r for r in roads if r != potential_start])
                    
                    # Calculate distance between roads
                    distance = math.hypot(
                        potential_end.start[0] - potential_start.start[0],
                        potential_end.start[1] - potential_start.start[1]
                    )
                    
                    # Ensure minimum distance for meaningful path
                    if distance > 200:  # Minimum distance for valid route
                        start_road, end_road = potential_start, potential_end
                        break
                
                if not start_road or not end_road:
                    print(f"Attempt {attempt + 1}: Could not find roads with sufficient distance")
                    continue
                
                # Get start and end positions
                start_pos = start_road.start
                end_pos = end_road.end
                
                print(f"Attempt {attempt + 1}: Finding path from {start_pos} to {end_pos}")
                
                # Find path using pathfinder
                path_nodes = self.pathfinder.find_path(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
                
                if path_nodes and len(path_nodes) > 1:
                    # Convert node IDs to waypoint coordinates
                    path = self.pathfinder.path_to_waypoints(path_nodes)
                    
                    if path and len(path) > 1:
                        # Validate path quality
                        path_length = self._calculate_path_length(path)
                        direct_distance = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
                        
                        # Path should not be too much longer than direct distance (reasonable routing)
                        if path_length > 0 and path_length < direct_distance * 3:  # Allow up to 3x direct distance
                            # Create car at start position
                            car = self.create_destination_mode_car(start_pos, end_pos)
                            if car:
                                # Keep the path prepared inside create_destination_mode_car
                                # (already offset/resampled/pruned and oriented). Do not overwrite it.
                                car.destination = end_pos  # Store destination
                            print(f"Created car with valid path: {len(path)} waypoints, {path_length:.1f} units long")
                            return car
                        else:
                            print(f"Attempt {attempt + 1}: Failed to create car object")
                    else:
                        print(f"Attempt {attempt + 1}: Path too long or invalid ({path_length:.1f} vs {direct_distance:.1f})")
                else:
                    print(f"Attempt {attempt + 1}: No path found or path too short")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with exception: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"FAILED: Could not create valid route car after {max_attempts} attempts")
        return None
    
    def _calculate_path_length(self, path):
        """Calculate total length of a path"""
        if not path or len(path) < 2:
            return 0
            
        total_length = 0
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            total_length += math.hypot(next_point[0] - current[0], next_point[1] - current[1])
            
        return total_length
    
    def create_destination_mode_car(self, start_pos, end_pos):
        """Create a car for destination mode with pathfinding"""
        from evolution import create_destination_mode_car
        return create_destination_mode_car(start_pos, end_pos, self.road_system, self.pathfinder)
    

    

    
    def validate_all_paths(self):
        """Validate that all cars have valid paths before allowing simulation to start"""
        if not self.cars:
            print("No cars to validate paths for")
            return False
            
        valid_count = 0
        total_count = len(self.cars)
        
        for car in self.cars:
            if hasattr(car, 'path_waypoints') and car.path_waypoints and len(car.path_waypoints) > 1:
                # Additional validation: check if path actually connects start to destination
                if self._validate_path_connectivity(car):
                    valid_count += 1
                    print(f"Car {id(car)} has valid path with {len(car.path_waypoints)} waypoints")
                else:
                    print(f"Car {id(car)} has invalid or disconnected path")
            else:
                print(f"Car {id(car)} has no valid path waypoints")
        
        self.valid_paths_count = valid_count
        self.required_paths_count = total_count
        self.paths_validated = (valid_count == total_count and valid_count > 0)
        
        print(f"Path validation: {valid_count}/{total_count} cars have valid paths")
        if self.paths_validated:
            print("All cars have valid paths - simulation can start")
        else:
            print("Not all cars have valid paths - simulation blocked")
            
        return self.paths_validated
    
    def _validate_path_connectivity(self, car):
        """Check if a car's path is actually connected and reachable"""
        if not hasattr(car, 'path_waypoints') or not car.path_waypoints:
            return False
            
        path = car.path_waypoints
        if len(path) < 2:
            return False
            
        # Check if waypoints are reasonably connected (no huge gaps)
        max_gap = 200  # Increased maximum allowed distance between consecutive waypoints
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            distance = math.hypot(next_point[0] - current[0], next_point[1] - current[1])
            if distance > max_gap:
                print(f"Path gap too large: {distance} units between waypoints {i} and {i+1}")
                return False
                
        # Check if start point is close to car's actual position (more lenient)
        start_point = path[0]
        car_distance_to_start = math.hypot(start_point[0] - car.x, start_point[1] - car.y)
        if car_distance_to_start > 100:  # Increased tolerance - car should be reasonably near start of path
            print(f"Car too far from path start: {car_distance_to_start} units (allowing up to 100)")
            return False
            
        return True
    
    def update_simulation(self):
        """Update one frame of the simulation (headless)"""
        if not self.running:
            return
            
        # Check if paths are validated before allowing updates
        if not getattr(self, 'paths_validated', False):
            # Try to re-validate paths (but with limits to prevent infinite loops)
            if not hasattr(self, '_validation_attempts'):
                self._validation_attempts = 0
                
            if self._validation_attempts < 3:  # Maximum 3 validation attempts
                self._validation_attempts += 1
                print(f"Validation attempt {self._validation_attempts}/3...")
                if not self.validate_all_paths():
                    if self._validation_attempts >= 3:
                        print("FORCING SIMULATION START: Path validation failed after 3 attempts")
                        print("Setting paths_validated=True to prevent infinite blocking")
                        self.paths_validated = True  # Force validation to prevent infinite loop
                        self.valid_paths_count = len(self.cars)
                        self.required_paths_count = len(self.cars)
                    else:
                        print("Simulation blocked: No valid paths found. Retrying...")
                        return  # Block simulation updates for now
                else:
                    print("Path validation successful")
            else:
                # Already tried 3 times, force simulation to start
                print("Path validation bypassed after maximum attempts")
                self.paths_validated = True
            
        if self.mode == 0:  # Training mode
            self.update_training_mode()
        elif self.mode == 1:  # Destination mode
            self.update_destination_mode()
    
    def update_training_mode(self):
        """Update training mode logic"""
        # Find best car and update cars
        alive_cars = 0
        saved_cars = 0
        best_car = None
        
        for car in self.cars:
            if not car.crashed or car.saved_state:
                car.calculate_fitness()
                if best_car is None or car.fitness > best_car.fitness:
                    best_car = car
                alive_cars += 1
                if car.saved_state:
                    saved_cars += 1
        
        # Update cars
        active_best_car = None
        for car in self.cars:
            if not car.crashed or car.saved_state:
                if not car.saved_state:
                    car.calculate_predictive_path()
                    car.update_path_following_accuracy()
                    car.move({})  # No keyboard input in headless mode
                    
                    if active_best_car is None or car.fitness > active_best_car.fitness:
                        if not car.crashed and not car.saved_state:
                            active_best_car = car
                
                car.update_raycasts()
        
        # Update camera to follow best car
        if active_best_car:
            self.camera.follow_target(active_best_car.x, active_best_car.y)
        
        # Evolution logic
        self.evolution_timer += 1
        unsaved_active_exists = any((not c.crashed) and (not c.saved_state) for c in self.cars)
        saved_active_exists = any((not c.crashed) and c.saved_state for c in self.cars)
        only_saved_remaining = (not unsaved_active_exists) and saved_active_exists
        
        if self.evolution_timer >= self.evolution_interval or alive_cars == 0 or only_saved_remaining:
            from evolution import evolve_population
            self.cars = evolve_population(self.cars, len(self.cars), self.road_system, self.pathfinder, self.generation)
            self.evolution_timer = 0
            self.generation += 1
            print(f"Generation {self.generation} started with {len(self.cars)} cars")
            
            # Update statistics
            if best_car:
                self.fitness_history.append(best_car.fitness)
                self.generation_history.append(self.generation)
                self.saved_cars_history.append(saved_cars)
    
    def update_destination_mode(self):
        """Update destination mode logic with path validation"""
        # Double-check path validation before allowing car movement
        if not getattr(self, 'paths_validated', False):
            print("Destination mode blocked: Cars do not have valid paths")
            return
            
        if self.destination_mode_car and not self.destination_mode_car.crashed:
            car = self.destination_mode_car
            
            # Verify car still has valid path before updating
            if not hasattr(car, 'path_waypoints') or not car.path_waypoints or len(car.path_waypoints) < 2:
                print(f"Car {id(car)} lost its path! Attempting to regenerate...")
                new_car = self.create_valid_route_car(max_attempts=3)
                if new_car:
                    # Transfer some state but use new path
                    new_car.x = car.x
                    new_car.y = car.y
                    new_car.angle = car.angle
                    self.destination_mode_car = new_car
                    self.cars = [new_car]
                    # Re-validate paths
                    self.validate_all_paths()
                    print("Route regenerated and validated successfully")
                else:
                    print("Failed to regenerate path. Stopping car.")
                    car.crashed = True
                return
            
            car.calculate_predictive_path()
            car.update_path_following_accuracy()
            car.move({})
            car.update_raycasts()
            self.camera.follow_target(car.x, car.y)
            
            # Check if car reached destination
            if hasattr(car, 'current_waypoint_index') and car.path_waypoints:
                if car.current_waypoint_index >= len(car.path_waypoints) - 1:
                    print(f"Car reached destination! Final fitness: {car.fitness}")
                    # Could create new route or stop here
        
        # Handle case where car crashed or lost
        active_cars = [car for car in self.cars if not car.crashed]
        if len(active_cars) == 0 and len(self.cars) > 0:
            print("All cars crashed or lost paths. Attempting to create new car with valid path...")
            new_car = self.create_valid_route_car(max_attempts=5)
            if new_car:
                self.cars = [new_car]
                self.destination_mode_car = new_car
                self.validate_all_paths()  # Re-validate after creating new car
                print("New car created and paths validated")
            else:
                print("Failed to create replacement car with valid path")
        else:
            self.cars = active_cars

class SimulationData:
    """Class to manage and format simulation data for 3D visualization"""
    
    def __init__(self):
        self.cars_data = []
        self.roads_data = []
        self.camera_data = {}
    
    def update_cars(self, cars: List):
        """Convert car objects to 3D visualization data"""
        self.cars_data = []
        for car in cars:
            if car.crashed and not car.saved_state:
                continue
                
            # Generate upcoming segments with turn instructions
            upcoming_segments = self._generate_upcoming_segments(car, simulation_core.road_system if simulation_core else None)
            
            car_data = {
                'id': id(car),
                'position': {
                    'x': car.x / 20,  # Scale down more for better 3D viewing
                    'y': 0.2,  # Lower height above ground
                    'z': -car.y / 20  # Flip Y to Z for 3D (Y is up in 3D)
                },
                'rotation': {
                    'x': 0,
                    'y': math.radians(car.angle),  # Convert to radians for Three.js
                    'z': 0
                },
                'color': self._pygame_color_to_hex(car.color),
                'speed': car.speed,
                'fitness': car.fitness,
                'crashed': car.crashed,
                'saved_state': car.saved_state,
                'raycast_distances': car.raycast_distances[:],
                'raycast_angles': car.raycast_angles[:],
                'path_waypoints': [(x/20, -y/20) for x, y in car.path_waypoints] if car.path_waypoints else [],
                'current_waypoint_index': getattr(car, 'current_waypoint_index', 0),  # Add waypoint progress tracking
                'upcoming_segments': upcoming_segments  # Add real navigation data
            }
            self.cars_data.append(car_data)
    
    def update_roads(self, road_system):
        """Convert road system to 3D visualization data"""
        if not road_system or not hasattr(road_system, 'road_segments'):
            return
            
        self.roads_data = []
        for segment in road_system.road_segments:
            road_data = {
                'id': id(segment),
                'start': {
                    'x': segment.start[0] / 20,
                    'y': 0,
                    'z': -segment.start[1] / 20
                },
                'end': {
                    'x': segment.end[0] / 20,
                    'y': 0,
                    'z': -segment.end[1] / 20
                },
                'width': segment.width / 20,
                'lane_count': segment.lane_count,
                'oneway': segment.oneway
            }
            self.roads_data.append(road_data)
    
    def update_camera(self, camera):
        """Update camera data for 3D visualization"""
        if camera:
            self.camera_data = {
                'x': camera.x / 20,
                'y': 25,  # Lower height for 3D camera
                'z': -camera.y / 20,
                'zoom': camera.zoom
            }
    
    def _pygame_color_to_hex(self, pygame_color) -> str:
        """Convert pygame color tuple to hex string"""
        if isinstance(pygame_color, tuple) and len(pygame_color) >= 3:
            return f"#{pygame_color[0]:02x}{pygame_color[1]:02x}{pygame_color[2]:02x}"
        return "#ffffff"  # Default to white
    
    def _format_turn_instruction(self, ang_diff_deg: float) -> str:
        """Format turn instruction based on angle difference"""
        a = abs(ang_diff_deg)
        if a < 10: return "Continue straight"
        # Sign convention appears inverted; swap left/right
        if a < 25: return ("Slight right" if ang_diff_deg > 0 else "Slight left")
        if a < 55: return ("Turn right" if ang_diff_deg > 0 else "Turn left")
        if a < 120: return ("Sharp right" if ang_diff_deg > 0 else "Sharp left")
        return "U-turn right" if ang_diff_deg > 0 else "U-turn left"
    
    def _collect_upcoming_segments(self, car, road_system, max_items=7):
        """Collect upcoming road segments with street names for navigation"""
        names = []
        if not getattr(car, 'path_waypoints', None) or len(car.path_waypoints) < 2:
            return names
        start_idx = max(0, getattr(car, 'current_waypoint_index', 0))
        last_name = None
        dist_accum = 0.0
        for i in range(start_idx, len(car.path_waypoints)-1):
            x1, y1 = car.path_waypoints[i]
            x2, y2 = car.path_waypoints[i+1]
            seg_len = math.hypot(x2-x1, y2-y1)
            if seg_len <= 1e-6:
                continue
            midx = (x1 + x2) * 0.5
            midy = (y1 + y2) * 0.5
            seg_name = None
            try:
                candidates = []
                if hasattr(road_system, 'spatial_grid'):
                    candidates = road_system.spatial_grid.get_nearby_segments(midx, midy, radius=60)
                if not candidates and hasattr(road_system, 'road_segments'):
                    candidates = road_system.road_segments[:200]
                best_seg = None
                best_d2 = 1e12
                for seg in candidates:
                    sx, sy = seg.start; ex, ey = seg.end
                    dx = ex - sx; dy = ey - sy
                    L2 = dx*dx + dy*dy
                    if L2 <= 1e-6:
                        continue
                    t = ((midx - sx)*dx + (midy - sy)*dy)/L2
                    t = 0 if t < 0 else (1 if t > 1 else t)
                    px = sx + dx*t; py = sy + dy*t
                    d2 = (px - midx)**2 + (py - midy)**2
                    if d2 < best_d2:
                        best_d2 = d2
                        best_seg = seg
                if best_seg is not None:
                    seg_name = getattr(best_seg, 'name', None) or getattr(best_seg, 'road_name', None)
            except Exception:
                seg_name = None
            if not seg_name:
                seg_name = f"Segment {i:03d}"
            dist_accum += seg_len
            if seg_name != last_name:
                names.append((seg_name, dist_accum))
                last_name = seg_name
            if len(names) >= max_items:
                break
        return names
    
    def _generate_upcoming_segments(self, car, road_system):
        """Generate upcoming navigation segments with turn instructions based on car's current progress"""
        segments = []
        if not getattr(car, 'path_waypoints', None) or len(car.path_waypoints) < 3:
            return segments
        
        # Get car's current position and waypoint progress
        current_wp_idx = max(0, getattr(car, 'current_waypoint_index', 0))
        car_x, car_y = car.x, car.y
        
        # Get upcoming road segments based on current progress
        upcoming_roads = self._collect_upcoming_segments(car, road_system, max_items=8)
        
        # Calculate segments starting from current position
        remaining_waypoints = car.path_waypoints[current_wp_idx:]
        if len(remaining_waypoints) < 3:
            return segments
        
        # Calculate distance to next waypoint for more accurate "in X distance" display
        if len(remaining_waypoints) > 0:
            next_wp_x, next_wp_y = remaining_waypoints[0]
            distance_to_next_wp = math.hypot(next_wp_x - car_x, next_wp_y - car_y)
        else:
            distance_to_next_wp = 0
        
        segment_idx = 0
        accumulated_distance = distance_to_next_wp  # Start with distance to next waypoint
        
        for i in range(len(remaining_waypoints) - 2):
            if segment_idx >= len(upcoming_roads) or len(segments) >= 4:
                break
                
            # Calculate turn angle between three consecutive waypoints
            p1 = remaining_waypoints[i]
            p2 = remaining_waypoints[i + 1]
            p3 = remaining_waypoints[i + 2]
            
            # Only add distance between waypoints for segments after the first
            if i > 0:
                segment_distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                accumulated_distance += segment_distance
            
            # Vector from p1 to p2
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            # Vector from p2 to p3
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate angle between vectors
            angle1 = math.atan2(v1[1], v1[0])
            angle2 = math.atan2(v2[1], v2[0])
            angle_diff = angle2 - angle1
            
            # Normalize angle difference to [-π, π]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Convert to degrees for turn instruction
            angle_diff_deg = math.degrees(angle_diff)
            
            # Only create a segment if there's a significant turn (more than 15 degrees)
            if abs(angle_diff_deg) > 15 or len(segments) == 0:  # Always include first segment
                # Get turn instruction
                instruction = self._format_turn_instruction(angle_diff_deg)
                
                # Get street name from road segments
                street_name = "Unknown Street"
                if segment_idx < len(upcoming_roads):
                    street_name = upcoming_roads[segment_idx][0]
                
                # Convert accumulated distance to appropriate units
                distance_m = accumulated_distance * 0.6  # Convert pixels to approximate meters
                
                # Format distance with more accurate representation
                if distance_m > 1000:
                    distance_str = f"{distance_m/1000:.1f} km"
                elif distance_m > 100:
                    distance_str = f"{int(distance_m)} m"
                else:
                    distance_str = f"{int(distance_m)} m"
                
                segments.append({
                    'name': street_name,
                    'street_name': street_name,
                    'distance': distance_m,
                    'distance_str': distance_str,
                    'action': instruction,
                    'turn_direction': instruction.lower(),
                    'angle_diff': angle_diff_deg,
                    'waypoint_index': current_wp_idx + i + 1  # Track which waypoint this relates to
                })
                
                segment_idx += 1
        
        return segments
    
    def update_from_simulation(self, simulation_core):
        """Update all data from the simulation core"""
        if simulation_core:
            self.update_cars(simulation_core.cars)
            self.update_roads(simulation_core.road_system)
            self.update_camera(simulation_core.camera)
    
    def to_dict(self):
        """Convert all data to dictionary for JSON serialization"""
        pathfinding_ok = True
        if simulation_core and simulation_core.cars:
            for car in simulation_core.cars:
                if not car.crashed and (not hasattr(car, 'path_waypoints') or not car.path_waypoints):
                    pathfinding_ok = False
                    break
        
        return {
            'cars': self.cars_data,
            'roads': self.roads_data,
            'camera': self.camera_data,
            'timestamp': time.time(),
            'generation': getattr(simulation_core, 'generation', 0) if simulation_core else 0,
            'mode': getattr(simulation_core, 'mode', 0) if simulation_core else 0,
            'pathfinding_ok': pathfinding_ok,
            'path_validation': {
                'paths_validated': getattr(simulation_core, 'paths_validated', False) if simulation_core else False,
                'valid_paths_count': getattr(simulation_core, 'valid_paths_count', 0) if simulation_core else 0,
                'required_paths_count': getattr(simulation_core, 'required_paths_count', 0) if simulation_core else 0
            }
        }

simulation_data = SimulationData()

# HTML template with Three.js 3D visualization
@app.route('/')
def index():
    with open('3dsim.html', 'r', encoding='utf-8') as file:
        html_template = file.read()
    return render_template_string(html_template)

async def websocket_handler(websocket, path):
    """Handle WebSocket connections"""
    print(f"New WebSocket client connected from {websocket.remote_address}")
    websocket_clients.add(websocket)
    print(f"Total WebSocket clients: {len(websocket_clients)}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                command = data.get('command')
                message_type = data.get('type')
                
                # Handle emergency stop messages
                if message_type == 'emergency_stop':
                    global emergency_stop_active
                    emergency_stop_active = data.get('active', False)
                    if emergency_stop_active:
                        print("EMERGENCY STOP ACTIVATED - Simulation paused")
                    else:
                        print("Emergency stop deactivated - Simulation resumed")
                
                elif command == 'pause':
                    if simulation_core:
                        simulation_core.running = False
                        print("Simulation paused")
                elif command == 'resume':
                    if simulation_core:
                        simulation_core.running = True
                        print("Simulation resumed")
                elif command == 'reset_generation':
                    if simulation_core and simulation_core.mode == 0:  # Training mode
                        print("Resetting generation...")
                        simulation_core.evolution_timer = simulation_core.evolution_interval
                elif command == 'change_mode':
                    if simulation_core:
                        new_mode = (simulation_core.mode + 1) % 3
                        simulation_core.mode = new_mode
                        simulation_core.initialize_mode()
                        print(f"Changed to mode {new_mode}")
                        
            except json.JSONDecodeError:
                print(f"Invalid JSON received: {message}")
            except Exception as e:
                print(f"Error handling WebSocket message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket client disconnected")
    finally:
        websocket_clients.remove(websocket)
        print(f"WebSocket client removed. Remaining clients: {len(websocket_clients)}")

async def broadcast_data():
    """Broadcast simulation data to all connected clients"""
    if websocket_clients and simulation_core and simulation_data:
        try:
            # Update visualization data from simulation
            simulation_data.update_from_simulation(simulation_core)
            data_dict = simulation_data.to_dict()
            
            # Debug logging
            if len(data_dict['cars']) == 0:
                print(f"No cars data to send. Simulation_core cars: {len(simulation_core.cars) if simulation_core.cars else 0}")
            elif len(data_dict['cars']) > 0:
                print(f"Broadcasting {len(data_dict['cars'])} cars and {len(data_dict['roads'])} roads")
            if len(data_dict['roads']) == 0:
                print(f"No roads data to send. Road system: {simulation_core.road_system is not None if simulation_core else 'No sim core'}")
            
            data = json.dumps(data_dict)
            
            # Create a copy of clients to avoid modification during iteration
            clients = websocket_clients.copy()
            for client in clients:
                try:
                    await client.send(data)
                except websockets.exceptions.ConnectionClosed:
                    websocket_clients.discard(client)
                except Exception as e:
                    print(f"Error sending data to client: {e}")
                    websocket_clients.discard(client)
        except Exception as e:
            print(f"Error in broadcast_data: {e}")
            import traceback
            traceback.print_exc()

def run_simulation():
    """Run the headless simulation core"""
    global simulation_core, simulation_running
    
    print("Starting 3D Car AI Simulation...")
    print("Open http://localhost:5000 in your browser to view the 3D visualization")
    
    # Initialize headless simulation
    simulation_core = HeadlessSimulationCore(
        width=1280,
        height=720,
        mode=1,  # Default to destination mode
        auto_mode=True,
        initial_scale=1.0
    )
    
    simulation_running = True
    
    # Simulation loop
    FPS = 60
    frame_duration = 1.0 / FPS
    
    try:
        while simulation_running:
            frame_start = time.time()
            
            # Update simulation only if running and not in emergency stop
            if simulation_core.running and not emergency_stop_active:
                simulation_core.update_simulation()
            elif emergency_stop_active:
                # During emergency stop, don't update car physics but keep everything else running
                pass
            
            # Control frame rate
            frame_end = time.time()
            frame_time = frame_end - frame_start
            if frame_time < frame_duration:
                time.sleep(frame_duration - frame_time)
                
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_running = False

def run_websocket_server():
    """Run the WebSocket server"""
    import websockets
    import asyncio
    
    # Set event loop policy for Windows threading
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Create new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def handle_client(websocket, path):
        await websocket_handler(websocket, path)
    
    async def periodic_broadcast():
        """Periodically broadcast data to clients"""
        print("Starting periodic broadcast loop...")
        broadcast_count = 0
        while True:  # Run indefinitely, check websocket_clients instead
            broadcast_count += 1
            if websocket_clients:
                print(f"Broadcasting to {len(websocket_clients)} clients... (#{broadcast_count})")
                await broadcast_data()
            elif broadcast_count % 50 == 0:  # Print every 5 seconds when no clients
                print("No WebSocket clients connected")
            await asyncio.sleep(1/30)  # 30 FPS data updates with 60fps interpolation
    
    async def main():
        # Start WebSocket server on all interfaces
        server = await websockets.serve(handle_client, "0.0.0.0", 8765)
        print("WebSocket server started on ws://localhost:8765")
        
        # Start periodic broadcasting
        broadcast_task = asyncio.create_task(periodic_broadcast())
        
        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            print("WebSocket server stopped")
        finally:
            broadcast_task.cancel()
            server.close()
    
    # Run the WebSocket server
    try:
        loop.run_until_complete(main())
    except Exception as e:
        print(f"WebSocket server error: {e}")
    finally:
        loop.close()

if __name__ == '__main__':
    print("Starting 3D Car AI Simulation...")
    print("Open http://localhost:5000 in your browser to view the 3D visualization")
    
    # Start simulation in a separate thread
    simulation_thread = threading.Thread(target=run_simulation)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    # Start WebSocket server in a separate thread
    websocket_thread = threading.Thread(target=run_websocket_server)
    websocket_thread.daemon = True
    websocket_thread.start()
    
    # Give WebSocket server time to start
    time.sleep(2)
    
    # Run Flask app
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        simulation_running = False
        print("Shutting down 3D visualization...")
