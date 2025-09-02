import sys
import os
import math
import requests
import json
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
                            QPushButton, QLabel, QFrame, QCompleter, QListWidget, 
                            QListWidgetItem, QAbstractItemView)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl, QTimer, QThread, pyqtSlot, QStringListModel
from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6.QtWebEngineWidgets import QWebEngineView


class AddressSearcher(QThread):
    """Background thread for searching addresses via API"""
    searchCompleted = pyqtSignal(list)  # List of address dictionaries
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.search_query = None
    
    def search_addresses(self, query):
        """Start searching for addresses with the given query"""
        self.search_query = query
        if not self.isRunning():
            self.start()
    
    def run(self):
        if not self.search_query:
            return
            
        try:
            # Use Nominatim API for address searching
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': self.search_query,
                'format': 'json',
                'limit': 8,  # Get up to 8 results for suggestions
                'addressdetails': 1,
                'countrycodes': 'nz',  # Restrict to New Zealand only
                'bounded': 1,
                'viewbox': '166.509144322,-47.2899370765,178.517093541,-34.4506617165'  # NZ bounding box
            }
            
            # Add proper headers to avoid 403 errors
            headers = {
                'User-Agent': 'RoutePlanner/1.0 (Educational Project)',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for location in data:
                    # Create address dictionary matching expected format
                    address = {
                        'lat': location['lat'],
                        'lon': location['lon'],
                        'display_name': self.clean_display_name(location['display_name'])
                    }
                    results.append(address)
                
                self.searchCompleted.emit(results)
            else:
                self.error.emit(f"Search failed with status code: {response.status_code}")
                
        except Exception as e:
            self.error.emit(f"Error searching addresses: {str(e)}")
    
    def clean_display_name(self, display_name):
        """Clean up the display name from Nominatim for better readability"""
        # Split by commas and take relevant parts
        parts = [part.strip() for part in display_name.split(',')]
        
        # Remove duplicate "New Zealand" entries and clean up
        cleaned_parts = []
        seen_parts = set()
        
        for part in parts:
            if part.lower() not in seen_parts and part.lower() != 'new zealand':
                cleaned_parts.append(part)
                seen_parts.add(part.lower())
        
        # Take the first 3-4 most relevant parts
        if len(cleaned_parts) > 4:
            cleaned_parts = cleaned_parts[:4]
        
        # Add New Zealand back at the end
        result = ', '.join(cleaned_parts) + ', New Zealand'
        return result


class LocationGeocoder(QThread):
    """Background thread for geocoding locations"""
    locationFound = pyqtSignal(str, float, float)  # location_name, lat, lng
    error = pyqtSignal(str)
    
    def __init__(self, location_name):
        super().__init__()
        self.location_name = location_name
    
    def run(self):
        try:
            # Use Nominatim API for geocoding with New Zealand focus
            success = self.try_nominatim_geocoding()
            
            if not success:
                self.error.emit(f"Could not find location '{self.location_name}' in New Zealand. Try a more specific address.")
                
        except Exception as e:
            self.error.emit(f"Error geocoding location: {str(e)}")
    
    def try_nominatim_geocoding(self):
        """Try geocoding with Nominatim API focused on New Zealand"""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': self.location_name,
                'format': 'json',
                'limit': 5,
                'addressdetails': 1,
                'countrycodes': 'nz',  # Restrict to New Zealand only
                'bounded': 1,
                'viewbox': '166.509144322,-47.2899370765,178.517093541,-34.4506617165'  # NZ bounding box
            }
            
            # Add proper headers to avoid 403 errors
            headers = {
                'User-Agent': 'RoutePlanner/1.0 (Educational Project)',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    location = data[0]
                    lat = float(location['lat'])
                    lng = float(location['lon'])
                    display_name = location['display_name']
                    self.locationFound.emit(display_name, lat, lng)
                    return True
            
            return False
            
        except Exception as e:
            print(f"Nominatim geocoding failed: {e}")
            return False
    
    



class OSMRouteMapView(QWebEngineView):
    """Web view that displays OSM route map"""
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #1a1a1a; border-radius: 12px;")
        self.start_location = None
        self.end_location = None
        
        # Load initial empty map
        self.load_initial_map()
        
        # Connect page loaded signal
        self.loadFinished.connect(self.on_load_finished)
    
    def load_initial_map(self):
        """Load initial map centered on default location"""
        html_content = self.create_map_html()
        self.setHtml(html_content)
    
    def create_map_html(self, start_lat=-41.2865, start_lng=174.7762, 
                       end_lat=None, end_lng=None, route_coords=None):
        """Create HTML content with Leaflet map"""
        
        # Calculate map center and zoom
        if end_lat and end_lng:
            center_lat = (start_lat + end_lat) / 2
            center_lng = (start_lng + end_lng) / 2
            
            # Calculate zoom level based on distance
            distance = self.calculate_distance(start_lat, start_lng, end_lat, end_lng)
            zoom = max(6, min(15, int(15 - math.log10(distance + 1) * 2)))
        else:
            center_lat, center_lng = start_lat, start_lng
            zoom = 13
        
        route_js = ""
        if route_coords:
            coords_str = json.dumps(route_coords)
            route_js = f"""
                // Add route line
                var routeCoords = {coords_str};
                var routeLine = L.polyline(routeCoords, {{
                    color: '#00ff88',
                    weight: 4,
                    opacity: 0.8,
                    smoothFactor: 1
                }}).addTo(map);
                
                // Fit map to route bounds
                if (routeCoords.length > 0) {{
                    var group = new L.featureGroup([routeLine]);
                    map.fitBounds(group.getBounds().pad(0.1));
                }}
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: #1a1a1a;
                    overflow: hidden;
                }}
                
                #map {{
                    height: 100vh;
                    width: 100%;
                    border-radius: 12px;
                }}
                
                .custom-marker {{
                    background: #fff;
                    border: 3px solid #00ff88;
                    border-radius: 50%;
                    width: 20px;
                    height: 20px;
                    box-shadow: 0 4px 12px rgba(0, 255, 136, 0.4);
                }}
                
                .start-marker {{
                    border-color: #00ff88;
                    background: #00ff88;
                }}
                
                .end-marker {{
                    border-color: #ff4444;
                    background: #ff4444;
                }}
                
                .leaflet-popup-content-wrapper {{
                    background: rgba(0, 0, 0, 0.8);
                    color: white;
                    border-radius: 8px;
                    backdrop-filter: blur(10px);
                }}
                
                .leaflet-popup-tip {{
                    background: rgba(0, 0, 0, 0.8);
                }}
                
                .route-info {{
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    background: rgba(0, 0, 0, 0.8);
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    z-index: 1000;
                    min-width: 200px;
                }}
                
                .route-stat {{
                    margin: 5px 0;
                    font-size: 14px;
                }}
                
                .route-stat strong {{
                    color: #00ff88;
                }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <script>
                // Initialize map
                var map = L.map('map', {{
                    center: [{center_lat}, {center_lng}],
                    zoom: {zoom},
                    zoomControl: true,
                    attributionControl: true
                }});
                
                // Add tile layer
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                    maxZoom: 19
                }}).addTo(map);
                
                // Add markers
                var startMarker = null;
                var endMarker = null;
                
                function addStartMarker(lat, lng, title) {{
                    if (startMarker) {{
                        map.removeLayer(startMarker);
                    }}
                    
                    var customIcon = L.divIcon({{
                        className: 'custom-marker start-marker',
                        iconSize: [20, 20],
                        iconAnchor: [10, 10]
                    }});
                    
                    startMarker = L.marker([lat, lng], {{icon: customIcon}})
                        .addTo(map)
                        .bindPopup('<b>Start:</b><br>' + title);
                }}
                
                function addEndMarker(lat, lng, title) {{
                    if (endMarker) {{
                        map.removeLayer(endMarker);
                    }}
                    
                    var customIcon = L.divIcon({{
                        className: 'custom-marker end-marker',
                        iconSize: [20, 20],
                        iconAnchor: [10, 10]
                    }});
                    
                    endMarker = L.marker([lat, lng], {{icon: customIcon}})
                        .addTo(map)
                        .bindPopup('<b>Destination:</b><br>' + title);
                }}
                
                // Add markers if coordinates provided
                {f"addStartMarker({start_lat}, {start_lng}, 'Start Location');" if start_lat and start_lng else ""}
                {f"addEndMarker({end_lat}, {end_lng}, 'End Location');" if end_lat and end_lng else ""}
                
                {route_js}
                
                // Expose functions to Python
                window.updateStartLocation = function(lat, lng, title) {{
                    addStartMarker(lat, lng, title);
                }};
                
                window.updateEndLocation = function(lat, lng, title) {{
                    addEndMarker(lat, lng, title);
                }};
                
                window.updateRoute = function(routeCoords, distance, duration) {{
                    // Remove existing route
                    map.eachLayer(function(layer) {{
                        if (layer instanceof L.Polyline && !(layer instanceof L.Polygon)) {{
                            if (layer !== startMarker && layer !== endMarker) {{
                                map.removeLayer(layer);
                            }}
                        }}
                    }});
                    
                    // Add new route
                    if (routeCoords && routeCoords.length > 0) {{
                        var routeLine = L.polyline(routeCoords, {{
                            color: '#00ff88',
                            weight: 4,
                            opacity: 0.8,
                            smoothFactor: 1
                        }}).addTo(map);
                        
                        // Fit map to route bounds
                        var group = new L.featureGroup([routeLine]);
                        map.fitBounds(group.getBounds().pad(0.1));
                        
                        // Add route info panel
                        var routeInfo = document.querySelector('.route-info');
                        if (!routeInfo) {{
                            routeInfo = document.createElement('div');
                            routeInfo.className = 'route-info';
                            document.body.appendChild(routeInfo);
                        }}
                        
                        routeInfo.innerHTML = `
                            <h3 style="margin: 0 0 10px 0; color: #00ff88;">Route Information</h3>
                            <div class="route-stat"><strong>Distance:</strong> ${{distance}} km</div>
                            <div class="route-stat"><strong>Est. Duration:</strong> ${{duration}} min</div>
                            <div class="route-stat"><strong>Route Type:</strong> Shortest Path</div>
                        `;
                    }}
                }};
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def calculate_distance(self, lat1, lng1, lat2, lng2):
        """Calculate distance between two points in kilometers"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = (math.sin(delta_lat / 2) * math.sin(delta_lat / 2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lng / 2) * math.sin(delta_lng / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def update_start_location(self, lat, lng, title):
        """Update start location marker"""
        self.start_location = (lat, lng, title)
        self.page().runJavaScript(f"window.updateStartLocation({lat}, {lng}, '{title}');")
    
    def update_end_location(self, lat, lng, title):
        """Update end location marker"""
        self.end_location = (lat, lng, title)
        self.page().runJavaScript(f"window.updateEndLocation({lat}, {lng}, '{title}');")
    
    def generate_route(self):
        """Generate and display route between start and end locations"""
        if not self.start_location or not self.end_location:
            return
        
        # Get route using OSRM API with fallback
        start_lat, start_lng, _ = self.start_location
        end_lat, end_lng, _ = self.end_location
        
        # Try OSRM first
        route_data = self.try_osrm_routing(start_lat, start_lng, end_lat, end_lng)
        
        if route_data:
            route_coords, distance_km, duration_min = route_data
        else:
            # Fallback to straight line route
            print("Using fallback straight-line routing")
            route_coords = [[start_lat, start_lng], [end_lat, end_lng]]
            distance_km = round(self.calculate_distance(start_lat, start_lng, end_lat, end_lng), 1)
            duration_min = round(distance_km * 2)  # Rough estimate: 30 km/h average
        
        # Update map with route
        coords_json = json.dumps(route_coords)
        self.page().runJavaScript(
            f"window.updateRoute({coords_json}, {distance_km}, {duration_min});"
        )
    
    def try_osrm_routing(self, start_lat, start_lng, end_lat, end_lng):
        """Try to get route from OSRM API"""
        try:
            # Use OSRM demo server for routing
            url = f"http://router.project-osrm.org/route/v1/driving/{start_lng},{start_lat};{end_lng},{end_lat}"
            params = {
                'overview': 'full',
                'geometries': 'geojson'
            }
            
            headers = {
                'User-Agent': 'RoutePlanner/1.0 (Educational Project)',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('routes'):
                    route = data['routes'][0]
                    geometry = route['geometry']['coordinates']
                    
                    # Convert to lat,lng format for Leaflet
                    route_coords = [[coord[1], coord[0]] for coord in geometry]
                    
                    # Calculate route statistics
                    distance_m = route['distance']
                    duration_s = route['duration']
                    
                    distance_km = round(distance_m / 1000, 1)
                    duration_min = round(duration_s / 60)
                    
                    return route_coords, distance_km, duration_min
            
            return None
            
        except Exception as e:
            print(f"OSRM routing failed: {e}")
            return None
    
    def on_load_finished(self, ok):
        """Called when the page finishes loading"""
        if ok:
            print("OSM Route Map loaded successfully")
        else:
            print("Failed to load OSM Route Map")


class AutoCompleteLineEdit(QLineEdit):
    """Custom QLineEdit with autocomplete functionality for locations"""
    locationSelected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.address_searcher = AddressSearcher()
        self.current_suggestions = []
        self.selected_location = None
        
        # Connect search results
        self.address_searcher.searchCompleted.connect(self.update_suggestions)
        
        # Timer for debounced searching
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.perform_search)
        
        # Create suggestion list widget
        self.suggestion_list = QListWidget()
        self.suggestion_list.setWindowFlags(Qt.WindowType.Popup)
        self.suggestion_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.suggestion_list.setMouseTracking(True)
        self.suggestion_list.setMaximumHeight(200)
        self.suggestion_list.setStyleSheet("""
            QListWidget {
                background-color: rgba(40, 40, 40, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                color: white;
                font-size: 13px;
                selection-background-color: rgba(0, 255, 136, 0.3);
                alternate-background-color: rgba(255, 255, 255, 0.05);
            }
            QListWidget::item {
                padding: 8px 12px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            QListWidget::item:hover {
                background-color: rgba(0, 255, 136, 0.2);
            }
            QListWidget::item:selected {
                background-color: rgba(0, 255, 136, 0.3);
            }
        """)
        
        # Connect signals
        self.textChanged.connect(self.on_text_changed)
        self.suggestion_list.itemClicked.connect(self.on_suggestion_clicked)
        
        # Handle key events
        self.suggestion_list.keyPressEvent = self.suggestion_key_press_event
    
    def on_text_changed(self, text):
        """Handle text changes with debounced searching"""
        if len(text.strip()) < 3:
            self.hide_suggestions()
            self.search_timer.stop()
            return
        
        # Debounce the search - wait 500ms before searching
        self.search_timer.stop()
        self.search_timer.start(500)
    
    def perform_search(self):
        """Perform the actual address search"""
        query = self.text().strip()
        if len(query) >= 3:
            self.address_searcher.search_addresses(query)
    
    def update_suggestions(self, addresses):
        """Update suggestions from API search results"""
        if not addresses:
            self.hide_suggestions()
            return
        
        self.current_suggestions = addresses
        
        # Convert to display format
        suggestions = []
        for addr in addresses[:8]:  # Limit to 8 suggestions
            display_name = addr['display_name']
            suggestions.append((display_name, display_name, "🇳🇿 New Zealand"))
        
        self.show_suggestions(suggestions)
    
    def show_suggestions(self, suggestions):
        """Show the suggestions list"""
        self.suggestion_list.clear()
        
        for city, display_name, country_info in suggestions:
            item = QListWidgetItem(f"{display_name}")
            item.setData(Qt.ItemDataRole.UserRole, display_name)
            self.suggestion_list.addItem(item)
        
        # Position the suggestion list below the input
        global_pos = self.mapToGlobal(self.rect().bottomLeft())
        self.suggestion_list.move(global_pos)
        self.suggestion_list.resize(self.width(), min(200, self.suggestion_list.sizeHintForRow(0) * len(suggestions) + 10))
        self.suggestion_list.show()
    
    def hide_suggestions(self):
        """Hide the suggestions list"""
        self.suggestion_list.hide()
    
    def on_suggestion_clicked(self, item):
        """Handle suggestion selection"""
        display_name = item.data(Qt.ItemDataRole.UserRole)
        
        # Find the full address data
        for addr in self.current_suggestions:
            if addr['display_name'] == display_name:
                self.setText(display_name)
                self.selected_location = addr
                break
        
        self.hide_suggestions()
        self.locationSelected.emit(display_name)
    
    def get_selected_coordinates(self):
        """Get coordinates for the currently selected location"""
        if self.selected_location:
            return (float(self.selected_location['lat']), 
                    float(self.selected_location['lon']), 
                    self.selected_location['display_name'])
        
        # Try to find in current suggestions based on text
        current_text = self.text().strip()
        for addr in self.current_suggestions:
            if addr['display_name'] == current_text:
                return float(addr['lat']), float(addr['lon']), addr['display_name']
        
        return None, None, current_text
    
    def suggestion_key_press_event(self, event):
        """Handle key events in suggestion list"""
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            current_item = self.suggestion_list.currentItem()
            if current_item:
                self.on_suggestion_clicked(current_item)
        elif event.key() == Qt.Key.Key_Escape:
            self.hide_suggestions()
            self.setFocus()
        else:
            # Pass other key events to the line edit
            QLineEdit.keyPressEvent(self, event)
    
    def keyPressEvent(self, event):
        """Handle key events in the line edit"""
        if event.key() == Qt.Key.Key_Down:
            if self.suggestion_list.isVisible():
                self.suggestion_list.setFocus()
                self.suggestion_list.setCurrentRow(0)
            return
        elif event.key() == Qt.Key.Key_Escape:
            self.hide_suggestions()
            return
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            if self.suggestion_list.isVisible():
                current_item = self.suggestion_list.currentItem()
                if current_item:
                    self.on_suggestion_clicked(current_item)
                    return
        
        super().keyPressEvent(event)
    
    def focusOutEvent(self, event):
        """Hide suggestions when focus is lost"""
        # Delay hiding to allow clicking on suggestions
        QTimer.singleShot(150, self.hide_suggestions)
        super().focusOutEvent(event)


class RoutePlannerWidget(QWidget):
    """Main widget for route planning interface"""
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                color: white;
                font-family: 'Segoe UI', sans-serif;
            }
            
            QLineEdit {
                background-color: rgba(255, 255, 255, 0.1);
                border: 2px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 14px;
                color: white;
            }
            
            QLineEdit:focus {
                border-color: #00ff88;
                background-color: rgba(255, 255, 255, 0.15);
            }
            
            QPushButton {
                background-color: #00ff88;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                color: black;
            }
            
            QPushButton:hover {
                background-color: #00dd77;
            }
            
            QPushButton:pressed {
                background-color: #00bb66;
            }
            
            QPushButton:disabled {
                background-color: rgba(255, 255, 255, 0.2);
                color: rgba(255, 255, 255, 0.5);
            }
            
            QLabel {
                font-size: 14px;
                color: rgba(255, 255, 255, 0.8);
            }
            
            QFrame {
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                background-color: rgba(255, 255, 255, 0.05);
            }
        """)
        
        self.init_ui()
        
        # Initialize location data
        self.start_coords = None
        self.end_coords = None
        
        # Geocoding threads
        self.start_geocoder = None
        self.end_geocoder = None
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Route Planner - OSM Navigation")
        self.setGeometry(100, 100, 1400, 900)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Title
        title_label = QLabel("Route Planner")
        title_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #00ff88; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        # Input panel
        input_frame = QFrame()
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(20, 20, 20, 20)
        input_layout.setSpacing(15)
        
        # Start location input
        start_label = QLabel("Starting Location:")
        start_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        input_layout.addWidget(start_label)
        
        self.start_input = AutoCompleteLineEdit()
        self.start_input.setPlaceholderText("Enter starting address or location...")
        self.start_input.returnPressed.connect(self.geocode_start_location)
        self.start_input.locationSelected.connect(self.on_start_location_autocomplete)
        input_layout.addWidget(self.start_input)
        
        # End location input
        end_label = QLabel("Destination:")
        end_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        input_layout.addWidget(end_label)
        
        self.end_input = AutoCompleteLineEdit()
        self.end_input.setPlaceholderText("Enter destination address or location...")
        self.end_input.returnPressed.connect(self.geocode_end_location)
        self.end_input.locationSelected.connect(self.on_end_location_autocomplete)
        input_layout.addWidget(self.end_input)
        
        # Buttons layout
        button_layout = QHBoxLayout()
        
        self.find_route_btn = QPushButton("Find Route")
        self.find_route_btn.clicked.connect(self.find_route)
        self.find_route_btn.setEnabled(False)
        button_layout.addWidget(self.find_route_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_route)
        button_layout.addWidget(self.clear_btn)
        
        input_layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("Enter starting location and destination to plan route")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: rgba(255, 255, 255, 0.6); font-style: italic;")
        input_layout.addWidget(self.status_label)
        
        main_layout.addWidget(input_frame)
        
        # Map view
        self.map_view = OSMRouteMapView()
        main_layout.addWidget(self.map_view, 1)  # Give map view most of the space
        
        self.setLayout(main_layout)
    
    def geocode_start_location(self):
        """Geocode the starting location using manual search"""
        location_text = self.start_input.text().strip()
        if not location_text:
            return
        
        self.status_label.setText("Finding starting location...")
        self.start_input.setEnabled(False)
        
        # Use the address searcher from the input widget
        self.start_input.address_searcher.searchCompleted.connect(self.on_start_manual_search_complete)
        self.start_input.address_searcher.search_addresses(location_text)
    
    def geocode_end_location(self):
        """Geocode the destination location using manual search"""
        location_text = self.end_input.text().strip()
        if not location_text:
            return
        
        self.status_label.setText("Finding destination...")
        self.end_input.setEnabled(False)
        
        # Use the address searcher from the input widget
        self.end_input.address_searcher.searchCompleted.connect(self.on_end_manual_search_complete)
        self.end_input.address_searcher.search_addresses(location_text)
    
    def on_start_manual_search_complete(self, addresses):
        """Handle manual search completion for start location"""
        self.start_input.setEnabled(True)
        
        if addresses:
            # Take the first result
            addr = addresses[0]
            lat, lng = float(addr['lat']), float(addr['lon'])
            display_name = addr['display_name']
            
            self.start_coords = (lat, lng, display_name)
            self.map_view.update_start_location(lat, lng, display_name)
            
            if self.end_coords:
                self.status_label.setText("Both locations found. Click 'Find Route' to generate route.")
                self.find_route_btn.setEnabled(True)
            else:
                self.status_label.setText("Starting location set. Enter destination.")
        else:
            self.status_label.setText("Could not find starting location. Try a different search.")
        
        # Disconnect the signal to avoid multiple connections
        self.start_input.address_searcher.searchCompleted.disconnect(self.on_start_manual_search_complete)
    
    def on_end_manual_search_complete(self, addresses):
        """Handle manual search completion for end location"""
        self.end_input.setEnabled(True)
        
        if addresses:
            # Take the first result
            addr = addresses[0]
            lat, lng = float(addr['lat']), float(addr['lon'])
            display_name = addr['display_name']
            
            self.end_coords = (lat, lng, display_name)
            self.map_view.update_end_location(lat, lng, display_name)
            
            if self.start_coords:
                self.status_label.setText("Both locations found. Click 'Find Route' to generate route.")
                self.find_route_btn.setEnabled(True)
            else:
                self.status_label.setText("Destination set. Enter starting location.")
        else:
            self.status_label.setText("Could not find destination. Try a different search.")
        
        # Disconnect the signal to avoid multiple connections
        self.end_input.address_searcher.searchCompleted.disconnect(self.on_end_manual_search_complete)
    
    def on_start_location_autocomplete(self, display_name):
        """Handle start location selected from autocomplete"""
        self.status_label.setText("Processing starting location...")
        
        # Get coordinates directly from the autocomplete widget
        lat, lng, full_name = self.start_input.get_selected_coordinates()
        
        if lat is not None and lng is not None:
            self.start_coords = (lat, lng, full_name)
            self.map_view.update_start_location(lat, lng, full_name)
            
            if self.end_coords:
                self.status_label.setText("Both locations found. Click 'Find Route' to generate route.")
                self.find_route_btn.setEnabled(True)
            else:
                self.status_label.setText("Starting location set. Enter destination.")
        else:
            self.status_label.setText("Could not get coordinates for starting location.")
    
    def on_end_location_autocomplete(self, display_name):
        """Handle end location selected from autocomplete"""
        self.status_label.setText("Processing destination...")
        
        # Get coordinates directly from the autocomplete widget
        lat, lng, full_name = self.end_input.get_selected_coordinates()
        
        if lat is not None and lng is not None:
            self.end_coords = (lat, lng, full_name)
            self.map_view.update_end_location(lat, lng, full_name)
            
            if self.start_coords:
                self.status_label.setText("Both locations found. Click 'Find Route' to generate route.")
                self.find_route_btn.setEnabled(True)
            else:
                self.status_label.setText("Destination set. Enter starting location.")
        else:
            self.status_label.setText("Could not get coordinates for destination.")
    
    @pyqtSlot(str, float, float)
    def on_start_location_found(self, display_name, lat, lng):
        """Handle start location found"""
        self.start_coords = (lat, lng, display_name)
        self.map_view.update_start_location(lat, lng, display_name)
        self.start_input.setEnabled(True)
        
        if self.end_coords:
            self.status_label.setText("Both locations found. Click 'Find Route' to generate route.")
            self.find_route_btn.setEnabled(True)
        else:
            self.status_label.setText("Starting location set. Enter destination.")
    
    @pyqtSlot(str, float, float)
    def on_end_location_found(self, display_name, lat, lng):
        """Handle end location found"""
        self.end_coords = (lat, lng, display_name)
        self.map_view.update_end_location(lat, lng, display_name)
        self.end_input.setEnabled(True)
        
        if self.start_coords:
            self.status_label.setText("Both locations found. Click 'Find Route' to generate route.")
            self.find_route_btn.setEnabled(True)
        else:
            self.status_label.setText("Destination set. Enter starting location.")
    
    @pyqtSlot(str)
    def on_geocoding_error(self, error_message):
        """Handle geocoding error"""
        self.status_label.setText(f"Error: {error_message}")
        self.start_input.setEnabled(True)
        self.end_input.setEnabled(True)
    
    def find_route(self):
        """Find and display route between locations"""
        if not self.start_coords or not self.end_coords:
            return
        
        self.status_label.setText("Calculating route...")
        self.find_route_btn.setEnabled(False)
        
        # Generate route on map
        self.map_view.generate_route()
        
        # Re-enable button and update status
        QTimer.singleShot(2000, self.on_route_generated)
    
    def on_route_generated(self):
        """Called after route is generated"""
        self.find_route_btn.setEnabled(True)
        self.status_label.setText("Route generated successfully!")
    
    def clear_route(self):
        """Clear all inputs and reset map"""
        self.start_input.clear()
        self.end_input.clear()
        self.start_coords = None
        self.end_coords = None
        self.find_route_btn.setEnabled(False)
        self.status_label.setText("Enter starting location and destination to plan route")
        
        # Reload map to clear markers and route
        self.map_view.load_initial_map()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Route Planner")
    app.setApplicationVersion("1.0")
    
    # Create and show the main window
    main_window = RoutePlannerWidget()
    main_window.show()
    
    sys.exit(app.exec())
