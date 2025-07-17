import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLineEdit, QPushButton, QLabel, 
                             QSplitter, QTextEdit, QListWidget, 
                             QListWidgetItem, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QFont
from PyQt5.QtWebEngineWidgets import QWebEngineView
import folium
import tempfile
import os
import time

def create_session():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    
    # Define retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

class GeolocationThread(QThread):
    """Thread for getting user's current location using IP geolocation"""
    location_ready = pyqtSignal(float, float, str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def run(self):
        try:
            # Try multiple IP geolocation services
            services = [
                "http://ip-api.com/json/",
                "https://ipapi.co/json/",
                "https://ipinfo.io/json"
            ]
            
            for service_url in services:
                try:
                    response = requests.get(
                        service_url, 
                        timeout=10,
                        headers={'User-Agent': 'LocationGUI/1.0'}
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract coordinates based on service
                    if "ip-api.com" in service_url:
                        lat = data.get('lat')
                        lon = data.get('lon')
                        location_name = f"{data.get('city', 'Unknown')}, {data.get('country', 'Unknown')}"
                    elif "ipapi.co" in service_url:
                        lat = data.get('latitude')
                        lon = data.get('longitude')
                        location_name = f"{data.get('city', 'Unknown')}, {data.get('country_name', 'Unknown')}"
                    elif "ipinfo.io" in service_url:
                        loc = data.get('loc', '').split(',')
                        if len(loc) == 2:
                            lat = float(loc[0])
                            lon = float(loc[1])
                        else:
                            continue
                        location_name = f"{data.get('city', 'Unknown')}, {data.get('country', 'Unknown')}"
                    
                    if lat and lon:
                        self.location_ready.emit(float(lat), float(lon), location_name)
                        return
                        
                except requests.exceptions.ConnectionError:
                    continue
                except requests.exceptions.Timeout:
                    continue
                except Exception as e:
                    continue
            
            # If all services fail, emit error
            self.error_occurred.emit("Unable to determine location from any service")
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class AddressSearchThread(QThread):
    """Thread for searching addresses using Nominatim API"""
    results_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, query, parent=None):
        super().__init__(parent)
        self.query = query
        
    def run(self):
        try:
            # Create session with retry strategy
            session = create_session()
            
            # Use Nominatim API for address search
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': self.query,
                'format': 'json',
                'limit': 10,
                'addressdetails': 1
            }
            
            headers = {
                'User-Agent': 'LocationGUI/1.0 (contact@example.com)'
            }
            
            # Retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = session.get(
                        url, 
                        params=params, 
                        headers=headers, 
                        timeout=15
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Format results
                    formatted_results = []
                    for item in data:
                        display_name = item.get('display_name', '')
                        lat = float(item.get('lat', 0))
                        lon = float(item.get('lon', 0))
                        
                        formatted_results.append({
                            'display_name': display_name,
                            'lat': lat,
                            'lon': lon
                        })
                    
                    self.results_ready.emit(formatted_results)
                    return
                    
                except requests.exceptions.ConnectionError as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    else:
                        self.error_occurred.emit(f"Connection failed after {max_retries} attempts. Check your internet connection.")
                        return
                        
                except requests.exceptions.Timeout as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    else:
                        self.error_occurred.emit(f"Request timed out after {max_retries} attempts. Server may be busy.")
                        return
                        
                except requests.exceptions.HTTPError as e:
                    self.error_occurred.emit(f"HTTP error: {e}")
                    return
                    
                except requests.exceptions.RequestException as e:
                    self.error_occurred.emit(f"Request failed: {str(e)}")
                    return
                    
            session.close()
                    
        except Exception as e:
            self.error_occurred.emit(f"Unexpected error: {str(e)}")

class LocationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_lat = 40.7128  # Default to NYC
        self.current_lon = -74.0060
        self.search_results = []
        self.temp_file = None
        self.location_name = "Default Location"
        
        self.init_ui()
        self.get_user_location()  # Get user's location on startup
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Location Search with OSM Map")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
            }
            QLineEdit {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
            QPushButton {
                padding: 8px 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                selection-background-color: #e3f2fd;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:hover {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create splitter for sidebar and main area
        splitter = QSplitter(Qt.Horizontal)
        
        # Create sidebar
        sidebar = self.create_sidebar()
        sidebar.setMaximumWidth(350)
        sidebar.setMinimumWidth(300)
        # Create map area
        self.map_view = QWebEngineView()
        splitter.addWidget(sidebar)
        splitter.addWidget(self.map_view)
        
        # Set splitter proportions
        splitter.setSizes([300, 900])
        
        # Set main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
    def create_sidebar(self):
        """Create the sidebar with address search functionality"""
        sidebar = QWidget()
        sidebar.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border-right: 1px solid #ddd;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Location Search")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Search input
        search_label = QLabel("Enter Address:")
        search_label.setStyleSheet("font-weight: bold; color: #34495e;")
        layout.addWidget(search_label)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("e.g., 1600 Pennsylvania Avenue, Washington, DC")
        self.search_input.returnPressed.connect(self.search_address)
        layout.addWidget(self.search_input)
        
        # Search button
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.search_address)
        layout.addWidget(search_btn)
        
        # Results label
        results_label = QLabel("Search Results:")
        results_label.setStyleSheet("font-weight: bold; color: #34495e; margin-top: 10px;")
        layout.addWidget(results_label)
        
        # Results list
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.on_result_selected)
        layout.addWidget(self.results_list)
        
        # Status label
        self.status_label = QLabel("Getting your location...")
        self.status_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        # Current location button
        current_location_btn = QPushButton("Refresh Current Location")
        current_location_btn.clicked.connect(self.get_user_location)
        layout.addWidget(current_location_btn)
        
        sidebar.setLayout(layout)
        return sidebar
        
    def get_user_location(self):
        """Get user's current location using IP geolocation"""
        self.status_label.setText("Getting your location...")
        
        # Start geolocation in separate thread
        self.geolocation_thread = GeolocationThread()
        self.geolocation_thread.location_ready.connect(self.handle_location_result)
        self.geolocation_thread.error_occurred.connect(self.handle_location_error)
        self.geolocation_thread.start()
        
    def handle_location_result(self, lat, lon, location_name):
        """Handle successful geolocation result"""
        self.current_lat = lat
        self.current_lon = lon
        self.location_name = location_name
        self.update_map()
        self.status_label.setText(f"Current location: {location_name}")
        
    def handle_location_error(self, error):
        """Handle geolocation error"""
        self.status_label.setText(f"Location error: Using default location")
        # Use default location (NYC) if geolocation fails
        self.current_lat = 40.7128
        self.current_lon = -74.0060
        self.location_name = "New York City, NY (Default)"
        self.update_map()
        
    def search_address(self):
        """Search for addresses using the input text"""
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Warning", "Please enter an address to search.")
            return
            
        self.status_label.setText("Searching...")
        self.results_list.clear()
        
        # Start search in separate thread
        self.search_thread = AddressSearchThread(query)
        self.search_thread.results_ready.connect(self.handle_search_results)
        self.search_thread.error_occurred.connect(self.handle_search_error)
        self.search_thread.start()
        
    def handle_search_results(self, results):
        """Handle search results from the thread"""
        self.search_results = results
        self.results_list.clear()
        
        if not results:
            self.status_label.setText("No results found.")
            return
            
        for result in results:
            item = QListWidgetItem(result['display_name'])
            item.setData(Qt.UserRole, result)
            self.results_list.addItem(item)
            
        self.status_label.setText(f"Found {len(results)} results.")
        
    def handle_search_error(self, error):
        """Handle search errors"""
        self.status_label.setText(f"Error: {error}")
        QMessageBox.critical(self, "Search Error", f"Failed to search: {error}")
        
    def on_result_selected(self, item):
        """Handle selection of a search result"""
        result = item.data(Qt.UserRole)
        if result:
            self.current_lat = result['lat']
            self.current_lon = result['lon']
            self.location_name = result['display_name']
            self.update_map()
            self.status_label.setText(f"Selected: {result['display_name'][:50]}...")
            
    def update_map(self):
        """Update the map with current location"""
        try:
            # Create folium map
            m = folium.Map(
                location=[self.current_lat, self.current_lon],
                zoom_start=15,
                tiles='OpenStreetMap'
            )
            
            # Add marker for current location
            folium.Marker(
                [self.current_lat, self.current_lon],
                popup=f"{self.location_name}<br>Lat: {self.current_lat:.4f}<br>Lon: {self.current_lon:.4f}",
                tooltip="Click for details",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Add circle around the location
            folium.Circle(
                location=[self.current_lat, self.current_lon],
                radius=500,
                popup="500m radius",
                color='blue',
                fill=True,
                opacity=0.3
            ).add_to(m)
            
            # Save map to temporary file
            if self.temp_file:
                try:
                    os.unlink(self.temp_file)
                except:
                    pass
                    
            self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html').name
            m.save(self.temp_file)
            
            # Load the HTML file into the web engine view
            file_url = QUrl.fromLocalFile(self.temp_file)
            self.map_view.load(file_url)
            
        except Exception as e:
            self.status_label.setText(f"Error updating map: {str(e)}")
            QMessageBox.critical(self, "Map Error", f"Failed to update map: {str(e)}")
    
    def closeEvent(self, event):
        """Clean up when closing the application"""
        if self.temp_file:
            try:
                os.unlink(self.temp_file)
            except:
                pass
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Location Search GUI")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = LocationGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()