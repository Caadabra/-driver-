import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal, QUrl, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6.QtWebEngineWidgets import QWebEngineView


class NavigationWebView(QWebEngineView):
    locationSelected = pyqtSignal(str)
    routingStarted = pyqtSignal(str, str)
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: black;")
        self.current_state = "search"  # "search", "routing", "navigation"
        
        # Create HTML content with embedded navigation GUI
        html_content = self.create_html_content()
        
        # Load the HTML content
        self.setHtml(html_content)
        
        # Connect page loaded signal
        self.loadFinished.connect(self.on_load_finished)
    
    def create_html_content(self):
        """Create HTML content with navigation interface styled like startup"""
        
        # Read the fluid.js file for background effect
        fluid_js_path = os.path.join(os.path.dirname(__file__), "fluid.js")
        try:
            with open(fluid_js_path, 'r') as f:
                fluid_js_content = f.read()
        except FileNotFoundError:
            fluid_js_content = "console.error('fluid.js not found');"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700&family=JetBrains+Mono:wght@100;200;300;400;500&display=swap');
                
                html, body {{
                    overflow: hidden;
                    margin: 0;
                    padding: 0;
                    position: absolute;
                    width: 100%;
                    height: 100%;
                    background: radial-gradient(ellipse at center, #0a0a0a 0%, #000000 70%);
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    font-optical-sizing: auto;
                    --mouse-x: 50%;
                    --mouse-y: 50%;
                }}

                canvas {{
                    width: 100%;
                    height: 100%;
                    display: block;
                    position: absolute;
                    top: 0;
                    left: 0;
                    z-index: 1;
                    opacity: 0.6;
                }}
                
                .overlay {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    z-index: 2;
                    transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                }}
                
                .main-container {{
                    backdrop-filter: blur(20px) saturate(180%);
                    -webkit-backdrop-filter: blur(20px) saturate(180%);
                    background: linear-gradient(135deg, 
                        rgba(255, 255, 255, 0.02) 0%,
                        rgba(255, 255, 255, 0.01) 50%,
                        rgba(0, 0, 0, 0.05) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 32px;
                    padding: 60px 80px;
                    text-align: center;
                    box-shadow: 
                        0 8px 32px rgba(0, 0, 0, 0.4),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1),
                        inset 0 -1px 0 rgba(0, 0, 0, 0.2);
                    position: relative;
                    overflow: hidden;
                    min-width: 500px;
                    max-width: 600px;
                    transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                }}
                
                .main-container::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 1px;
                    background: linear-gradient(90deg, 
                        transparent 0%,
                        rgba(255, 255, 255, 0.2) 50%,
                        transparent 100%);
                }}
                
                .title {{
                    font-size: 48px;
                    font-weight: 100;
                    font-family: 'Inter', sans-serif;
                    color: #ffffff;
                    margin: 0 0 16px 0;
                    letter-spacing: -0.02em;
                    line-height: 0.9;
                    background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    position: relative;
                }}
                
                .title::after {{
                    content: '';
                    position: absolute;
                    bottom: -8px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 40px;
                    height: 1px;
                    background: linear-gradient(90deg, 
                        transparent 0%,
                        rgba(255, 255, 255, 0.4) 50%,
                        transparent 100%);
                }}
                
                .subtitle {{
                    font-size: 12px;
                    font-weight: 300;
                    font-family: 'JetBrains Mono', monospace;
                    color: rgba(255, 255, 255, 0.6);
                    margin: 24px 0 40px 0;
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                    opacity: 0.8;
                }}
                
                .search-container {{
                    width: 100%;
                    margin: 0 auto 32px auto;
                    position: relative;
                }}
                
                .glass-search-bar {{
                    width: 100%;
                    height: 56px;
                    backdrop-filter: blur(12px) saturate(120%);
                    -webkit-backdrop-filter: blur(12px) saturate(120%);
                    background: rgba(255, 255, 255, 0.08);
                    border: 1px solid rgba(255, 255, 255, 0.15);
                    border-radius: 28px;
                    padding: 0 24px;
                    display: flex;
                    align-items: center;
                    box-shadow: 
                        0 8px 32px rgba(0, 0, 0, 0.3),
                        inset 0 1px 0 rgba(255, 255, 255, 0.2);
                    position: relative;
                    overflow: hidden;
                    transition: all 0.3s ease-out;
                }}
                
                .glass-search-bar:hover {{
                    background: rgba(255, 255, 255, 0.1);
                    border-color: rgba(255, 255, 255, 0.2);
                    box-shadow: 
                        0 12px 40px rgba(0, 0, 0, 0.4),
                        inset 0 1px 0 rgba(255, 255, 255, 0.25);
                }}
                
                .glass-search-bar::after {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 1px;
                    background: linear-gradient(90deg, 
                        transparent 0%,
                        rgba(255, 255, 255, 0.3) 50%,
                        transparent 100%);
                    border-radius: 28px 28px 0 0;
                    opacity: 0.6;
                }}
                
                .search-input {{
                    flex: 1;
                    background: transparent;
                    border: none;
                    outline: none;
                    color: rgba(255, 255, 255, 0.95);
                    font-size: 16px;
                    font-weight: 300;
                    font-family: 'Inter', sans-serif;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    letter-spacing: 0.01em;
                }}
                
                .search-input::placeholder {{
                    color: rgba(255, 255, 255, 0.5);
                    font-weight: 300;
                    transition: color 0.3s ease;
                }}
                
                .search-input:focus::placeholder {{
                    color: rgba(255, 255, 255, 0.35);
                }}
                
                .search-icon {{
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 16px;
                    margin-right: 12px;
                    transition: all 0.3s ease;
                    opacity: 0.8;
                }}
                
                .destination-container {{
                    width: 100%;
                    margin: 16px auto 0 auto;
                    position: relative;
                    opacity: 0;
                    transform: translateY(20px);
                    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                    pointer-events: none;
                }}
                
                .destination-container.show {{
                    opacity: 1;
                    transform: translateY(0);
                    pointer-events: auto;
                }}
                
                .glass-destination-bar {{
                    width: 100%;
                    height: 56px;
                    backdrop-filter: blur(12px) saturate(120%);
                    -webkit-backdrop-filter: blur(12px) saturate(120%);
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 28px;
                    padding: 0 24px;
                    display: flex;
                    align-items: center;
                    box-shadow: 
                        0 6px 24px rgba(0, 0, 0, 0.2),
                        inset 0 1px 0 rgba(255, 255, 255, 0.15);
                    position: relative;
                    overflow: hidden;
                    transition: all 0.3s ease-out;
                }}
                
                .glass-destination-bar:hover {{
                    background: rgba(255, 255, 255, 0.08);
                    border-color: rgba(255, 255, 255, 0.15);
                }}
                
                .destination-input {{
                    flex: 1;
                    background: transparent;
                    border: none;
                    outline: none;
                    color: rgba(255, 255, 255, 0.95);
                    font-size: 16px;
                    font-weight: 300;
                    font-family: 'Inter', sans-serif;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    letter-spacing: 0.01em;
                }}
                
                .destination-input::placeholder {{
                    color: rgba(255, 255, 255, 0.4);
                    font-weight: 300;
                }}
                
                .destination-icon {{
                    color: rgba(255, 255, 255, 0.5);
                    font-size: 16px;
                    margin-right: 12px;
                    opacity: 0.7;
                }}
                
                .action-buttons {{
                    display: flex;
                    gap: 16px;
                    margin-top: 32px;
                    opacity: 0;
                    transform: translateY(20px);
                    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) 0.2s;
                    pointer-events: none;
                }}
                
                .action-buttons.show {{
                    opacity: 1;
                    transform: translateY(0);
                    pointer-events: auto;
                }}
                
                .glass-button {{
                    flex: 1;
                    height: 48px;
                    background: linear-gradient(135deg, 
                        rgba(255, 255, 255, 0.1) 0%,
                        rgba(255, 255, 255, 0.05) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.15);
                    border-radius: 24px;
                    color: rgba(255, 255, 255, 0.9);
                    font-family: 'Inter', sans-serif;
                    font-size: 14px;
                    font-weight: 400;
                    cursor: pointer;
                    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
                    backdrop-filter: blur(10px);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    position: relative;
                    overflow: hidden;
                }}
                
                .glass-button:hover {{
                    background: linear-gradient(135deg, 
                        rgba(255, 255, 255, 0.15) 0%,
                        rgba(255, 255, 255, 0.08) 100%);
                    border-color: rgba(255, 255, 255, 0.25);
                    transform: translateY(-2px);
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
                }}
                
                .glass-button:active {{
                    transform: translateY(0);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                }}
                
                .glass-button.primary {{
                    background: linear-gradient(135deg, 
                        rgba(0, 150, 255, 0.2) 0%,
                        rgba(0, 120, 255, 0.1) 100%);
                    border-color: rgba(0, 150, 255, 0.3);
                }}
                
                .glass-button.primary:hover {{
                    background: linear-gradient(135deg, 
                        rgba(0, 150, 255, 0.3) 0%,
                        rgba(0, 120, 255, 0.15) 100%);
                    border-color: rgba(0, 150, 255, 0.4);
                }}
                
                .status-indicator {{
                    position: absolute;
                    bottom: 24px;
                    left: 50%;
                    transform: translateX(-50%);
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 10px;
                    font-weight: 300;
                    color: rgba(255, 255, 255, 0.4);
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                    transition: all 0.3s ease;
                }}
                
                .status-indicator.ready {{
                    color: rgba(0, 255, 150, 0.6);
                }}
                
                .status-indicator.calculating {{
                    color: rgba(255, 200, 0, 0.6);
                }}
                
                .status-indicator.error {{
                    color: rgba(255, 100, 100, 0.6);
                }}
                
                /* Routing state styles */
                .overlay.routing {{
                    justify-content: flex-start;
                    padding-top: 80px;
                }}
                
                .main-container.routing {{
                    max-width: 800px;
                    min-width: 600px;
                    padding: 40px 60px 60px 60px;
                }}
                
                .title.routing {{
                    font-size: 32px;
                    margin-bottom: 8px;
                }}
                
                .subtitle.routing {{
                    margin-bottom: 24px;
                }}
                
                /* Map container for routing view */
                .map-preview {{
                    width: 100%;
                    height: 200px;
                    background: linear-gradient(135deg, 
                        rgba(255, 255, 255, 0.02) 0%,
                        rgba(0, 0, 0, 0.1) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.05);
                    border-radius: 16px;
                    margin: 24px 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: rgba(255, 255, 255, 0.3);
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 12px;
                    opacity: 0;
                    transform: translateY(20px);
                    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) 0.3s;
                }}
                
                .map-preview.show {{
                    opacity: 1;
                    transform: translateY(0);
                }}
                
                .route-info {{
                    display: flex;
                    justify-content: space-between;
                    margin: 16px 0;
                    padding: 16px;
                    background: rgba(255, 255, 255, 0.02);
                    border: 1px solid rgba(255, 255, 255, 0.05);
                    border-radius: 12px;
                    opacity: 0;
                    transform: translateY(20px);
                    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) 0.4s;
                }}
                
                .route-info.show {{
                    opacity: 1;
                    transform: translateY(0);
                }}
                
                .route-stat {{
                    text-align: center;
                    flex: 1;
                }}
                
                .route-stat-label {{
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 10px;
                    color: rgba(255, 255, 255, 0.4);
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    margin-bottom: 4px;
                }}
                
                .route-stat-value {{
                    font-family: 'Inter', sans-serif;
                    font-size: 16px;
                    font-weight: 300;
                    color: rgba(255, 255, 255, 0.9);
                }}
            </style>
        </head>
        <body>
            <canvas></canvas>
            
            <div class="overlay" id="overlay">
                <div class="main-container" id="mainContainer">
                    <div class="title" id="title">navigation</div>
                    <div class="subtitle" id="subtitle">Enter your destination</div>
                    
                    <div class="search-container">
                        <div class="glass-search-bar">
                            <div class="search-icon">📍</div>
                            <input type="text" class="search-input" id="startInput" placeholder="Starting location..." />
                        </div>
                    </div>
                    
                    <div class="destination-container" id="destinationContainer">
                        <div class="glass-destination-bar">
                            <div class="destination-icon">🎯</div>
                            <input type="text" class="destination-input" id="destinationInput" placeholder="Where to?" />
                        </div>
                    </div>
                    
                    <div class="action-buttons" id="actionButtons">
                        <button class="glass-button" onclick="resetSearch()">Reset</button>
                        <button class="glass-button primary" onclick="calculateRoute()">Calculate Route</button>
                    </div>
                    
                    <div class="map-preview" id="mapPreview">
                        Map preview will appear here
                    </div>
                    
                    <div class="route-info" id="routeInfo">
                        <div class="route-stat">
                            <div class="route-stat-label">Distance</div>
                            <div class="route-stat-value" id="routeDistance">--</div>
                        </div>
                        <div class="route-stat">
                            <div class="route-stat-label">Duration</div>
                            <div class="route-stat-value" id="routeDuration">--</div>
                        </div>
                        <div class="route-stat">
                            <div class="route-stat-label">ETA</div>
                            <div class="route-stat-value" id="routeETA">--</div>
                        </div>
                    </div>
                    
                    <div class="status-indicator ready" id="statusIndicator">Ready</div>
                </div>
            </div>
            
            <script>
                {fluid_js_content}
                
                let currentState = 'search';
                let isTransitioning = false;
                
                // DOM elements
                const overlay = document.getElementById('overlay');
                const mainContainer = document.getElementById('mainContainer');
                const title = document.getElementById('title');
                const subtitle = document.getElementById('subtitle');
                const startInput = document.getElementById('startInput');
                const destinationContainer = document.getElementById('destinationContainer');
                const destinationInput = document.getElementById('destinationInput');
                const actionButtons = document.getElementById('actionButtons');
                const mapPreview = document.getElementById('mapPreview');
                const routeInfo = document.getElementById('routeInfo');
                const statusIndicator = document.getElementById('statusIndicator');
                
                // Initialize
                document.addEventListener('DOMContentLoaded', function() {{
                    setTimeout(() => {{
                        startInput.focus();
                    }}, 1000);
                }});
                
                // Handle input changes
                startInput.addEventListener('input', function() {{
                    if (this.value.trim() && currentState === 'search') {{
                        showDestinationInput();
                    }} else if (!this.value.trim() && currentState === 'search') {{
                        hideDestinationInput();
                    }}
                }});
                
                destinationInput.addEventListener('input', function() {{
                    if (this.value.trim() && startInput.value.trim()) {{
                        actionButtons.classList.add('show');
                    }} else {{
                        actionButtons.classList.remove('show');
                    }}
                }});
                
                // Handle enter key
                startInput.addEventListener('keydown', function(e) {{
                    if (e.key === 'Enter' && this.value.trim()) {{
                        showDestinationInput();
                        setTimeout(() => destinationInput.focus(), 300);
                    }}
                }});
                
                destinationInput.addEventListener('keydown', function(e) {{
                    if (e.key === 'Enter' && this.value.trim() && startInput.value.trim()) {{
                        calculateRoute();
                    }}
                }});
                
                // Handle escape key
                document.addEventListener('keydown', function(e) {{
                    if (e.key === 'Escape') {{
                        if (currentState === 'routing') {{
                            resetSearch();
                        }} else if (currentState === 'search') {{
                            hideDestinationInput();
                            startInput.focus();
                        }}
                    }}
                }});
                
                function showDestinationInput() {{
                    if (isTransitioning) return;
                    
                    destinationContainer.classList.add('show');
                    updateStatus('Enter destination', 'ready');
                }}
                
                function hideDestinationInput() {{
                    if (isTransitioning) return;
                    
                    destinationContainer.classList.remove('show');
                    actionButtons.classList.remove('show');
                    destinationInput.value = '';
                    updateStatus('Ready', 'ready');
                }}
                
                function calculateRoute() {{
                    if (isTransitioning) return;
                    if (!startInput.value.trim() || !destinationInput.value.trim()) return;
                    
                    isTransitioning = true;
                    currentState = 'routing';
                    
                    updateStatus('Calculating route...', 'calculating');
                    
                    // Transition to routing view
                    overlay.classList.add('routing');
                    mainContainer.classList.add('routing');
                    title.classList.add('routing');
                    subtitle.classList.add('routing');
                    
                    title.textContent = 'route preview';
                    subtitle.textContent = `${{startInput.value}} → ${{destinationInput.value}}`;
                    
                    // Hide action buttons
                    actionButtons.classList.remove('show');
                    
                    // Show route elements after transition
                    setTimeout(() => {{
                        mapPreview.classList.add('show');
                        routeInfo.classList.add('show');
                        
                        // Simulate route calculation
                        setTimeout(() => {{
                            document.getElementById('routeDistance').textContent = '12.4 km';
                            document.getElementById('routeDuration').textContent = '18 min';
                            document.getElementById('routeETA').textContent = new Date(Date.now() + 18 * 60000).toLocaleTimeString([], {{hour: '2-digit', minute:'2-digit'}});
                            
                            updateStatus('Route calculated', 'ready');
                            isTransitioning = false;
                            
                            // Emit signal to PyQt
                            if (window.pyqtBridge) {{
                                window.pyqtBridge.routingStarted(startInput.value, destinationInput.value);
                            }}
                        }}, 1500);
                    }}, 800);
                }}
                
                function resetSearch() {{
                    if (isTransitioning) return;
                    
                    isTransitioning = true;
                    currentState = 'search';
                    
                    // Hide route elements
                    mapPreview.classList.remove('show');
                    routeInfo.classList.remove('show');
                    
                    // Transition back to search view
                    setTimeout(() => {{
                        overlay.classList.remove('routing');
                        mainContainer.classList.remove('routing');
                        title.classList.remove('routing');
                        subtitle.classList.remove('routing');
                        
                        title.textContent = 'navigation';
                        subtitle.textContent = 'Enter your destination';
                        
                        // Clear inputs
                        startInput.value = '';
                        destinationInput.value = '';
                        
                        // Reset UI state
                        hideDestinationInput();
                        
                        setTimeout(() => {{
                            startInput.focus();
                            updateStatus('Ready', 'ready');
                            isTransitioning = false;
                        }}, 300);
                    }}, 400);
                }}
                
                function updateStatus(text, type) {{
                    statusIndicator.textContent = text;
                    statusIndicator.className = `status-indicator ${{type}}`;
                }}
                
                // Forward mouse events to fluid simulation
                function forwardMouseEvent(event) {{
                    const canvas = document.querySelector('canvas');
                    if (canvas) {{
                        const rect = canvas.getBoundingClientRect();
                        const newEvent = new MouseEvent(event.type, {{
                            clientX: event.clientX,
                            clientY: event.clientY,
                            offsetX: event.clientX - rect.left,
                            offsetY: event.clientY - rect.top,
                            bubbles: true,
                            cancelable: true
                        }});
                        canvas.dispatchEvent(newEvent);
                    }}
                }}
                
                // Handle mouse events
                document.addEventListener('mousemove', forwardMouseEvent);
                document.addEventListener('mousedown', forwardMouseEvent);
                document.addEventListener('mouseup', forwardMouseEvent);
                
                // Handle window resize
                window.addEventListener('resize', function() {{
                    if (typeof resizeCanvas === 'function') {{
                        resizeCanvas();
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def on_load_finished(self, ok):
        """Called when the web page has finished loading"""
        if ok:
            # Inject Python bridge for communication
            self.page().runJavaScript("""
                window.pyqtBridge = {
                    routingStarted: function(start, destination) {
                        // This will be handled by PyQt
                        console.log('Route requested:', start, 'to', destination);
                    }
                };
            """)
            print("Navigation GUI loaded successfully")
        else:
            print("Failed to load Navigation GUI")


class NavigationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: black;")
        
        # Set up 60 FPS cap for smooth animation
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_frame)
        self.fps_timer.start(16)  # 16ms = ~60 FPS
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        # Set window properties
        self.setWindowTitle("Driver - Navigation Interface")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create the navigation web view
        self.web_view = NavigationWebView()
        layout.addWidget(self.web_view)
        
        self.setLayout(layout)
        
        # Connect signals
        self.web_view.locationSelected.connect(self.handle_location_selected)
        self.web_view.routingStarted.connect(self.handle_routing_started)
    
    def update_frame(self):
        """Update frame at 60 FPS"""
        # This method is called 60 times per second
        # Add any frame-based updates here if needed
        self.update()
    
    def handle_location_selected(self, location):
        """Handle when a location is selected"""
        print(f"Location selected: {location}")
        # Here you would typically process the location
    
    def handle_routing_started(self, start_location, destination):
        """Handle when routing is started"""
        print(f"Routing started from '{start_location}' to '{destination}'")
        # Here you would typically start the routing calculation
        # and potentially transition to a different view or emit signals
    
    def closeEvent(self, event):
        """Clean up resources when closing"""
        if hasattr(self, 'fps_timer'):
            self.fps_timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Driver Navigation GUI")
    app.setApplicationVersion("1.0")
    
    # Create and show the navigation window
    nav_window = NavigationWidget()
    nav_window.show()
    
    sys.exit(app.exec())
