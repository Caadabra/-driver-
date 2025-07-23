import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal, QUrl, QTimer, QPropertyAnimation, QEasingCurve, QRect, QPoint
from PyQt6.QtGui import QFont, QPalette, QColor, QPainter, QBrush, QLinearGradient
from PyQt6.QtWebEngineWidgets import QWebEngineView


class MainGUIView(QWebEngineView):
    locationSelected = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: black;")
        self.current_state = "search"  # "search", "routing", "navigation"
        
        # Create HTML content with embedded main GUI
        html_content = self.create_html_content()
        
        # Load the HTML content
        self.setHtml(html_content)
        
        # Connect page loaded signal
        self.loadFinished.connect(self.on_load_finished)
    
    def create_html_content(self):
        """Create HTML content with main GUI interface"""
        
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
                    height: 100vh;
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    font-optical-sizing: auto;
                    --mouse-x: 50%;
                    --mouse-y: 50%;
                    background: #000000;
                }}

                canvas {{
                    display: none;
                }}
                
                /* Background map animation */
                .background-map {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    z-index: 1;
                    opacity: 0.4;
                    filter: grayscale(100%) contrast(1.1) brightness(0.6);
                    transform-origin: center center;
                    transform: scale(1.1);
                    transition: transform 0.3s ease-out;
                }}
                
                .background-map:hover {{
                    transform: scale(1.05);
                }}
                
                .map-overlay {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: radial-gradient(circle at center, 
                        transparent 0%, 
                        transparent 50%, 
                        rgba(0, 0, 0, 0.2) 80%, 
                        rgba(0, 0, 0, 0.6) 100%);
                    z-index: 2;
                    pointer-events: none;
                }}
                
                .main-container {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    z-index: 3;
                    transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                }}
                
                /* Search container styling */
                .search-container {{
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: min(520px, 80vw);
                    height: 65px;
                    z-index: 4;
                    transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                }}
                
                .glass-search-bar {{
                    width: 100%;
                    height: 100%;
                    backdrop-filter: blur(12px) saturate(120%);
                    -webkit-backdrop-filter: blur(12px) saturate(120%);
                    background: rgba(255, 255, 255, 0.08);
                    border: 1px solid rgba(255, 255, 255, 0.15);
                    border-radius: 32px;
                    padding: 0 28px;
                    display: flex;
                    align-items: center;
                    box-shadow: 
                        0 8px 32px rgba(0, 0, 0, 0.3),
                        inset 0 1px 0 rgba(255, 255, 255, 0.2);
                    position: relative;
                    overflow: hidden;
                    transition: all 0.3s ease-out;
                    z-index: 5;
                    contain: layout style paint;
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
                    border-radius: 32px 32px 0 0;
                    opacity: 0.6;
                }}
                
                .glass-search-bar::before {{
                    display: none;
                }}
                
                .search-input {{
                    flex: 1;
                    background: transparent;
                    border: none;
                    outline: none;
                    color: rgba(255, 255, 255, 0.95);
                    font-size: clamp(15px, 2.8vw, 20px);
                    font-weight: 300;
                    font-family: 'Inter', sans-serif;
                    placeholder-color: rgba(255, 255, 255, 0.45);
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    letter-spacing: 0.01em;
                    z-index: 6;
                    position: relative;
                }}
                
                .search-input:focus {{
                    color: rgba(255, 255, 255, 1);
                }}
                
                .search-input.routing {{
                    font-size: clamp(13px, 2vw, 16px);
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
                    font-size: 18px;
                    margin-right: 16px;
                    transition: all 0.3s ease;
                    opacity: 0.8;
                    z-index: 6;
                    position: relative;
                }}
                
                .search-input:focus + .search-icon {{
                    color: rgba(255, 255, 255, 0.8);
                    opacity: 1;
                }}
                
                /* Search suggestions dropdown */
                .suggestions-dropdown {{
                    position: absolute;
                    top: 100%;
                    left: 0;
                    right: 0;
                    margin-top: 8px;
                    backdrop-filter: blur(12px) saturate(120%);
                    background: rgba(255, 255, 255, 0.08);
                    border: 1px solid rgba(255, 255, 255, 0.15);
                    border-radius: 18px;
                    overflow: hidden;
                    opacity: 0;
                    transform: translateY(-4px) scale(0.99);
                    transition: all 0.2s ease-out;
                    pointer-events: none;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    z-index: 10;
                    contain: layout style paint;
                }}
                
                .suggestions-dropdown.show {{
                    opacity: 1;
                    transform: translateY(0) scale(1);
                    pointer-events: auto;
                }}
                
                .suggestion-item {{
                    padding: 16px 20px;
                    color: rgba(255, 255, 255, 0.85);
                    font-size: 15px;
                    font-weight: 300;
                    cursor: pointer;
                    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
                    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
                    position: relative;
                    overflow: hidden;
                    backdrop-filter: blur(10px);
                    display: flex;
                    align-items: center;
                    gap: 14px;
                }}
                
                .suggestion-icon {{
                    font-size: 18px;
                    opacity: 0.8;
                    flex-shrink: 0;
                }}
                
                .suggestion-content {{
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 2px;
                }}
                
                .suggestion-name {{
                    font-size: 15px;
                    font-weight: 400;
                    color: rgba(255, 255, 255, 0.9);
                }}
                
                .suggestion-address {{
                    font-size: 12px;
                    font-weight: 300;
                    color: rgba(255, 255, 255, 0.5);
                    font-family: 'Inter', sans-serif;
                }}
                
                .suggestion-item::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: -100%;
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(90deg, 
                        transparent 0%,
                        rgba(255, 255, 255, 0.1) 50%,
                        transparent 100%);
                    transition: left 0.3s ease;
                }}
                
                .suggestion-item:hover::before {{
                    left: 100%;
                }}
                
                .suggestion-item:last-child {{
                    border-bottom: none;
                }}
                
                .suggestion-item:hover {{
                    background: rgba(255, 255, 255, 0.12);
                    color: rgba(255, 255, 255, 1);
                    backdrop-filter: blur(15px);
                    transform: translateX(4px);
                    border-bottom-color: rgba(255, 255, 255, 0.15);
                }}
                
                .suggestion-item:hover .suggestion-icon {{
                    opacity: 1;
                    transform: scale(1.1);
                }}
                
                .suggestion-item:hover .suggestion-name {{
                    color: rgba(255, 255, 255, 1);
                }}
                
                .suggestion-item:hover .suggestion-address {{
                    color: rgba(255, 255, 255, 0.7);
                }}
                /* Route panel styling */
                .route-panel {{
                    position: absolute;
                    top: 100px;
                    left: 20px;
                    width: min(380px, 26vw);
                    height: calc(100vh - 120px);
                    backdrop-filter: blur(12px) saturate(120%);
                    background: rgba(255, 255, 255, 0.07);
                    border: 1px solid rgba(255, 255, 255, 0.12);
                    border-radius: 20px;
                    padding: 20px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
                    opacity: 0;
                    display: none;
                    pointer-events: none;
                    box-sizing: border-box;
                    overflow: hidden;
                    z-index: 4;
                    contain: layout style paint;
                }}
                
                .route-panel.show {{
                    opacity: 1;
                    display: block;
                    pointer-events: auto;
                }}
                
                .route-header {{
                    margin-bottom: 20px;
                }}
                
                .route-title {{
                    color: rgba(255, 255, 255, 0.9);
                    font-size: 18px;
                    font-weight: 300;
                    margin-bottom: 6px;
                }}
                
                .route-subtitle {{
                    color: rgba(255, 255, 255, 0.5);
                    font-size: 12px;
                    font-family: 'JetBrains Mono', monospace;
                    letter-spacing: 0.05em;
                }}
                
                .route-options {{
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                    max-height: calc(100vh - 300px);
                    overflow-y: auto;
                    scrollbar-width: none;
                    -ms-overflow-style: none;
                }}
                
                .route-options::-webkit-scrollbar {{
                    display: none;
                }}
                
                .route-option {{
                    backdrop-filter: blur(8px);
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 15px;
                    padding: 16px;
                    cursor: pointer;
                    transition: all 0.2s ease-out;
                    position: relative;
                    overflow: hidden;
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    contain: layout style paint;
                }}
                
                .route-icon {{
                    font-size: 20px;
                    opacity: 0.8;
                    flex-shrink: 0;
                    transition: all 0.2s ease;
                }}
                
                .route-content {{
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }}
                
                .route-name {{
                    color: rgba(255, 255, 255, 0.9);
                    font-size: 14px;
                    font-weight: 400;
                    margin: 0;
                }}
                
                .route-details {{
                    display: flex;
                    justify-content: flex-start;
                    align-items: center;
                    gap: 12px;
                }}
                
                .route-time {{
                    color: rgba(100, 200, 255, 0.8);
                    font-size: 12px;
                    font-weight: 500;
                }}
                
                .route-distance {{
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 12px;
                    font-family: 'JetBrains Mono', monospace;
                }}
                
                .route-option::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: -100%;
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(90deg, 
                        transparent 0%,
                        rgba(255, 255, 255, 0.08) 50%,
                        transparent 100%);
                    transition: left 0.4s ease;
                }}
                
                .route-option:hover::before {{
                    left: 100%;
                }}
                
                .route-option.selected {{
                    background: linear-gradient(135deg, rgba(100, 200, 255, 0.12) 0%, rgba(100, 200, 255, 0.04) 100%) !important;
                    border-color: rgba(100, 200, 255, 0.25) !important;
                    transform: translateY(-1px);
                }}
                
                .route-option:hover:not(.selected) {{
                    background: rgba(255, 255, 255, 0.07);
                    border-color: rgba(255, 255, 255, 0.15);
                    transform: translateY(-1px);
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
                }}
                
                .route-option:hover:not(.selected) .route-icon {{
                    opacity: 1;
                    transform: scale(1.05);
                }}
                
                .route-fastest {{
                    border-left: 3px solid rgba(100, 200, 255, 0.6);
                }}
                
                .route-shortest {{
                    border-left: 3px solid rgba(255, 200, 100, 0.6);
                }}
                
                .route-scenic {{
                    border-left: 3px solid rgba(200, 255, 100, 0.6);
                }}
                
                .route-eco {{
                    border-left: 3px solid rgba(100, 255, 150, 0.6);
                }}
                
                .route-badge {{
                    background: linear-gradient(135deg, rgba(100, 200, 255, 0.2), rgba(100, 200, 255, 0.1));
                    border: 1px solid rgba(100, 200, 255, 0.3);
                    border-radius: 12px;
                    padding: 4px 8px;
                    font-size: 10px;
                    font-weight: 500;
                    color: rgba(100, 200, 255, 0.9);
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    flex-shrink: 0;
                }}
                
                /* Status indicator styling */
                .status-indicator {{
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    padding: 8px 16px;
                    background: rgba(0, 0, 0, 0.6);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 20px;
                    backdrop-filter: blur(10px);
                    color: rgba(255, 255, 255, 0.8);
                    font-size: 12px;
                    font-weight: 300;
                    z-index: 5;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                
                .status-dot {{
                    width: 6px;
                    height: 6px;
                    border-radius: 50%;
                    background: rgba(100, 255, 150, 0.8);
                    animation: pulse 2s infinite;
                }}
                
                @keyframes pulse {{
                    0%, 100% {{ opacity: 1; }}
                    50% {{ opacity: 0.5; }}
                }}
                
                /* Map container styling */
                .map-container {{
                    position: absolute;
                    top: 20px;
                    left: calc(min(420px, 28vw) + 20px);
                    width: calc(100vw - min(420px, 28vw) - 40px);
                    height: calc(100vh - 40px);
                    backdrop-filter: blur(8px) saturate(110%);
                    background: rgba(255, 255, 255, 0.04);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 20px;
                    opacity: 0;
                    transform: translateX(30px);
                    transition: all 0.4s ease-out;
                    pointer-events: none;
                    overflow: hidden;
                    box-sizing: border-box;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
                    z-index: 3;
                    contain: layout style paint;
                }}
                
                .map-container::after {{
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
                    z-index: 1;
                }}
                
                .map-container.show {{
                    opacity: 1;
                    transform: translateX(0);
                    pointer-events: auto;
                }}
                
                .map-placeholder {{
                    width: 100%;
                    height: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                    color: rgba(255, 255, 255, 0.4);
                    font-size: 18px;
                    font-weight: 300;
                    z-index: 1;
                    position: relative;
                }}
                
                .welcome-message {{
                    position: absolute;
                    top: 38%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    text-align: center;
                    opacity: 1;
                    transition: all 0.6s ease-out;
                    z-index: 3;
                    pointer-events: none;
                    contain: layout style paint;
                }}
                
                .welcome-message.hide {{
                    opacity: 0;
                    transform: translate(-50%, -60%);
                    pointer-events: none;
                }}
                
                .welcome-title {{
                    font-size: clamp(24px, 4.5vw, 36px);
                    font-weight: 200;
                    color: rgba(255, 255, 255, 0.95);
                    margin-bottom: 10px;
                    letter-spacing: -0.02em;
                    text-shadow: 0 0 40px rgba(255, 255, 255, 0.15);
                    line-height: 1.2;
                }}
                
                .welcome-subtitle {{
                    font-size: clamp(11px, 1.6vw, 13px);
                    font-weight: 300;
                    color: rgba(255, 255, 255, 0.65);
                    font-family: 'Inter', sans-serif;
                    letter-spacing: 0.02em;
                    text-shadow: 0 0 20px rgba(255, 255, 255, 0.08);
                    opacity: 0.9;
                }}
            </style>
        </head>
        <body>
            <canvas></canvas>
            
            <!-- Background map iframe -->
            <div class="background-map">
                <iframe 
                    src="https://www.openstreetmap.org/export/embed.html?bbox=-74.0150%2C40.7000%2C-73.9750%2C40.7600&layer=mapnik"
                    width="100%" 
                    height="100%" 
                    frameborder="0" 
                    style="border: 0; filter: invert(1) hue-rotate(180deg) saturate(0.3) brightness(0.7);">
                </iframe>
            </div>
            
            <!-- Map overlay for vignette effect -->
            <div class="map-overlay"></div>
            
            <div class="main-container search-state" id="mainContainer">
                <!-- Status indicator -->
                <div class="status-indicator" id="statusIndicator">
                    <div class="status-dot"></div>
                    <span>Ready</span>
                </div>
                
                <!-- Welcome message -->
                <div class="welcome-message" id="welcomeMessage">
                    <div class="welcome-title">Where to?</div>
                    <div class="welcome-subtitle">Enter your destination to begin</div>
                </div>
                
                <!-- Search input container -->
                <div class="search-container" id="searchContainer">
                    <div class="glass-search-bar" id="searchBar">
                        <div class="search-icon"></div>
                        <input type="text" class="search-input" id="searchInput" 
                               placeholder="Where do you want to go?" 
                               autocomplete="off">
                    </div>
                    
                    <div class="suggestions-dropdown" id="suggestionsDropdown">
                        <div class="suggestion-item" onclick="selectLocation('Times Square, New York')">
                            <span class="suggestion-icon"></span>
                            <div class="suggestion-content">
                                <div class="suggestion-name">Times Square</div>
                                <div class="suggestion-address">New York, NY</div>
                            </div>
                        </div>
                        <div class="suggestion-item" onclick="selectLocation('Central Park, New York')">
                            <span class="suggestion-icon"></span>
                            <div class="suggestion-content">
                                <div class="suggestion-name">Central Park</div>
                                <div class="suggestion-address">New York, NY</div>
                            </div>
                        </div>
                        <div class="suggestion-item" onclick="selectLocation('Brooklyn Bridge, New York')">
                            <span class="suggestion-icon"></span>
                            <div class="suggestion-content">
                                <div class="suggestion-name">Brooklyn Bridge</div>
                                <div class="suggestion-address">New York, NY</div>
                            </div>
                        </div>
                        <div class="suggestion-item" onclick="selectLocation('Empire State Building, New York')">
                            <span class="suggestion-icon"></span>
                            <div class="suggestion-content">
                                <div class="suggestion-name">Empire State Building</div>
                                <div class="suggestion-address">New York, NY</div>
                            </div>
                        </div>
                        <div class="suggestion-item" onclick="selectLocation('Statue of Liberty, New York')">
                            <span class="suggestion-icon"></span>
                            <div class="suggestion-content">
                                <div class="suggestion-name">Statue of Liberty</div>
                                <div class="suggestion-address">New York, NY</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Route options panel -->
                <div class="route-panel" id="routePanel">
                    <div class="route-header">
                        <div class="route-title">Route Options</div>
                        <div class="route-subtitle">Choose your preferred route</div>
                    </div>
                    
                    <div class="route-options">
                        <div class="route-option route-fastest" onclick="selectRoute('fastest', this)">
                            <div class="route-icon"></div>
                            <div class="route-content">
                                <div class="route-name">Fastest Route</div>
                                <div class="route-details">
                                    <span class="route-time">24 min</span>
                                    <span class="route-distance">12.3 mi</span>
                                </div>
                            </div>
                            <div class="route-badge">Recommended</div>
                        </div>
                        
                        <div class="route-option route-shortest" onclick="selectRoute('shortest', this)">
                            <div class="route-icon"></div>
                            <div class="route-content">
                                <div class="route-name">Shortest Route</div>
                                <div class="route-details">
                                    <span class="route-time">28 min</span>
                                    <span class="route-distance">10.8 mi</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="route-option route-scenic" onclick="selectRoute('scenic', this)">
                            <div class="route-icon"></div>
                            <div class="route-content">
                                <div class="route-name">Scenic Route</div>
                                <div class="route-details">
                                    <span class="route-time">35 min</span>
                                    <span class="route-distance">15.2 mi</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="route-option route-eco" onclick="selectRoute('eco', this)">
                            <div class="route-icon"></div>
                            <div class="route-content">
                                <div class="route-name">Eco-Friendly</div>
                                <div class="route-details">
                                    <span class="route-time">32 min</span>
                                    <span class="route-distance">13.7 mi</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Map display container -->
                <div class="map-container" id="mapContainer">
                    <div class="map-placeholder">
                        <div style="font-size: 24px; margin-bottom: 16px; width: 40px; height: 40px; border: 2px solid rgba(255,255,255,0.3); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px;">MAP</div>
                        <div>3D Interactive Map</div>
                        <div style="font-size: 14px; margin-top: 8px; opacity: 0.6;">Real-time road visualization</div>
                    </div>
                </div>
            </div>
            
            <script>
                // User location and map functions
                let userLocation = {{ lat: 40.7128, lng: -73.9880 }};
                
                function updateBackgroundMap(lat, lng) {{
                    const bbox = {{
                        west: lng - 0.02,
                        south: lat - 0.03,
                        east: lng + 0.02,
                        north: lat + 0.03
                    }};
                    
                    const mapUrl = `https://www.openstreetmap.org/export/embed.html?bbox=${{bbox.west}}%2C${{bbox.south}}%2C${{bbox.east}}%2C${{bbox.north}}&layer=mapnik`;
                    
                    const iframe = document.querySelector('.background-map iframe');
                    if (iframe) {{
                        iframe.src = mapUrl;
                    }}
                }}
                
                // Get user's location
                if (navigator.geolocation) {{
                    navigator.geolocation.getCurrentPosition(
                        function(position) {{
                            userLocation.lat = position.coords.latitude;
                            userLocation.lng = position.coords.longitude;
                            updateBackgroundMap(userLocation.lat, userLocation.lng);
                        }},
                        function(error) {{
                            console.log('Geolocation error:', error);
                            updateBackgroundMap(userLocation.lat, userLocation.lng);
                        }},
                        {{
                            enableHighAccuracy: true,
                            timeout: 10000,
                            maximumAge: 300000
                        }}
                    );
                }} else {{
                    console.log('Geolocation not supported');
                    updateBackgroundMap(userLocation.lat, userLocation.lng);
                }}
                
                // Mouse tracking for parallax glints with performance optimization
                let mouseTimeout;
                document.body.addEventListener("pointermove", (e) => {{
                    if (mouseTimeout) return;
                    
                    mouseTimeout = requestAnimationFrame(() => {{
                        try {{
                            const {{ clientX: x, clientY: y }} = e;
                            const {{ innerWidth: w, innerHeight: h }} = window;
                            
                            if (x >= 0 && x <= w && y >= 0 && y <= h) {{
                                const mouseX = Math.max(0, Math.min(100, (x / w) * 100));
                                const mouseY = Math.max(0, Math.min(100, (y / h) * 100));
                                document.documentElement.style.setProperty('--mouse-x', mouseX + '%');
                                document.documentElement.style.setProperty('--mouse-y', mouseY + '%');
                            }}
                        }} catch (error) {{
                            // Silently ignore errors
                        }}
                        mouseTimeout = null;
                    }});
                }});
                
                // Handle mouse leave
                document.body.addEventListener("mouseleave", () => {{
                    document.documentElement.style.setProperty('--mouse-x', '50%');
                    document.documentElement.style.setProperty('--mouse-y', '50%');
                }});
                
                let currentState = 'search';
                let isTransitioning = false;
                
                // Search input functionality
                const searchInput = document.getElementById('searchInput');
                const suggestionsDropdown = document.getElementById('suggestionsDropdown');
                
                searchInput.addEventListener('input', function() {{
                    const value = this.value.trim();
                    if (value.length > 0) {{
                        suggestionsDropdown.classList.add('show');
                    }} else {{
                        suggestionsDropdown.classList.remove('show');
                    }}
                }});
                
                searchInput.addEventListener('focus', function() {{
                    if (this.value.trim().length > 0) {{
                        suggestionsDropdown.classList.add('show');
                    }}
                }});
                
                document.addEventListener('click', function(e) {{
                    if (!e.target.closest('.search-container')) {{
                        suggestionsDropdown.classList.remove('show');
                    }}
                }});
                
                // Location selection function
                function selectLocation(location) {{
                    if (isTransitioning) return;
                    isTransitioning = true;
                    
                    updateStatus('Calculating routes...', 'calculating');
                    
                    searchInput.value = location;
                    suggestionsDropdown.classList.remove('show');
                    
                    document.getElementById('welcomeMessage').classList.add('hide');
                    
                    setTimeout(() => {{
                        transformToSidePanel();
                    }}, 300);
                }}
                
                // Status update function
                function updateStatus(message, type = 'ready') {{
                    const statusIndicator = document.getElementById('statusIndicator');
                    const statusText = statusIndicator.querySelector('span');
                    const statusDot = statusIndicator.querySelector('.status-dot');
                    
                    statusText.textContent = message;
                    statusDot.className = 'status-dot';
                    
                    switch(type) {{
                        case 'calculating':
                            statusDot.style.background = 'rgba(255, 200, 100, 0.8)';
                            break;
                        case 'routing':
                            statusDot.style.background = 'rgba(100, 200, 255, 0.8)';
                            break;
                        case 'navigating':
                            statusDot.style.background = 'rgba(100, 255, 150, 0.8)';
                            break;
                        default:
                            statusDot.style.background = 'rgba(100, 255, 150, 0.8)';
                    }}
                }}
                
                // Transform search bar into route panel
                function transformToSidePanel() {{
                    const searchContainer = document.getElementById('searchContainer');
                    const searchBar = document.getElementById('searchBar');
                    const searchInput = document.getElementById('searchInput');
                    const routePanel = document.getElementById('routePanel');
                    
                    searchContainer.style.transition = 'all 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                    searchContainer.style.top = '20px';
                    searchContainer.style.left = '20px';
                    searchContainer.style.transform = 'translate(0, 0)';
                            searchContainer.style.width = 'min(350px, 24vw)';
                            searchContainer.style.height = '55px';                    searchBar.style.transition = 'all 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                    searchBar.style.borderRadius = '18px';
                    searchBar.style.padding = '0 18px';
                    searchBar.style.height = '55px';
                    
                    searchInput.style.transition = 'all 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                    searchInput.style.fontSize = 'clamp(13px, 1.8vw, 16px)';
                    
                    setTimeout(() => {{
                        routePanel.style.opacity = '0';
                        routePanel.style.display = 'block';
                        routePanel.style.transform = 'translateX(0) scaleY(0)';
                        routePanel.style.transformOrigin = 'top';
                        routePanel.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
                        
                        setTimeout(() => {{
                            routePanel.style.opacity = '1';
                            routePanel.style.transform = 'translateX(0) scaleY(1)';
                            routePanel.classList.add('show');
                        }}, 50);
                        
                        setTimeout(() => {{
                            const mapContainer = document.getElementById('mapContainer');
                            mapContainer.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
                            mapContainer.classList.add('show');
                            
                            updateStatus('Routes available', 'routing');
                            currentState = 'routing';
                            isTransitioning = false;
                        }}, 400);
                        
                    }}, 800);
                }}
                
                // Route selection function
                function selectRoute(routeType, element) {{
                    if (isTransitioning) return;
                    
                    try {{
                        console.log('Selected route:', routeType);
                        
                        updateStatus('Starting navigation...', 'navigating');
                        
                        // Find the route option element
                        const routeElement = element || (window.event && window.event.target ? window.event.target.closest('.route-option') : null);
                        if (routeElement) {{
                            // Remove selection from all other routes
                            document.querySelectorAll('.route-option').forEach(option => {{
                                option.classList.remove('selected');
                                option.style.background = '';
                                option.style.borderColor = '';
                            }});
                            
                            // Add selection to clicked route
                            routeElement.classList.add('selected');
                            routeElement.style.background = 'linear-gradient(135deg, rgba(100, 200, 255, 0.15) 0%, rgba(100, 200, 255, 0.05) 100%)';
                            routeElement.style.borderColor = 'rgba(100, 200, 255, 0.3)';
                            
                            setTimeout(() => {{
                                updateStatus('Navigation active', 'navigating');
                            }}, 1000);
                        }}
                    }} catch (error) {{
                        console.log('Route selection error:', error);
                        updateStatus('Ready', 'ready');
                    }}
                }}
                
                // Initialize application
                document.addEventListener('DOMContentLoaded', function() {{
                    setTimeout(() => {{
                        searchInput.focus();
                    }}, 1000);
                }});
                
                // Handle escape key to return to search
                document.addEventListener('keydown', function(e) {{
                    if (e.key === 'Escape' && currentState === 'routing') {{
                        resetToSearch();
                    }}
                }});
                
                function resetToSearch() {{
                    if (isTransitioning) return;
                    isTransitioning = true;
                    
                    currentState = 'search';
                    
                    document.getElementById('mapContainer').classList.remove('show');
                    
                    setTimeout(() => {{
                        const routePanel = document.getElementById('routePanel');
                        routePanel.style.transform = 'translateX(0) scaleY(0)';
                        routePanel.style.opacity = '0';
                        
                        setTimeout(() => {{
                            routePanel.classList.remove('show');
                            routePanel.style.display = 'none';
                            
                            const searchContainer = document.getElementById('searchContainer');
                            const searchBar = document.getElementById('searchBar');
                            const searchInput = document.getElementById('searchInput');
                            
                            searchContainer.style.transition = 'all 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                            searchContainer.style.top = '50%';
                            searchContainer.style.left = '50%';
                            searchContainer.style.transform = 'translate(-50%, -50%)';
                            searchContainer.style.width = 'min(520px, 80vw)';
                            searchContainer.style.height = '65px';
                            
                            searchBar.style.transition = 'all 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                            searchBar.style.borderRadius = '32px';
                            searchBar.style.padding = '0 28px';
                            searchBar.style.height = '100%';
                            
                            searchInput.style.transition = 'all 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                            searchInput.style.fontSize = 'clamp(15px, 2.8vw, 20px)';
                            
                            setTimeout(() => {{
                                document.getElementById('welcomeMessage').classList.remove('hide');
                                searchInput.value = '';
                                searchInput.focus();
                                updateStatus('Ready', 'ready');
                                isTransitioning = false;
                            }}, 1200);
                            
                        }}, 300);
                    }}, 200);
                }}
                
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
        """Called when the page finishes loading"""
        if ok:
            print("Main GUI loaded successfully")
        else:
            print("Failed to load Main GUI")


class MainGUIWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: black;")
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        # Set window properties
        self.setWindowTitle("Driver - Main Interface")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create the main web view
        self.web_view = MainGUIView()
        layout.addWidget(self.web_view)
        
        self.setLayout(layout)
        
        # Connect signals
        self.web_view.locationSelected.connect(self.handle_location_selected)
    
    def handle_location_selected(self, location):
        """Handle when a location is selected"""
        print(f"Location selected: {location}")
        # Here you would typically trigger routing calculations
    
    def show_routing_interface(self):
        """Show the routing interface"""
        # This would be called from the startup GUI
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Driver Main GUI")
    app.setApplicationVersion("1.0")
    
    # Create and show the main window
    main_window = MainGUIWidget()
    main_window.show()
    
    sys.exit(app.exec())
