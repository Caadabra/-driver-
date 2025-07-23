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

                /* Remove the old morphGradient keyframes and replace with procedural animation */

                canvas {{
                    display: none;
                }}
                
                .main-container {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    z-index: 10;
                    transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                }}
                
                /* Search State - Center Glass Bar */
                .search-container {{
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: min(600px, 85vw);
                    height: 80px;
                    z-index: 20;
                    transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                }}
                
                .glass-search-bar {{
                    width: 100%;
                    height: 100%;
                    backdrop-filter: blur(30px) saturate(150%);
                    -webkit-backdrop-filter: blur(30px) saturate(150%);
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 40px;
                    padding: 0 40px;
                    display: flex;
                    align-items: center;
                    box-shadow: 
                        0 20px 60px rgba(0, 0, 0, 0.8),
                        inset 0 1px 0 rgba(255, 255, 255, 0.2),
                        inset 0 -1px 0 rgba(255, 255, 255, 0.05);
                    position: relative;
                    overflow: hidden;
                }}
                
                .glass-search-bar::before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: calc(var(--mouse-x, 50%) - 100px);
                    width: 200px;
                    height: 200%;
                    background: linear-gradient(45deg, 
                        transparent 0%,
                        rgba(255, 255, 255, 0.15) 40%,
                        rgba(255, 255, 255, 0.25) 50%,
                        rgba(255, 255, 255, 0.15) 60%,
                        transparent 100%);
                    transform: skew(-20deg);
                    transition: left 0.3s ease-out;
                    pointer-events: none;
                    opacity: 0;
                    animation: parallaxGlint 4s ease-in-out infinite;
                }}
                
                @keyframes parallaxGlint {{
                    0%, 100% {{
                        opacity: 0;
                        left: calc(var(--mouse-x, 50%) - 200px);
                    }}
                    25% {{
                        opacity: 0.8;
                        left: calc(var(--mouse-x, 50%) - 50px);
                    }}
                    75% {{
                        opacity: 0.8;
                        left: calc(var(--mouse-x, 50%) + 50px);
                    }}
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
                        rgba(255, 255, 255, 0.4) 50%,
                        transparent 100%);
                }}
                
                .search-input {{
                    flex: 1;
                    background: transparent;
                    border: none;
                    outline: none;
                    color: rgba(255, 255, 255, 0.9);
                    font-size: clamp(16px, 3vw, 24px);
                    font-weight: 300;
                    font-family: 'Inter', sans-serif;
                    placeholder-color: rgba(255, 255, 255, 0.4);
                    transition: font-size 0.6s cubic-bezier(0.4, 0, 0.2, 1);
                }}
                
                .search-input.routing {{
                    font-size: clamp(14px, 2vw, 18px);
                }}
                
                .search-input::placeholder {{
                    color: rgba(255, 255, 255, 0.4);
                    font-weight: 200;
                }}
                
                .search-icon {{
                    color: rgba(255, 255, 255, 0.5);
                    font-size: 20px;
                    margin-right: 20px;
                    transition: all 0.3s ease;
                }}
                
                /* Dropdown for search suggestions */
                .suggestions-dropdown {{
                    position: absolute;
                    top: 100%;
                    left: 0;
                    right: 0;
                    margin-top: 10px;
                    backdrop-filter: blur(30px) saturate(150%);
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 20px;
                    overflow: hidden;
                    opacity: 0;
                    transform: translateY(-10px);
                    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                    pointer-events: none;
                    box-shadow: 
                        0 20px 60px rgba(0, 0, 0, 0.8),
                        inset 0 1px 0 rgba(255, 255, 255, 0.15);
                    position: relative;
                }}
                
                .suggestions-dropdown::before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: calc(var(--mouse-x, 50%) - 80px);
                    width: 160px;
                    height: 200%;
                    background: linear-gradient(45deg, 
                        transparent 0%,
                        rgba(255, 255, 255, 0.1) 40%,
                        rgba(255, 255, 255, 0.2) 50%,
                        rgba(255, 255, 255, 0.1) 60%,
                        transparent 100%);
                    transform: skew(-20deg);
                    transition: left 0.3s ease-out;
                    pointer-events: none;
                    opacity: 0;
                    animation: parallaxGlint 5s ease-in-out infinite;
                    animation-delay: 1s;
                }}
                
                .suggestions-dropdown.show {{
                    opacity: 1;
                    transform: translateY(0);
                    pointer-events: auto;
                }}
                
                .suggestion-item {{
                    padding: 16px 30px;
                    color: rgba(255, 255, 255, 0.8);
                    font-size: 16px;
                    font-weight: 300;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                    position: relative;
                    overflow: hidden;
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
                    background: rgba(255, 255, 255, 0.08);
                    color: rgba(255, 255, 255, 1);
                    backdrop-filter: blur(40px);
                }}
                /* Route Panel - Left Side */
                .route-panel {{
                    position: absolute;
                    top: 100px;
                    left: 20px;
                    width: min(380px, 26vw);
                    height: calc(100vh - 120px);
                    backdrop-filter: blur(30px) saturate(150%);
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 20px;
                    padding: 20px;
                    box-shadow: 
                        0 25px 80px rgba(0, 0, 0, 0.9),
                        inset 0 1px 0 rgba(255, 255, 255, 0.15),
                        inset 0 -1px 0 rgba(255, 255, 255, 0.05);
                    opacity: 0;
                    display: none;
                    pointer-events: none;
                    box-sizing: border-box;
                    overflow: hidden;
                    position: relative;
                }}
                
                .route-panel::before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: calc(var(--mouse-x, 50%) - 100px);
                    width: 200px;
                    height: 200%;
                    background: linear-gradient(45deg, 
                        transparent 0%,
                        rgba(255, 255, 255, 0.08) 40%,
                        rgba(255, 255, 255, 0.15) 50%,
                        rgba(255, 255, 255, 0.08) 60%,
                        transparent 100%);
                    transform: skew(-20deg);
                    transition: left 0.3s ease-out;
                    pointer-events: none;
                    opacity: 0;
                    animation: parallaxGlint 6s ease-in-out infinite;
                    animation-delay: 2s;
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
                    backdrop-filter: blur(25px);
                    background: rgba(255, 255, 255, 0.04);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 15px;
                    padding: 15px;
                    cursor: pointer;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    position: relative;
                    overflow: hidden;
                    box-shadow: 
                        0 8px 25px rgba(0, 0, 0, 0.6),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1);
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
                        rgba(255, 255, 255, 0.12) 50%,
                        transparent 100%);
                    transition: left 0.4s ease;
                }}
                
                .route-option::after {{
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
                }}
                
                .route-option:hover::before {{
                    left: 100%;
                }}
                
                .route-option:hover {{
                    background: rgba(255, 255, 255, 0.08);
                    border-color: rgba(255, 255, 255, 0.2);
                    transform: translateY(-2px);
                    box-shadow: 
                        0 15px 40px rgba(0, 0, 0, 0.8),
                        inset 0 2px 0 rgba(255, 255, 255, 0.15);
                }}
                
                .route-name {{
                    color: rgba(255, 255, 255, 0.9);
                    font-size: 14px;
                    font-weight: 400;
                    margin-bottom: 6px;
                }}
                
                .route-details {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
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
                
                /* 3D Map Container - Right Side */
                .map-container {{
                    position: absolute;
                    top: 20px;
                    left: calc(min(400px, 26vw) + 40px);
                    width: calc(100vw - min(400px, 26vw) - 60px);
                    height: calc(100vh - 40px);
                    backdrop-filter: blur(30px) saturate(150%);
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 20px;
                    opacity: 0;
                    transform: translateX(50px);
                    transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                    pointer-events: none;
                    overflow: hidden;
                    box-sizing: border-box;
                    box-shadow: 
                        0 25px 80px rgba(0, 0, 0, 0.9),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1);
                    position: relative;
                }}
                
                .map-container::before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: calc(var(--mouse-x, 50%) - 150px);
                    width: 300px;
                    height: 200%;
                    background: linear-gradient(45deg, 
                        transparent 0%,
                        rgba(255, 255, 255, 0.05) 40%,
                        rgba(255, 255, 255, 0.1) 50%,
                        rgba(255, 255, 255, 0.05) 60%,
                        transparent 100%);
                    transform: skew(-20deg);
                    transition: left 0.3s ease-out;
                    pointer-events: none;
                    opacity: 0;
                    animation: parallaxGlint 8s ease-in-out infinite;
                    animation-delay: 3s;
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
                    z-index: 2;
                    position: relative;
                }}
                
                /* Welcome message */
                .welcome-message {{
                    position: absolute;
                    top: 35%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    text-align: center;
                    opacity: 1;
                    transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                    z-index: 15;
                    animation: welcomeFloat 8s ease-in-out infinite;
                    pointer-events: none;
                }}

                @keyframes welcomeFloat {{
                    0%, 100% {{
                        transform: translate(-50%, -50%) translateY(0px);
                    }}
                    50% {{
                        transform: translate(-50%, -50%) translateY(-10px);
                    }}
                }}
                
                .welcome-message.hide {{
                    opacity: 0;
                    transform: translate(-50%, -60%);
                    pointer-events: none;
                }}
                
                .welcome-title {{
                    font-size: clamp(28px, 5vw, 42px);
                    font-weight: 100;
                    color: rgba(255, 255, 255, 0.9);
                    margin-bottom: 12px;
                    letter-spacing: -0.02em;
                    text-shadow: 0 0 30px rgba(255, 255, 255, 0.2);
                }}
                
                .welcome-subtitle {{
                    font-size: clamp(11px, 1.8vw, 14px);
                    font-weight: 300;
                    color: rgba(255, 255, 255, 0.6);
                    font-family: 'JetBrains Mono', monospace;
                    letter-spacing: 0.05em;
                    text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
                }}
            </style>
        </head>
        <body>
            <canvas></canvas>
            
            <div class="main-container search-state" id="mainContainer">
                <!-- Welcome Message -->
                <div class="welcome-message" id="welcomeMessage">
                    <div class="welcome-title">Where to?</div>
                    <div class="welcome-subtitle">Enter your destination to begin</div>
                </div>
                
                <!-- Search Container -->
                <div class="search-container" id="searchContainer">
                    <div class="glass-search-bar" id="searchBar">
                        <div class="search-icon"></div>
                        <input type="text" class="search-input" id="searchInput" 
                               placeholder="Where do you want to go?" 
                               autocomplete="off">
                    </div>
                    
                    <div class="suggestions-dropdown" id="suggestionsDropdown">
                        <div class="suggestion-item" onclick="selectLocation('Times Square, New York')">Times Square, New York</div>
                        <div class="suggestion-item" onclick="selectLocation('Central Park, New York')">Central Park, New York</div>
                        <div class="suggestion-item" onclick="selectLocation('Brooklyn Bridge, New York')">Brooklyn Bridge, New York</div>
                        <div class="suggestion-item" onclick="selectLocation('Empire State Building, New York')">Empire State Building, New York</div>
                        <div class="suggestion-item" onclick="selectLocation('Statue of Liberty, New York')">Statue of Liberty, New York</div>
                    </div>
                </div>
                
                <!-- Route Panel -->
                <div class="route-panel" id="routePanel">
                    <div class="route-header">
                        <div class="route-title">Route Options</div>
                        <div class="route-subtitle">Choose your preferred route</div>
                    </div>
                    
                    <div class="route-options">
                        <div class="route-option" onclick="selectRoute('fastest')">
                            <div class="route-name">Fastest Route</div>
                            <div class="route-details">
                                <span class="route-time">24 min</span>
                                <span class="route-distance">12.3 mi</span>
                            </div>
                        </div>
                        
                        <div class="route-option" onclick="selectRoute('shortest')">
                            <div class="route-name">Shortest Route</div>
                            <div class="route-details">
                                <span class="route-time">28 min</span>
                                <span class="route-distance">10.8 mi</span>
                            </div>
                        </div>
                        
                        <div class="route-option" onclick="selectRoute('scenic')">
                            <div class="route-name">Scenic Route</div>
                            <div class="route-details">
                                <span class="route-time">35 min</span>
                                <span class="route-distance">15.2 mi</span>
                            </div>
                        </div>
                        
                        <div class="route-option" onclick="selectRoute('eco')">
                            <div class="route-name">Eco-Friendly</div>
                            <div class="route-details">
                                <span class="route-time">32 min</span>
                                <span class="route-distance">13.7 mi</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Map Container -->
                <div class="map-container" id="mapContainer">
                    <div class="map-placeholder">
                        <div style="font-size: 24px; margin-bottom: 16px; width: 40px; height: 40px; border: 2px solid rgba(255,255,255,0.3); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px;">MAP</div>
                        <div>3D Interactive Map</div>
                        <div style="font-size: 14px; margin-top: 8px; opacity: 0.6;">Real-time road visualization</div>
                    </div>
                </div>
            </div>
            
            <script>
                // Mouse tracking for parallax glints
                document.body.addEventListener("pointermove", (e) => {{
                    const {{ clientX: x, clientY: y }} = e;
                    const {{ innerWidth: w, innerHeight: h }} = window;
                    const mouseX = (x / w) * 100;
                    const mouseY = (y / h) * 100;
                    document.documentElement.style.setProperty('--mouse-x', mouseX + '%');
                    document.documentElement.style.setProperty('--mouse-y', mouseY + '%');
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
                
                // Location selection
                function selectLocation(location) {{
                    if (isTransitioning) return;
                    isTransitioning = true;
                    
                    searchInput.value = location;
                    suggestionsDropdown.classList.remove('show');
                    
                    // Hide welcome message
                    document.getElementById('welcomeMessage').classList.add('hide');
                    
                    // Start the transformation animation
                    setTimeout(() => {{
                        transformToSidePanel();
                    }}, 300);
                }}
                
                // Transform search bar into side panel
                function transformToSidePanel() {{
                    const searchContainer = document.getElementById('searchContainer');
                    const searchBar = document.getElementById('searchBar');
                    const searchInput = document.getElementById('searchInput');
                    const routePanel = document.getElementById('routePanel');
                    
                    // First, animate the search bar to the top-left position
                    searchContainer.style.transition = 'all 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                    searchContainer.style.top = '20px';
                    searchContainer.style.left = '20px';
                    searchContainer.style.transform = 'translate(0, 0)';
                            searchContainer.style.width = 'min(380px, 26vw)';
                            searchContainer.style.height = '60px';                    // Morph the search bar styling
                    searchBar.style.transition = 'all 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                    searchBar.style.borderRadius = '20px';
                    searchBar.style.padding = '0 20px';
                    searchBar.style.height = '60px';
                    
                    // Adjust input font size
                    searchInput.style.transition = 'all 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                    searchInput.style.fontSize = 'clamp(14px, 2vw, 18px)';
                    
                    // After the search bar reaches position, start expanding into route panel
                    setTimeout(() => {{
                        // Show the route panel with expansion animation
                        routePanel.style.opacity = '0';
                        routePanel.style.display = 'block';
                        routePanel.style.transform = 'translateX(0) scaleY(0)';
                        routePanel.style.transformOrigin = 'top';
                        routePanel.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
                        
                        // Trigger the expansion
                        setTimeout(() => {{
                            routePanel.style.opacity = '1';
                            routePanel.style.transform = 'translateX(0) scaleY(1)';
                            routePanel.classList.add('show');
                        }}, 50);
                        
                        // Show map container after route panel
                        setTimeout(() => {{
                            const mapContainer = document.getElementById('mapContainer');
                            mapContainer.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
                            mapContainer.classList.add('show');
                            
                            currentState = 'routing';
                            isTransitioning = false;
                        }}, 400);
                        
                    }}, 800);
                }}
                
                // Route selection
                function selectRoute(routeType) {{
                    console.log('Selected route:', routeType);
                    // Here you would typically start navigation
                    
                    // Add visual feedback
                    event.target.style.background = 'linear-gradient(135deg, rgba(100, 200, 255, 0.15) 0%, rgba(100, 200, 255, 0.05) 100%)';
                    event.target.style.borderColor = 'rgba(100, 200, 255, 0.3)';
                    
                    setTimeout(() => {{
                        event.target.style.background = '';
                        event.target.style.borderColor = '';
                    }}, 1000);
                }}
                
                // Initialize
                document.addEventListener('DOMContentLoaded', function() {{
                    // Focus search input after a delay
                    setTimeout(() => {{
                        searchInput.focus();
                    }}, 1000);
                }});
                
                // Handle escape key to go back
                document.addEventListener('keydown', function(e) {{
                    if (e.key === 'Escape' && currentState === 'routing') {{
                        // Transition back to search
                        resetToSearch();
                    }}
                }});
                
                function resetToSearch() {{
                    if (isTransitioning) return;
                    isTransitioning = true;
                    
                    currentState = 'search';
                    
                    // Hide map and route panel first
                    document.getElementById('mapContainer').classList.remove('show');
                    
                    setTimeout(() => {{
                        // Collapse route panel
                        const routePanel = document.getElementById('routePanel');
                        routePanel.style.transform = 'translateX(0) scaleY(0)';
                        routePanel.style.opacity = '0';
                        
                        setTimeout(() => {{
                            routePanel.classList.remove('show');
                            routePanel.style.display = 'none';
                            
                            // Transform search bar back to center
                            const searchContainer = document.getElementById('searchContainer');
                            const searchBar = document.getElementById('searchBar');
                            const searchInput = document.getElementById('searchInput');
                            
                            searchContainer.style.transition = 'all 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                            searchContainer.style.top = '50%';
                            searchContainer.style.left = '50%';
                            searchContainer.style.transform = 'translate(-50%, -50%)';
                            searchContainer.style.width = 'min(600px, 85vw)';
                            searchContainer.style.height = '80px';
                            
                            searchBar.style.transition = 'all 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                            searchBar.style.borderRadius = '40px';
                            searchBar.style.padding = '0 40px';
                            searchBar.style.height = '100%';
                            
                            searchInput.style.transition = 'all 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                            searchInput.style.fontSize = 'clamp(16px, 3vw, 24px)';
                            
                            // Show welcome message and clear input after transformation
                            setTimeout(() => {{
                                document.getElementById('welcomeMessage').classList.remove('hide');
                                searchInput.value = '';
                                searchInput.focus();
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
