import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QStackedWidget
from PyQt6.QtCore import Qt, pyqtSignal, QUrl
from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6.QtWebEngineWidgets import QWebEngineView


class FluidWebView(QWebEngineView):
    clicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: black;")
        
        # Create HTML content with embedded fluid simulation
        html_content = self.create_html_content()
        
        # Load the HTML content
        self.setHtml(html_content)
        
        # Connect page loaded signal
        self.loadFinished.connect(self.on_load_finished)
    
    def create_html_content(self):
        """Create HTML content with embedded fluid simulation and overlay text"""
        
        # Read the fluid.js file
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
                }}

                canvas {{
                    width: 100%;
                    height: 100%;
                    display: block;
                    position: absolute;
                    top: 0;
                    left: 0;
                    z-index: 1;
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
                    pointer-events: none;
                }}
                
                .glass-container {{
                    backdrop-filter: blur(20px) saturate(180%);
                    -webkit-backdrop-filter: blur(20px) saturate(180%);
                    background: linear-gradient(135deg, 
                        rgba(255, 255, 255, 0.02) 0%,
                        rgba(255, 255, 255, 0.01) 50%,
                        rgba(0, 0, 0, 0.05) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 32px;
                    padding: 80px 120px;
                    text-align: center;
                    box-shadow: 
                        0 8px 32px rgba(0, 0, 0, 0.4),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1),
                        inset 0 -1px 0 rgba(0, 0, 0, 0.2);
                    position: relative;
                    overflow: hidden;
                    min-width: 400px;
                }}
                
                .glass-container::before {{
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
                    font-size: 72px;
                    font-weight: 100;
                    font-family: 'Inter', sans-serif;
                    color: #ffffff;
                    margin: 0 0 16px 0;
                    letter-spacing: -0.04em;
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
                    font-size: 14px;
                    font-weight: 300;
                    font-family: 'JetBrains Mono', monospace;
                    color: rgba(255, 255, 255, 0.6);
                    margin: 24px 0 40px 0;
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                    opacity: 0.8;
                }}
                
                .slider-container {{
                    position: relative;
                    width: 280px;
                    height: 56px;
                    margin: 0 auto;
                    pointer-events: auto;
                }}
                
                .slider-track {{
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(135deg, 
                        rgba(0, 0, 0, 0.2) 0%,
                        rgba(0, 0, 0, 0.1) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.05);
                    border-radius: 28px;
                    position: relative;
                    overflow: hidden;
                    backdrop-filter: blur(10px);
                }}
                
                .slider-track::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 1px;
                    background: linear-gradient(90deg, 
                        transparent 0%,
                        rgba(255, 255, 255, 0.1) 50%,
                        transparent 100%);
                }}
                
                .slider-button {{
                    position: absolute;
                    top: 4px;
                    left: 4px;
                    width: 48px;
                    height: 48px;
                    background: linear-gradient(135deg, 
                        rgba(255, 255, 255, 0.15) 0%,
                        rgba(255, 255, 255, 0.05) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    border-radius: 24px;
                    cursor: grab;
                    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    backdrop-filter: blur(10px);
                    box-shadow: 
                        0 4px 16px rgba(0, 0, 0, 0.3),
                        inset 0 1px 0 rgba(255, 255, 255, 0.2);
                }}
                
                .slider-button:hover {{
                    background: linear-gradient(135deg, 
                        rgba(255, 255, 255, 0.2) 0%,
                        rgba(255, 255, 255, 0.1) 100%);
                    box-shadow: 
                        0 6px 20px rgba(0, 0, 0, 0.4),
                        inset 0 1px 0 rgba(255, 255, 255, 0.3);
                }}
                
                .slider-button:active {{
                    cursor: grabbing;
                    transform: scale(0.98);
                }}
                
                .slider-arrow {{
                    color: rgba(255, 255, 255, 0.8);
                    font-size: 20px;
                    font-weight: 200;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 100%;
                    height: 100%;
                }}
                
                .slider-text {{
                    position: absolute;
                    top: 50%;
                    left: 60px;
                    right: 20px;
                    transform: translateY(-50%);
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 12px;
                    font-weight: 300;
                    color: rgba(255, 255, 255, 0.4);
                    letter-spacing: 0.05em;
                    text-transform: uppercase;
                    text-align: center;
                    pointer-events: none;
                    transition: opacity 0.2s ease;
                }}
                
                .slider-container.sliding .slider-text {{
                    opacity: 0;
                }}
                
                .slider-container.completed .slider-button {{
                    left: calc(100% - 52px);
                    background: linear-gradient(135deg, 
                        rgba(100, 200, 255, 0.3) 0%,
                        rgba(100, 200, 255, 0.1) 100%);
                    border-color: rgba(100, 200, 255, 0.4);
                }}
                
                .slider-container.completed .slider-arrow {{
                    color: rgba(100, 200, 255, 1);
                }}
                
                .arrow {{
                    font-size: 16px;
                    color: rgba(255, 255, 255, 0.3);
                    margin-top: 24px;
                    font-weight: 200;
                    animation: float 3s ease-in-out infinite;
                }}
                
                @keyframes float {{
                    0%, 100% {{
                        transform: translateY(0px);
                        opacity: 0.4;
                    }}
                    50% {{
                        transform: translateY(-8px);
                        opacity: 0.7;
                    }}
                }}
                
                .clickable-area {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    z-index: 3;
                    cursor: default;
                    pointer-events: auto;
                }}
                
                /* Remove custom cursor - show normal cursor */
                
                .tos-container {{
                    position: absolute;
                    bottom: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    z-index: 4;
                    pointer-events: none;
                }}
                
                .tos-text {{
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 10px;
                    font-weight: 300;
                    color: rgba(255, 255, 255, 0.25);
                    text-align: center;
                    line-height: 1.4;
                    letter-spacing: 0.02em;
                    max-width: 600px;
                    padding: 0 20px;
                }}
                
                .tos-text a {{
                    color: rgba(255, 255, 255, 0.4);
                    text-decoration: none;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                    transition: all 0.2s ease;
                    pointer-events: auto;
                }}
                
                .tos-text a:hover {{
                    color: rgba(255, 255, 255, 0.6);
                    border-bottom-color: rgba(255, 255, 255, 0.3);
                }}
            </style>
        </head>
        <body>
            <canvas></canvas>
            
            <div class="overlay">
                <div class="glass-container">
                    <div class="title">driver</div>
                    <div class="subtitle">Built to Think. Born to Drive.</div>
                    <div class="slider-container" id="sliderContainer">
                        <div class="slider-track">
                            <div class="slider-button" id="sliderButton">
                                <div class="slider-arrow">→</div>
                            </div>
                            <div class="slider-text">slide to unlock</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="tos-container">
                <div class="tos-text">
                    By using this application, you agree to our <a href="#" onclick="return false;">Terms of Service</a> and <a href="#" onclick="return false;">Privacy Policy</a>.<br>
                    This software is provided "as is" without warranty. Use at your own risk. © 2025 Driver AI.
                </div>
            </div>
            
            <div class="clickable-area" onclick="handleClick()" 
                 onmousemove="forwardMouseEvent(event)" 
                 onmousedown="forwardMouseEvent(event)" 
                 onmouseup="forwardMouseEvent(event)"
                 style="pointer-events: none;"></div>
            
            <script>
                {fluid_js_content}
                
                function handleClick() {{
                    // Send click event to PyQt
                    if (window.pyqtBridge) {{
                        window.pyqtBridge.clicked();
                    }}
                    console.log('Clicked!');
                }}
                
                // Slider functionality
                let isDragging = false;
                let startX = 0;
                let currentX = 0;
                let sliderButton = null;
                let sliderContainer = null;
                let maxSlide = 0;
                let lastSliderX = 0; // Track last position for fluid interaction
                
                // Function to update fluid based on slider movement
                function updateFluidFromSlider(newX) {{
                    if (typeof pointers !== 'undefined' && pointers.length > 0) {{
                        const pointer = pointers[0];
                        const sliderRect = sliderContainer.getBoundingClientRect();
                        
                        // Convert slider position to canvas coordinates
                        const sliderCenterX = sliderRect.left + newX + 24; // 24 is half button width
                        const sliderCenterY = sliderRect.top + sliderRect.height / 2;
                        
                        // Calculate velocity based on movement
                        const dx = (newX - lastSliderX) * 10; // Scale the movement
                        const dy = 0; // No vertical movement for slider
                        
                        // Update pointer position and movement
                        pointer.x = sliderCenterX;
                        pointer.y = sliderCenterY;
                        pointer.dx = dx;
                        pointer.dy = dy;
                        pointer.moved = true;
                        
                        lastSliderX = newX;
                    }}
                }}
                
                document.addEventListener('DOMContentLoaded', function() {{
                    sliderButton = document.getElementById('sliderButton');
                    sliderContainer = document.getElementById('sliderContainer');
                    
                    if (sliderButton && sliderContainer) {{
                        maxSlide = sliderContainer.offsetWidth - sliderButton.offsetWidth - 8;
                        
                        // Mouse events
                        sliderButton.addEventListener('mousedown', startDrag);
                        document.addEventListener('mousemove', drag);
                        document.addEventListener('mouseup', endDrag);
                        
                        // Touch events
                        sliderButton.addEventListener('touchstart', startDrag);
                        document.addEventListener('touchmove', drag);
                        document.addEventListener('touchend', endDrag);
                    }}
                }});
                
                function startDrag(e) {{
                    isDragging = true;
                    sliderContainer.classList.add('sliding');
                    
                    const clientX = e.type === 'touchstart' ? e.touches[0].clientX : e.clientX;
                    const rect = sliderButton.getBoundingClientRect();
                    startX = clientX - rect.left;
                    
                    // Initialize last position for fluid interaction
                    lastSliderX = parseInt(sliderButton.style.left) || 4;
                    
                    e.preventDefault();
                }}
                
                function drag(e) {{
                    if (!isDragging) return;
                    
                    const clientX = e.type === 'touchmove' ? e.touches[0].clientX : e.clientX;
                    const containerRect = sliderContainer.getBoundingClientRect();
                    currentX = clientX - containerRect.left - startX;
                    
                    // Constrain movement
                    currentX = Math.max(4, Math.min(currentX, maxSlide));
                    
                    sliderButton.style.left = currentX + 'px';
                    
                    // Update fluid simulation
                    updateFluidFromSlider(currentX);
                    
                    e.preventDefault();
                }}
                
                function endDrag(e) {{
                    if (!isDragging) return;
                    
                    isDragging = false;
                    sliderContainer.classList.remove('sliding');
                    
                    // Check if slider reached the end
                    if (currentX >= maxSlide - 20) {{
                        // Complete the slide
                        sliderButton.style.left = maxSlide + 'px';
                        sliderContainer.classList.add('completed');
                        
                        // Trigger app start after a short delay
                        setTimeout(() => {{
                            if (window.pyqtBridge) {{
                                window.pyqtBridge.clicked();
                            }}
                            console.log('App unlocked!');
                        }}, 500);
                    }} else {{
                        // Snap back to start
                        sliderButton.style.left = '4px';
                        currentX = 4;
                    }}
                }}
                
                function forwardMouseEvent(event) {{
                    // Don't forward events if they're on the slider
                    const sliderContainer = document.getElementById('sliderContainer');
                    if (sliderContainer && sliderContainer.contains(event.target)) {{
                        return;
                    }}
                    
                    // Forward mouse events to the canvas
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
                    clicked: function() {
                        // This will be handled by PyQt
                    }
                };
            """)
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        self.clicked.emit()
        super().mousePressEvent(event)


class DriverApp(QApplication):
    def __init__(self):
        super().__init__(sys.argv)
        
        # Create main window
        self.main_window = QWidget()
        self.main_window.setWindowTitle("Driver - Fluid Simulation")
        self.main_window.setStyleSheet("background-color: black;")
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create fluid web view
        self.fluid_view = FluidWebView()
        self.fluid_view.clicked.connect(self.on_start_clicked)
        
        # Add to layout
        layout.addWidget(self.fluid_view)
        self.main_window.setLayout(layout)
        
        # Show fullscreen
        self.main_window.showFullScreen()
    
    def on_start_clicked(self):
        """Handle the start action"""
        print("Starting application...")
        # Here you can add code to transition to your main application
        # For now, we'll just print a message
        
    def run(self):
        """Run the application"""
        return self.exec()


if __name__ == "__main__":
    app = DriverApp()
    sys.exit(app.run())
