"""
Vehicle physics and controls for the driving simulation
"""
from panda3d.core import Vec3

class Vehicle:
    def __init__(self, loader, render):
        self.loader = loader
        self.render = render
        
        # Create the car (a simple cube)
        self.model = self.loader.loadModel("models/box")
        self.model.setScale(1, 2, 0.5)
        self.model.setPos(0, 0, 0)
        self.model.reparentTo(self.render)
        self.model.setColor(1, 0, 0, 1)  # Red color
        self.model.setTextureOff(1)      # Remove any texture
        
        # Physics properties
        self.velocity = 0.0
        self.max_speed = 20.0
        self.acceleration_rate = 5.0
        self.deceleration_rate = 20.0
        self.friction = 5.0
        self.crashed = False
        
        # Controls
        self.controls = {"forward": False, "backward": False, "left": False, "right": False}
        
    def set_control(self, key, value):
        """Set control state (pressed/released)"""
        if key in self.controls:
            self.controls[key] = value
            
    def update(self, dt, is_on_road):
        """Update vehicle physics"""
        # Update velocity based on controls and physics
        if self.controls["forward"]:
            self.velocity += self.acceleration_rate * dt
        elif self.controls["backward"]:
            self.velocity -= self.deceleration_rate * dt
        else:
            if self.velocity > 0:
                self.velocity = max(self.velocity - self.friction * dt, 0)
            elif self.velocity < 0:
                self.velocity = min(self.velocity + self.friction * dt, 0)
        
        # Clamp velocity to limits
        self.velocity = max(min(self.velocity, self.max_speed), -self.max_speed/2)
        
        # Handle steering
        if self.controls["left"]:
            self.model.setH(self.model.getH() + 100 * dt)
        if self.controls["right"]:
            self.model.setH(self.model.getH() - 100 * dt)
        
        # Move vehicle
        self.model.setPos(self.model, 0, self.velocity * dt, 0)
        # Crash detection
        if not is_on_road:
            self.crashed = True
        # Return current position for camera updates
        return self.model.getPos()
        
    def get_position(self):
        """Get the current position of the vehicle"""
        return self.model.getPos()
        
    def get_model(self):
        """Get the vehicle model for camera lookAt"""
        return self.model
    
    @property
    def speed(self):
        return abs(self.velocity)
