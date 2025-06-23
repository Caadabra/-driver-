"""
Camera system for the simulation
Handles camera movement and world-to-screen coordinate conversion
"""
import pygame


class Camera:
    def __init__(self, x, y, screen_width, screen_height):
        self.x = x
        self.y = y
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.zoom = 1.0
        self.manual_mode = False
        self.bounds = None
        
        # Camera movement speeds
        self.move_speed = 5
        self.zoom_speed = 0.1
        
    def set_bounds(self, min_x, min_y, max_x, max_y):
        """Set bounds for camera movement"""
        self.bounds = (min_x, min_y, max_x, max_y)
    
    def follow_target(self, target_x, target_y):
        """Make camera follow a target"""
        if not self.manual_mode:
            # Smooth camera following
            self.x += (target_x - self.x) * 0.1
            self.y += (target_y - self.y) * 0.1
    
    def update(self, keys):
        """Update camera based on keyboard input"""
        # Toggle manual mode
        if keys[pygame.K_c]:
            self.manual_mode = True
        
        # Manual camera control
        if self.manual_mode:
            move_speed = self.move_speed / self.zoom
            if keys[pygame.K_w]:
                self.y -= move_speed
            if keys[pygame.K_s]:
                self.y += move_speed
            if keys[pygame.K_a]:
                self.x -= move_speed
            if keys[pygame.K_d]:
                self.x += move_speed
        
        # Zoom control
        if keys[pygame.K_q]:
            self.zoom = max(0.1, self.zoom - self.zoom_speed)
        if keys[pygame.K_e]:
            self.zoom = min(5.0, self.zoom + self.zoom_speed)
        
        # Apply bounds if set
        if self.bounds:
            min_x, min_y, max_x, max_y = self.bounds
            padding = 100
            self.x = max(min_x - padding, min(max_x + padding, self.x))
            self.y = max(min_y - padding, min(max_y + padding, self.y))
    
    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen coordinates"""
        screen_x = int((world_x - self.x) * self.zoom + self.screen_width // 2)
        screen_y = int((world_y - self.y) * self.zoom + self.screen_height // 2)
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates"""
        world_x = (screen_x - self.screen_width // 2) / self.zoom + self.x
        world_y = (screen_y - self.screen_height // 2) / self.zoom + self.y
        return world_x, world_y
