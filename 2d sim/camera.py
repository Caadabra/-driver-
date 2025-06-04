"""
Camera system for 2D car simulation with smooth following and bounds checking
"""
import math
import pygame

class Camera:
    def __init__(self, x=0, y=0, screen_width=1920, screen_height=1080):
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Camera settings
        self.follow_speed = 0.05  # How quickly camera follows target (0-1)
        self.zoom = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 2.0
        
        # Bounds (will be set by road system)
        self.min_x = -1000
        self.max_x = 1000
        self.min_y = -1000
        self.max_y = 1000
        
        # Manual control
        self.manual_mode = False
        self.manual_speed = 5
    
    def set_bounds(self, min_x, min_y, max_x, max_y, padding=200):
        """Set camera movement bounds based on road network"""
        self.min_x = min_x - padding
        self.max_x = max_x + padding
        self.min_y = min_y - padding
        self.max_y = max_y + padding
    
    def follow_target(self, target_x, target_y):
        """Set camera to smoothly follow a target"""
        self.target_x = target_x
        self.target_y = target_y
        self.manual_mode = False
    
    def update(self, keys=None):
        """Update camera position"""
        if keys and (keys[pygame.K_w] or keys[pygame.K_a] or keys[pygame.K_s] or keys[pygame.K_d]):
            # Manual camera control
            self.manual_mode = True
            if keys[pygame.K_w]:
                self.target_y -= self.manual_speed / self.zoom
            if keys[pygame.K_s]:
                self.target_y += self.manual_speed / self.zoom
            if keys[pygame.K_a]:
                self.target_x -= self.manual_speed / self.zoom
            if keys[pygame.K_d]:
                self.target_x += self.manual_speed / self.zoom
        
        # Zoom control
        if keys:
            if keys[pygame.K_q]:
                self.zoom = max(self.min_zoom, self.zoom - 0.02)
            if keys[pygame.K_e]:
                self.zoom = min(self.max_zoom, self.zoom + 0.02)
        
        # Smooth camera movement
        if not self.manual_mode:
            # Smooth following
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            self.x += dx * self.follow_speed
            self.y += dy * self.follow_speed
        else:
            # Direct movement for manual control
            self.x = self.target_x
            self.y = self.target_y
        
        # Apply bounds
        effective_width = self.screen_width / (2 * self.zoom)
        effective_height = self.screen_height / (2 * self.zoom)
        
        self.x = max(self.min_x + effective_width, min(self.max_x - effective_width, self.x))
        self.y = max(self.min_y + effective_height, min(self.max_y - effective_height, self.y))
    
    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen coordinates"""
        screen_x = (world_x - self.x) * self.zoom + self.screen_width // 2
        screen_y = (world_y - self.y) * self.zoom + self.screen_height // 2
        return int(screen_x), int(screen_y)
    
    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates"""
        world_x = (screen_x - self.screen_width // 2) / self.zoom + self.x
        world_y = (screen_y - self.screen_height // 2) / self.zoom + self.y
        return world_x, world_y
    
    def is_visible(self, world_x, world_y, margin=50):
        """Check if a world position is visible on screen"""
        screen_x, screen_y = self.world_to_screen(world_x, world_y)
        return (-margin <= screen_x <= self.screen_width + margin and 
                -margin <= screen_y <= self.screen_height + margin)
    
    def get_visible_bounds(self):
        """Get world coordinates of the visible area"""
        # Get corners of screen in world coordinates
        top_left = self.screen_to_world(0, 0)
        bottom_right = self.screen_to_world(self.screen_width, self.screen_height)
        
        return (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
