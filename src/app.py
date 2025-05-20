"""
Main application class for the driving simulation
"""
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import ClockObject, Vec3
import math

from .vehicle import Vehicle
from .terrain import create_grid, generate_noise_texture
from .road import RoadSystem
from .constants import DEFAULT_LAT, DEFAULT_LON, DEFAULT_RADIUS

class DrivingApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()
        
        # Generate noise texture
        self.noise_texture = generate_noise_texture()
        
        # Create the vehicle
        self.vehicle = Vehicle(self.loader, self.render)
        
        # Create the grid floor but hide it by default
        self.grid = create_grid(self.render)
        self.grid.reparentTo(self.render)
        self.grid.setTexture(self.noise_texture, 1)
        
        # Create road system
        self.road_system = RoadSystem(self.render, self.noise_texture, 
                                      self.vehicle.model.getScale().getX())
        self.road_system.load_roads_from_osm(DEFAULT_LAT, DEFAULT_LON, DEFAULT_RADIUS)
        
        # Position car at spawn point if available
        if self.road_system.spawn_point:
            self.vehicle.model.setPos(self.road_system.spawn_point)
        
        # Set up the camera
        self.setup_camera()
        
        # Set up controls
        self.setup_controls()
        
        # Add update task
        self.taskMgr.add(self.update, "update")
        
    def setup_camera(self):
        """Set up the main camera"""
        self.camera.setPos(0, 0, 5)
        self.camera.lookAt(self.vehicle.model)
        
    def setup_controls(self):
        """Set up keyboard controls"""
        self.accept("arrow_up", self.set_key, ["forward", True])
        self.accept("arrow_up-up", self.set_key, ["forward", False])
        self.accept("arrow_down", self.set_key, ["backward", True])
        self.accept("arrow_down-up", self.set_key, ["backward", False])
        self.accept("arrow_left", self.set_key, ["left", True])
        self.accept("arrow_left-up", self.set_key, ["left", False])
        self.accept("arrow_right", self.set_key, ["right", True])
        self.accept("arrow_right-up", self.set_key, ["right", False])
        
    def set_key(self, key, value):
        """Set key state for vehicle controls"""
        self.vehicle.set_control(key, value)
        
    
    def update(self, task):
        """Main update loop"""
        dt = ClockObject.getGlobalClock().getDt()
        # Print which side of the road the vehicle is on
        side = self.road_system.get_road_side(self.vehicle.get_position())

        
        # Update vehicle and check road state
        vehicle_pos = self.vehicle.update(dt, self.road_system.is_on_road(self.vehicle.get_position()))
        
        # Show/hide grid based on road position
        if self.road_system.is_on_road(self.vehicle.get_position()):
            self.grid.hide()
        else:
            self.grid.show()
            
        # Smooth camera update
        desired_offset = Vec3(0, -5, 5)
        desired_cam_pos = vehicle_pos + self.render.getRelativeVector(self.vehicle.get_model(), desired_offset)
        current_cam_pos = self.camera.getPos()
        
        # Smooth factor based on damping and dt. Adjust damping factor as needed.
        damping = 5.0
        t = 1 - math.exp(-damping * dt)
        new_cam_pos = current_cam_pos + (desired_cam_pos - current_cam_pos) * t
        
        self.camera.setPos(new_cam_pos)
        self.camera.lookAt(self.vehicle.get_model())
        
        return task.cont

