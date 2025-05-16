"""
Terrain and grid generation for the driving simulation
"""
import random
from panda3d.core import LineSegs, NodePath, PNMImage, Texture, Point3

def generate_noise_texture(size=256):
    """Generate a noise texture for terrain"""
    image = PNMImage(size, size, 4)
    # Generate noise: each pixel gets a random shade of gray.
    for x in range(size):
        for y in range(size):
            noise_val = random.random()
            image.set_xel(x, y, noise_val, noise_val, noise_val)
            image.set_alpha(x, y, 1.0)
    tex = Texture()
    tex.load(image)
    return tex

def create_grid(render, size=1000, step=5):
    """Create a reference grid on the ground"""
    lines = LineSegs()
    lines.setColor(0.7, 0.7, 0.7, 1)
    for i in range(-size, size+1, step):
        lines.moveTo(i, -size, 0)
        lines.drawTo(i, size, 0)
        lines.moveTo(-size, i, 0)
        lines.drawTo(size, i, 0)
    grid_node = lines.create()
    grid_np = NodePath(grid_node)
    return grid_np
