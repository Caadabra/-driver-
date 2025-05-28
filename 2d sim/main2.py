import pygame
import random
import math
import torch
import torch.nn as nn
import numpy as np

pygame.init()

WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Driving Game")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

clock = pygame.time.Clock()
FPS = 60

class CarAI(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=3):
        super(CarAI, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def forward(self, x):
        return self.network(x)

class Car:
    def __init__(self, x, y, color, use_ai=False):
        self.x = x
        self.y = y
        self.angle = 180
        self.speed = 0
        self.color = color
        self.width = 12  # Smaller car
        self.height = 20  # Smaller car
        self.checkpoints_passed = []  # Track order of checkpoints passed
        self.current_checkpoint = 0  # Next checkpoint to reach
        self.laps_completed = 0
        self.raycast_length = 150
        self.raycast_angles = [-60, -30, 0, 30, 60]  # Angles relative to car's direction
        self.raycast_distances = []
        self.use_ai = use_ai
        self.fitness = 0
        self.time_alive = 0
        self.distance_traveled = 0
        self.crashed = False
        
        if self.use_ai:
            self.ai = CarAI()            # Initialize with random weights
            self.randomize_weights()

    def draw(self, walls, is_best=False):
        # Create car surface with transparency for non-best cars
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        if is_best:
            car_surface.fill(self.color)
        else:
            # Make other cars 40% transparent
            transparent_color = (*self.color, 100)  # 100/255 ≈ 40% opacity
            car_surface.fill(transparent_color)
        
        rotated_surface = pygame.transform.rotate(car_surface, -self.angle)
        rotated_rect = rotated_surface.get_rect(center=(self.x, self.y))
        screen.blit(rotated_surface, rotated_rect.topleft)
        
        # Only draw raycasts for the best car
        if is_best:
            self.draw_raycasts(walls)
    
    def move(self, keys):
        if self.use_ai:
            self.ai_move()
        else:
            if keys[pygame.K_UP]:
                self.speed += 0.2
            if keys[pygame.K_DOWN]:
                self.speed -= 0.2
            if keys[pygame.K_LEFT]:
                self.angle -= 5
            if keys[pygame.K_RIGHT]:
                self.angle += 5

        # Update position
        prev_x, prev_y = self.x, self.y
        self.x += math.sin(math.radians(self.angle)) * self.speed
        self.y -= math.cos(math.radians(self.angle)) * self.speed

        # Calculate distance traveled for fitness
        if self.use_ai:
            distance = math.sqrt((self.x - prev_x)**2 + (self.y - prev_y)**2)
            self.distance_traveled += distance
            self.time_alive += 1

        self.speed *= 0.95

    def check_collision(self, checkpoints):
        """Check checkpoint collision with sequential progression"""
        if self.current_checkpoint < len(checkpoints):
            checkpoint = checkpoints[self.current_checkpoint]
            car_rect = pygame.Rect(self.x - self.width//2, self.y - self.height//2, 
                                  self.width, self.height)
            
            if car_rect.colliderect(checkpoint):
                self.checkpoints_passed.append(self.current_checkpoint)
                self.current_checkpoint += 1
                  # Complete lap if all checkpoints passed
                if self.current_checkpoint >= len(checkpoints):
                    self.laps_completed += 1
                    self.current_checkpoint = 0  # Reset for next lap

    def cast_ray(self, angle, track_points):
        """Cast a single ray from the car's position at the given angle"""
        ray_angle = math.radians(self.angle + angle)
        ray_x = self.x
        ray_y = self.y
        
        for distance in range(1, self.raycast_length):
            ray_x = self.x + math.sin(ray_angle) * distance
            ray_y = self.y - math.cos(ray_angle) * distance
            
            # Check if ray hits track boundaries
            if not point_in_track(ray_x, ray_y, outer_points, inner_points):
                return distance, ray_x, ray_y
            
            # Check if ray hits screen boundaries
            if ray_x < 0 or ray_x > WIDTH or ray_y < 0 or ray_y > HEIGHT:
                return distance, ray_x, ray_y
        
        return self.raycast_length, ray_x, ray_y
    
    def update_raycasts(self, track_points):
        """Update all raycast distances"""
        self.raycast_distances = []
        for angle in self.raycast_angles:
            distance, _, _ = self.cast_ray(angle, track_points)
            self.raycast_distances.append(distance)
    
    def draw_raycasts(self, track_points):
        """Draw raycast lines"""
        for angle in self.raycast_angles:
            distance, end_x, end_y = self.cast_ray(angle, track_points)
            
            # Draw ray line
            pygame.draw.line(screen, (255, 255, 0), (self.x, self.y), (end_x, end_y), 1)
            
            # Draw hit point
            pygame.draw.circle(screen, (255, 0, 0), (int(end_x), int(end_y)), 2)

    def randomize_weights(self):
        """Initialize neural network with random weights"""
        for param in self.ai.parameters():
            param.data.uniform_(-1, 1)
    
    def ai_move(self):
        """Use AI to control the car"""
        if len(self.raycast_distances) < 5:
            return
        
        # Normalize raycast distances
        normalized_distances = [d / self.raycast_length for d in self.raycast_distances]
        
        # Convert to tensor
        inputs = torch.tensor(normalized_distances, dtype=torch.float32)
        
        # Get AI output
        with torch.no_grad():
            outputs = self.ai(inputs)
        
        # Interpret outputs: [acceleration, left_turn, right_turn]
        acceleration = outputs[0].item()
        left_turn = outputs[1].item()
        right_turn = outputs[2].item()
        
        # Apply acceleration
        self.speed += acceleration * 0.3
        
        # Apply turning (right_turn - left_turn gives net turning)
        turn_strength = (right_turn - left_turn) * 3
        self.angle += turn_strength
    
    def calculate_fitness(self):
        """Calculate fitness based on checkpoint progression, laps, and survival"""
        checkpoint_bonus = len(self.checkpoints_passed) * 50
        lap_bonus = self.laps_completed * 1000
        distance_bonus = self.distance_traveled * 0.1
        time_bonus = self.time_alive * 0.05
        
        # Bonus for being close to next checkpoint
        progress_bonus = self.current_checkpoint * 25
        
        self.fitness = checkpoint_bonus + lap_bonus + distance_bonus + time_bonus + progress_bonus
        return self.fitness
    
    def check_wall_collision(self, walls):
        """Check if car collided with walls"""
        car_rect = pygame.Rect(self.x - self.width//2, self.y - self.height//2, 
                              self.width, self.height)
        
        for wall in walls:
            if car_rect.colliderect(wall):
                self.crashed = True
                return True
        
        # Check screen boundaries
        if (self.x < 0 or self.x > WIDTH or self.y < 0 or self.y > HEIGHT):
            self.crashed = True
            return True
        
        return False

    def clone(self):
        """Create a clone of this car with the same neural network weights"""
        new_car = Car(self.x, self.y, self.color, use_ai=True)
        if self.use_ai:
            # Copy weights from this car to the new car
            new_car.ai.load_state_dict(self.ai.state_dict())
        # Reset tracking variables
        new_car.checkpoints_passed = []
        new_car.current_checkpoint = 0
        new_car.laps_completed = 0
        return new_car
    
    def mutate(self, mutation_rate=0.1, mutation_strength=0.5):
        """Mutate the neural network weights"""
        if not self.use_ai:
            return
        
        with torch.no_grad():
            for param in self.ai.parameters():
                # Apply mutation to some weights
                mutation_mask = torch.rand_like(param) < mutation_rate
                mutation = torch.randn_like(param) * mutation_strength
                param.data += mutation_mask.float() * mutation



def draw_neural_network(car, x_pos, y_pos):
    """Draw a visualization of the neural network for the given car"""
    if not car.use_ai or len(car.raycast_distances) < 5:
        return

    base_x = x_pos
    base_y = y_pos
    y_offset = 200
    x_offset = 0

    # Network dimensions
    layer_spacing = 120
    node_spacing = 30
    node_radius = 8

    # Get network values
    inputs = car.raycast_distances.copy()
    normalized_inputs = [d / car.raycast_length for d in inputs]

    # Calculate outputs
    input_tensor = torch.tensor(normalized_inputs, dtype=torch.float32)
    with torch.no_grad():
        hidden1 = torch.relu(car.ai.network[0](input_tensor))
        hidden2 = torch.relu(car.ai.network[2](hidden1))
        outputs = torch.tanh(car.ai.network[4](hidden2))

    # Convert to lists for visualization
    hidden1_values = hidden1.tolist()[:8]  # Show first 8 hidden nodes
    hidden2_values = hidden2.tolist()[:8]  # Show first 8 hidden nodes
    output_values = outputs.tolist()

    layers = [
        ("Inputs", normalized_inputs, (0, 0, 255)),      # Blue for inputs
        ("Hidden 1", hidden1_values, (128, 128, 128)),   # Gray for hidden
        ("Hidden 2", hidden2_values, (128, 128, 128)),   # Gray for hidden
        ("Outputs", output_values, (255, 0, 0))          # Red for outputs
    ]

    # Draw title
    font = pygame.font.Font(None, 24)
    title_text = font.render("Neural Network (Best Car)", True, WHITE)
    screen.blit(title_text, (base_x + 60, base_y + y_offset + 60))

    # Draw layers
    for layer_idx, (layer_name, values, color) in enumerate(layers):
        layer_x = base_x + layer_idx * layer_spacing
        layer_y_start = base_y + (len(values) - 1) * node_spacing // 2
        
        # Draw layer label
        font_small = pygame.font.Font(None, 18)
        label_text = font_small.render(layer_name, True, WHITE)
        screen.blit(label_text, (layer_x - 20, base_y + 30 + y_offset))
        
        # Draw nodes
        for node_idx, value in enumerate(values):
            node_y = base_y - layer_y_start + node_idx * node_spacing
            
            # Color intensity based on activation value
            intensity = min(255, max(0, int(abs(value) * 255)))
            if value >= 0:
                node_color = (intensity, intensity // 2, 0)  # Orange for positive
            else:
                node_color = (0, intensity // 2, intensity)  # Blue for negative
            
            # Draw node
            pygame.draw.circle(screen, node_color, (layer_x, node_y + y_offset), node_radius)
            pygame.draw.circle(screen, WHITE, (layer_x, node_y + y_offset), node_radius, 2)
            
            # Draw connections to next layer
            if layer_idx < len(layers) - 1:
                next_layer_values = layers[layer_idx + 1][1]
                next_layer_x = base_x + (layer_idx + 1) * layer_spacing
                next_layer_y_start = base_y + (len(next_layer_values) - 1) * node_spacing // 2
                
                for next_node_idx in range(len(next_layer_values)):
                    next_node_y = base_y - next_layer_y_start + next_node_idx * node_spacing
                    
                    # Draw connection line with alpha based on weight strength
                    connection_color = (100, 100, 100)  # Gray connections
                    pygame.draw.line(screen, connection_color, 
                                    (layer_x + node_radius, node_y + y_offset), 
                                    (next_layer_x - node_radius, next_node_y + y_offset), 1)
        
        # Draw value labels for output layer
        if layer_name == "Outputs":
            labels = ["Accel", "Left", "Right"]
            for i, (label, value) in enumerate(zip(labels, values)):
                text = font_small.render(f"{label}: {value:.2f}", True, WHITE)
                screen.blit(text, (layer_x + 70, base_y - layer_y_start + i * node_spacing - 8 + y_offset))

def create_oval_track():
    """Create an oval racing track with smooth curves"""
    track_points = []
    
    # Track parameters
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    outer_radius_x, outer_radius_y = 280, 200
    inner_radius_x, inner_radius_y = 160, 120
    
    # Generate outer boundary points
    outer_points = []
    inner_points = []
    
    for i in range(64):  # More points for smoother curves
        angle = 2 * math.pi * i / 64
        
        # Outer boundary
        outer_x = center_x + outer_radius_x * math.cos(angle)
        outer_y = center_y + outer_radius_y * math.sin(angle)
        outer_points.append((outer_x, outer_y))
        
        # Inner boundary
        inner_x = center_x + inner_radius_x * math.cos(angle)
        inner_y = center_y + inner_radius_y * math.sin(angle)
        inner_points.append((inner_x, inner_y))
    
    return outer_points, inner_points

def point_in_track(x, y, outer_points, inner_points):
    """Check if a point is inside the track (between inner and outer boundaries)"""
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    
    # Calculate distance from center
    dx = x - center_x
    dy = y - center_y
    
    # Use ellipse equation for outer and inner boundaries
    outer_dist = (dx**2 / 280**2) + (dy**2 / 200**2)
    inner_dist = (dx**2 / 160**2) + (dy**2 / 120**2)
    
    # Point is on track if outside inner boundary but inside outer boundary
    return inner_dist > 1 and outer_dist < 1

def check_wall_collision_track(car, outer_points, inner_points):
    """Check if car collided with track boundaries"""
    # Check multiple points around the car for collision
    car_points = [
        (car.x, car.y),  # Center
        (car.x - car.width//2, car.y - car.height//2),  # Top-left
        (car.x + car.width//2, car.y - car.height//2),  # Top-right
        (car.x - car.width//2, car.y + car.height//2),  # Bottom-left
        (car.x + car.width//2, car.y + car.height//2),  # Bottom-right
    ]
    
    for point in car_points:
        if not point_in_track(point[0], point[1], outer_points, inner_points):
            return True
    
    return False

def draw_track(outer_points, inner_points):
    """Draw the oval track"""
    # Draw track surface (grass)
    pygame.draw.ellipse(screen, (34, 139, 34), 
                       (WIDTH//2 - 280, HEIGHT//2 - 200, 560, 400))  # Green grass
    
    # Draw track road (gray)
    pygame.draw.ellipse(screen, (64, 64, 64), 
                       (WIDTH//2 - 280, HEIGHT//2 - 200, 560, 400))
    
    # Draw inner grass area
    pygame.draw.ellipse(screen, (34, 139, 34), 
                       (WIDTH//2 - 160, HEIGHT//2 - 120, 320, 240))
    
    # Draw track boundaries
    pygame.draw.ellipse(screen, BLACK, 
                       (WIDTH//2 - 280, HEIGHT//2 - 200, 560, 400), 3)  # Outer boundary
    pygame.draw.ellipse(screen, BLACK, 
                       (WIDTH//2 - 160, HEIGHT//2 - 120, 320, 240), 3)  # Inner boundary

# Create track
outer_points, inner_points = create_oval_track()

def create_checkpoints():
    """Create checkpoints as bands across the track"""
    checkpoints = []
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    
    # Create 8 checkpoints around the track
    for i in range(8):
        angle = (2 * math.pi * i) / 8
        
        # Calculate checkpoint position on the middle of the track
        middle_radius_x = 220  # Between inner (160) and outer (280)
        middle_radius_y = 160  # Between inner (120) and outer (200)
        
        checkpoint_x = center_x + middle_radius_x * math.cos(angle)
        checkpoint_y = center_y + middle_radius_y * math.sin(angle)
        
        # Create rectangular checkpoint (band across track)
        checkpoint_width = 60
        checkpoint_height = 15
        
        checkpoint = pygame.Rect(checkpoint_x - checkpoint_width//2, 
                                checkpoint_y - checkpoint_height//2,
                                checkpoint_width, checkpoint_height)
        checkpoints.append(checkpoint)
    
    return checkpoints

def evolve_population(cars, population_size=60):
    """Evolve the population using genetic algorithm"""
    # Calculate fitness for all cars
    for car in cars:
        car.calculate_fitness()
    
    # Sort cars by fitness (highest first)
    cars.sort(key=lambda x: x.fitness, reverse=True)
    
    print(f"Generation evolved! Best fitness: {cars[0].fitness:.2f}")
    
    # Keep top 20% as elites
    elite_count = max(1, population_size // 5)
    elites = cars[:elite_count]
    
    new_cars = []
    
    # Add elites to new population
    for elite in elites:
        new_car = elite.clone()
        new_car.x = car_spawn_x
        new_car.y = car_spawn_y
        new_car.angle = 0
        new_car.speed = 0
        new_car.crashed = False
        new_car.fitness = 0
        new_car.time_alive = 0
        new_car.distance_traveled = 0
        new_car.checkpoints_passed = []
        new_car.current_checkpoint = 0
        new_car.laps_completed = 0
        new_cars.append(new_car)
    
    # Fill rest with mutated copies of top performers
    while len(new_cars) < population_size:
        # Select parent from top 50%
        parent_index = random.randint(0, min(len(cars) - 1, population_size // 2 - 1))
        parent = cars[parent_index]
        
        # Create mutated offspring
        offspring = parent.clone()
        offspring.mutate(mutation_rate=0.15, mutation_strength=0.3)
        offspring.x = car_spawn_x
        offspring.y = car_spawn_y
        offspring.angle = 0
        offspring.speed = 0
        offspring.crashed = False
        offspring.fitness = 0
        offspring.time_alive = 0
        offspring.distance_traveled = 0
        offspring.checkpoints_passed = []
        offspring.current_checkpoint = 0
        offspring.laps_completed = 0
        offspring.color = random.choice([RED, GREEN, BLUE])
        new_cars.append(offspring)
    
    return new_cars

checkpoints = create_checkpoints()

# Set car spawn position on the track
car_spawn_x, car_spawn_y = WIDTH // 2 + 220, HEIGHT // 2

# Create AI-controlled cars
cars = [Car(car_spawn_x, car_spawn_y, random.choice([RED, GREEN, BLUE]), use_ai=True) for _ in range(20)]

# Evolution timer
evolution_timer = 0
generation = 1
evolution_interval = 8 * FPS  # 8 seconds at 60 FPS

running = True
while running:
    screen.fill((34, 139, 34))  # Green background (grass)

    # Draw the track
    draw_track(outer_points, inner_points)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    keys = pygame.key.get_pressed()    # Update and draw cars
    alive_cars = 0
    best_car = None
    
    # First, find the best car (highest fitness among alive cars)
    for car in cars:
        if not car.crashed:
            car.calculate_fitness()
            if best_car is None or car.fitness > best_car.fitness:
                best_car = car
    
    # Update and draw all cars
    for car in cars:
        if not car.crashed:
            car.move(keys)
            car.update_raycasts(outer_points + inner_points)  # Use track boundaries for raycasting
            car.check_collision(checkpoints)
            
            # Check collision with track boundaries
            if check_wall_collision_track(car, outer_points, inner_points):
                car.crashed = True
            else:
                # Draw car with transparency (only best car is fully opaque)
                is_best = (car == best_car)
                car.draw(outer_points + inner_points, is_best)
                alive_cars += 1

    # Evolution timer
    evolution_timer += 1

    # Evolve after 8 seconds or when all cars are dead
    if evolution_timer >= evolution_interval or alive_cars == 0:
        cars = evolve_population(cars)
        evolution_timer = 0
        generation += 1
        print(f"Starting generation {generation}")

    # Display generation info
    font = pygame.font.Font(None, 36)
    text = font.render(f"Generation: {generation} | Alive: {alive_cars} | Time: {evolution_timer // FPS}s", True, BLACK)
    screen.blit(text, (10, 10))

    # Draw checkpoints as colorful bands
    for i, checkpoint in enumerate(checkpoints):
        # Highlight the next checkpoint for the best car
        if len(cars) > 0 and not cars[0].crashed:
            color = (255, 255, 0) if i == cars[0].current_checkpoint else (255, 165, 0)
        else:
            color = (255, 165, 0)
        pygame.draw.rect(screen, color, checkpoint)
        
        # Draw checkpoint numbers
        font_small = pygame.font.Font(None, 24)
        text_num = font_small.render(str(i), True, BLACK)
        screen.blit(text_num, (checkpoint.centerx - 6, checkpoint.centery - 8))    # Draw neural network visualization for the best car
    if best_car is not None:
        draw_neural_network(best_car, 650, 100)
    
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
