import pygame
import random
import math
import torch
import torch.nn as nn
import numpy as np

pygame.init()

WIDTH, HEIGHT = 800, 600
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
        self.angle = 0
        self.speed = 0
        self.color = color
        self.width = 20
        self.height = 40
        self.checkpoints_passed = set()
        self.raycast_length = 200
        self.raycast_angles = [-60, -30, 0, 30, 60]  # Angles relative to car's direction
        self.raycast_distances = []
        self.use_ai = use_ai
        self.fitness = 0
        self.time_alive = 0
        self.distance_traveled = 0
        self.crashed = False
        
        if self.use_ai:
            self.ai = CarAI()
            # Initialize with random weights
            self.randomize_weights()

    def draw(self, walls):
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        car_surface.fill(self.color)
        rotated_surface = pygame.transform.rotate(car_surface, -self.angle)
        rotated_rect = rotated_surface.get_rect(center=(self.x, self.y))
        screen.blit(rotated_surface, rotated_rect.topleft)
          # Draw raycasts
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
        for i, checkpoint in enumerate(checkpoints):
            if checkpoint.collidepoint(self.x, self.y):
                self.checkpoints_passed.add(i)

    def cast_ray(self, angle, walls):
        """Cast a single ray from the car's position at the given angle"""
        ray_angle = math.radians(self.angle + angle)
        ray_x = self.x
        ray_y = self.y
        
        for distance in range(1, self.raycast_length):
            ray_x = self.x + math.sin(ray_angle) * distance
            ray_y = self.y - math.cos(ray_angle) * distance
            
            # Check if ray hits any wall
            for wall in walls:
                if wall.collidepoint(ray_x, ray_y):
                    return distance, ray_x, ray_y
            
            # Check if ray hits screen boundaries
            if ray_x < 0 or ray_x > WIDTH or ray_y < 0 or ray_y > HEIGHT:
                return distance, ray_x, ray_y
        
        return self.raycast_length, ray_x, ray_y
    
    def update_raycasts(self, walls):
        """Update all raycast distances"""
        self.raycast_distances = []
        for angle in self.raycast_angles:
            distance, _, _ = self.cast_ray(angle, walls)
            self.raycast_distances.append(distance)
    
    def draw_raycasts(self, walls):
        """Draw raycast lines"""
        for angle in self.raycast_angles:
            distance, end_x, end_y = self.cast_ray(angle, walls)
            
            # Draw ray line
            pygame.draw.line(screen, (255, 255, 0), (self.x, self.y), (end_x, end_y), 2)
            
            # Draw hit point
            pygame.draw.circle(screen, (255, 0, 0), (int(end_x), int(end_y)), 3)

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
        """Calculate fitness based on distance traveled and survival time"""
        self.fitness = self.distance_traveled + (self.time_alive * 0.1) + (len(self.checkpoints_passed) * 100)
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

inner_wall = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 100, 200, 200)

# Create outer walls to form a track
outer_walls = [
    pygame.Rect(0, 0, WIDTH, 20),  # Top wall
    pygame.Rect(0, HEIGHT - 20, WIDTH, 20),  # Bottom wall
    pygame.Rect(0, 0, 20, HEIGHT),  # Left wall
    pygame.Rect(WIDTH - 20, 0, 20, HEIGHT)  # Right wall
]

# Create walls list for raycast collision detection
walls = [inner_wall] + outer_walls

def draw_walls():
    pygame.draw.rect(screen, BLACK, inner_wall, width=5)
    for wall in outer_walls:
        pygame.draw.rect(screen, BLACK, wall)

def create_checkpoints():
    """Create checkpoints in a circular pattern around the track"""
    checkpoints = []
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    radius = 250  # Distance from center for checkpoints
    num_checkpoints = 8  # Number of checkpoints around the circle
    
    for i in range(num_checkpoints):
        angle = (2 * math.pi * i) / num_checkpoints
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        # Create small rectangular checkpoints
        checkpoint = pygame.Rect(x - 10, y - 10, 20, 20)
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
        new_car.checkpoints_passed = set()
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
        offspring.checkpoints_passed = set()
        offspring.color = random.choice([RED, GREEN, BLUE])
        new_cars.append(offspring)
        return new_cars

checkpoints = create_checkpoints()

car_spawn_x, car_spawn_y = 400, 180

# Create AI-controlled cars
cars = [Car(car_spawn_x, car_spawn_y, random.choice([RED, GREEN, BLUE]), use_ai=True) for _ in range(20)]

# Evolution timer
evolution_timer = 0
generation = 1
evolution_interval = 15 * FPS  # 15 seconds at 60 FPS

running = True
while running:
    screen.fill(WHITE)

    draw_walls()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    keys = pygame.key.get_pressed()

    # Update and draw cars
    alive_cars = 0
    for car in cars:
        if not car.crashed:
            car.move(keys)
            car.update_raycasts(walls)
            car.check_collision(checkpoints)
            car.check_wall_collision(walls)
            car.draw(walls)
            alive_cars += 1

    # Evolution timer
    evolution_timer += 1
    
    # Evolve after 15 seconds or when all cars are dead
    if evolution_timer >= evolution_interval or alive_cars == 0:
        cars = evolve_population(cars)
        evolution_timer = 0
        generation += 1
        print(f"Starting generation {generation}")

    # Display generation info
    font = pygame.font.Font(None, 36)
    text = font.render(f"Generation: {generation} | Alive: {alive_cars} | Time: {evolution_timer // FPS}s", True, BLACK)
    screen.blit(text, (10, 10))

    for checkpoint in checkpoints:
        pygame.draw.rect(screen, RED, checkpoint)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
