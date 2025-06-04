import pygame
import random
import math
import torch
import torch.nn as nn
import numpy as np
from osm_roads import OSMRoadSystem
from camera import Camera

pygame.init()

WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Cars Learning on Real Roads")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

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
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

class Car:
    def __init__(self, x, y, color, use_ai=False, road_system=None):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.color = color
        self.width = 12/1.4
        self.height = 20/1.4
        self.raycast_length = 100        
        self.raycast_angles = [-60, -30, 0, 30, 60]
        self.raycast_distances = []
        self.use_ai = use_ai
        self.fitness = 0
        self.time_alive = 0
        self.distance_traveled = 0
        self.crashed = False
        self.road_system = road_system
        self.last_valid_position = (x, y)
        self.off_road_time = 0
        self.max_off_road_time = 0  # Instantly crash when leaving road
        if self.use_ai:
            self.ai = CarAI()
            self.randomize_weights()

    def draw(self, camera, is_best=False):
        # Convert world position to screen position
        screen_x, screen_y = camera.world_to_screen(self.x, self.y)
        
        # Only draw if visible (optimized check)
        if (screen_x < -50 or screen_x > camera.screen_width + 50 or 
            screen_y < -50 or screen_y > camera.screen_height + 50):
            return
          # Create car surface
        car_surface = pygame.Surface((self.width * camera.zoom, self.height * camera.zoom), pygame.SRCALPHA)
        
        if is_best:
            car_surface.fill((255, 0, 0))  # Red for best car
        else:
            car_surface.fill(self.color)
        
        # Rotate and draw
        rotated_surface = pygame.transform.rotate(car_surface, -self.angle)
        rotated_rect = rotated_surface.get_rect(center=(screen_x, screen_y))
        screen.blit(rotated_surface, rotated_rect.topleft)
        
        # Draw raycasts for best car
        if is_best:
            self.draw_raycasts(camera)
    
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

        prev_x, prev_y = self.x, self.y
        self.x += math.sin(math.radians(self.angle)) * self.speed
        self.y -= math.cos(math.radians(self.angle)) * self.speed

        if self.use_ai:
            distance = math.sqrt((self.x - prev_x)**2 + (self.y - prev_y)**2)
            self.distance_traveled += distance
            self.time_alive += 1

        self.speed *= 0.95
        
        # Check if car is on road
        if self.road_system:
            if self.road_system.is_point_on_road(self.x, self.y):
                self.last_valid_position = (self.x, self.y)
                self.off_road_time = 0
            else:
                self.off_road_time += 1
                if self.off_road_time > self.max_off_road_time:
                    self.crashed = True

    def update_raycasts(self):
        """Update raycast distances using optimized road system"""
        if not self.road_system:
            self.raycast_distances = [self.raycast_length] * len(self.raycast_angles)
            return
        
        self.raycast_distances = []
        for angle_offset in self.raycast_angles:
            ray_angle = self.angle + angle_offset
            distance = self.road_system.raycast_to_road_edge(
                self.x, self.y, ray_angle, self.raycast_length
            )
            self.raycast_distances.append(distance)
    def draw_raycasts(self, camera):
        """Draw raycasts relative to camera"""
        for i, angle_offset in enumerate(self.raycast_angles):
            if i < len(self.raycast_distances):
                distance = self.raycast_distances[i]
                ray_angle = math.radians(self.angle + angle_offset)
                
                # Match the car's movement pattern: sin for x, -cos for y
                end_x = self.x + math.sin(ray_angle) * distance
                end_y = self.y - math.cos(ray_angle) * distance
                
                start_screen = camera.world_to_screen(self.x, self.y)
                end_screen = camera.world_to_screen(end_x, end_y)
                pygame.draw.line(screen, YELLOW, start_screen, end_screen, 1)
                pygame.draw.circle(screen, RED, end_screen, 3)

    def randomize_weights(self):
        for param in self.ai.parameters():
            param.data.uniform_(-1, 1)
    
    def ai_move(self):
        if len(self.raycast_distances) < 5:
            return
        
        normalized_distances = [d / self.raycast_length for d in self.raycast_distances]
        inputs = torch.tensor(normalized_distances, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.ai(inputs)
        acceleration = outputs[0].item()
        left_turn = outputs[1].item()
        right_turn = outputs[2].item()
        self.speed += acceleration * 0.3
        turn_strength = (right_turn - left_turn) * 3
        self.angle += turn_strength
    def calculate_fitness(self):
        # Massive penalty for being off road - this encourages staying on roads
        if self.off_road_time > 0:
            # Heavy penalty that grows exponentially with off-road time
            off_road_penalty = (self.off_road_time ** 2) * 2.0  # Exponential penalty
            # Additional penalty for going off road at all
            off_road_base_penalty = 100
        else:
            off_road_penalty = 0
            off_road_base_penalty = 0

        # Reward staying on road
        road_bonus = self.time_alive * 0.5 if self.off_road_time == 0 else 0
        distance_bonus = self.distance_traveled * 0.2 if self.off_road_time < 10 else 0  # Increased distance reward
        time_bonus = self.time_alive * 0.05

        # Reward for maintaining higher speed (average speed)
        avg_speed = self.distance_traveled / self.time_alive if self.time_alive > 0 else 0
        speed_bonus = avg_speed * 10  # Tune this factor as needed

        self.fitness = distance_bonus + time_bonus + road_bonus + speed_bonus - off_road_penalty - off_road_base_penalty
        return self.fitness
    def clone(self):
        new_car = Car(self.x, self.y, self.color, use_ai=True, road_system=self.road_system)
        if self.use_ai:
            new_car.ai.load_state_dict(self.ai.state_dict())
        return new_car
    
    def mutate(self, mutation_rate=0.1, mutation_strength=0.5):
        if not self.use_ai:
            return
        
        with torch.no_grad():
            for param in self.ai.parameters():
                mutation_mask = torch.rand_like(param) < mutation_rate
                mutation = torch.randn_like(param) * mutation_strength
                param.data += mutation_mask.float() * mutation

                # def draw_neural_network(car, camera):
                #     if not car.use_ai or len(car.raycast_distances) < 5:
                #         return
                #
                #     # Fixed position on screen (not affected by camera)
                #     base_x = 50
                #     base_y = 50
                #
                #     layer_spacing = 120
                #     node_spacing = 30
                #     node_radius = 8
                #
                #     inputs = car.raycast_distances.copy()
                #     normalized_inputs = [d / car.raycast_length for d in inputs]
                #
                #     input_tensor = torch.tensor(normalized_inputs, dtype=torch.float32)
                #     with torch.no_grad():
                #         hidden1 = torch.relu(car.ai.network[0](input_tensor))
                #         hidden2 = torch.relu(car.ai.network[2](hidden1))
                #         outputs = torch.tanh(car.ai.network[4](hidden2))
                #
                #     hidden1_values = hidden1.tolist()[:8]
                #     hidden2_values = hidden2.tolist()[:8]
                #     output_values = outputs.tolist()
                #
                #     layers = [
                #         ("Inputs", normalized_inputs, (0, 0, 255)),
                #         ("Hidden 1", hidden1_values, (128, 128, 128)),
                #         ("Hidden 2", hidden2_values, (128, 128, 128)),
                #         ("Outputs", output_values, (255, 0, 0))
                #     ]
                #
                #     font = pygame.font.Font(None, 24)
                #     title_text = font.render("Neural Network (Best Car)", True, WHITE)
                #     screen.blit(title_text, (base_x, base_y))
                #
                #     for layer_idx, (layer_name, values, color) in enumerate(layers):
                #         layer_x = base_x + layer_idx * layer_spacing
                #         layer_y_start = base_y + 50 + (len(values) - 1) * node_spacing // 2
                #         
                #         font_small = pygame.font.Font(None, 18)
                #         label_text = font_small.render(layer_name, True, WHITE)
                #         screen.blit(label_text, (layer_x - 20, base_y + 30))
                #         
                #         for node_idx, value in enumerate(values):
                #             node_y = layer_y_start - (len(values) - 1) * node_spacing // 2 + node_idx * node_spacing
                #             intensity = min(255, max(0, int(abs(value) * 255)))
                #             if value >= 0:
                #                 node_color = (intensity, intensity // 2, 0)
                #             else:
                #                 node_color = (0, intensity // 2, intensity)
                #             pygame.draw.circle(screen, node_color, (layer_x, node_y), node_radius)
                #             pygame.draw.circle(screen, WHITE, (layer_x, node_y), node_radius, 2)
                #             
                #             if layer_idx < len(layers) - 1:
                #                 next_layer_values = layers[layer_idx + 1][1]
                #                 next_layer_x = base_x + (layer_idx + 1) * layer_spacing
                #                 next_layer_y_start = base_y + 50 + (len(next_layer_values) - 1) * node_spacing // 2
                #                 for next_node_idx in range(len(next_layer_values)):
                #                     next_node_y = next_layer_y_start - (len(next_layer_values) - 1) * node_spacing // 2 + next_node_idx * node_spacing
                #                     connection_color = (100, 100, 100)
                #                     pygame.draw.line(screen, connection_color, 
                #                                     (layer_x + node_radius, node_y), 
                #                                     (next_layer_x - node_radius, next_node_y), 1)
                #                                     
                #         if layer_name == "Outputs":
                #             labels = ["Accel", "Left", "Right"]
                #             for i, (label, value) in enumerate(zip(labels, values)):
                #                 text = font_small.render(f"{label}: {value:.2f}", True, WHITE)
                #                 node_y = layer_y_start - (len(values) - 1) * node_spacing // 2 + i * node_spacing
                #                 screen.blit(text, (layer_x + 20, node_y - 8))


def evolve_population(cars, population_size=30, road_system=None):
    for car in cars:
        car.calculate_fitness()
    cars.sort(key=lambda x: x.fitness, reverse=True)
    print(f"Generation evolved! Best fitness: {cars[0].fitness:.2f}")
    
    elite_count = max(1, population_size // 5)
    elites = cars[:elite_count]
    new_cars = []
    
    # Add elites
    for elite in elites:
        new_car = elite.clone()
        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point() if road_system else (0, 0, 0)
        new_car.x = spawn_x
        new_car.y = spawn_y
        new_car.angle = spawn_angle
        new_car.speed = 0
        new_car.crashed = False
        new_car.fitness = 0
        new_car.time_alive = 0
        new_car.distance_traveled = 0
        new_car.off_road_time = 0
        new_cars.append(new_car)
    
    # Create offspring
    while len(new_cars) < population_size:
        parent_index = random.randint(0, min(len(cars) - 1, population_size // 2 - 1))
        parent = cars[parent_index]
        offspring = parent.clone()
        offspring.mutate(mutation_rate=0.15, mutation_strength=0.3)
        
        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point() if road_system else (0, 0, 0)
        offspring.x = spawn_x
        offspring.y = spawn_y
        offspring.angle = spawn_angle
        offspring.speed = 0
        offspring.crashed = False
        offspring.fitness = 0
        offspring.time_alive = 0
        offspring.distance_traveled = 0
        offspring.off_road_time = 0
        offspring.color = random.choice([RED, GREEN, BLUE])
        new_cars.append(offspring)
    
    return new_cars


# Initialize OSM road system (you can change these coordinates)
# Example locations:
# New York: 40.7128, -74.0060
# London: 51.5074, -0.1278
# Auckland: -36.8485, 174.7633


print("Loading road system...")
road_system = OSMRoadSystem(center_lat=-36.8825825, center_lon=174.9143453, radius=1000)

# Initialize camera
camera = Camera(0, 0, WIDTH, HEIGHT)
road_bounds = road_system.get_road_bounds()
camera.set_bounds(*road_bounds)

# Create initial population
population_size = 20  # Reduced from 30 for better performance
cars = []
for i in range(population_size):
    spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
    color = random.choice([RED, GREEN, BLUE])
    car = Car(spawn_x, spawn_y, color, use_ai=True, road_system=road_system)
    cars.append(car)

evolution_timer = 0
generation = 1
evolution_interval = 20 * FPS  # 10 seconds per generation

running = True
while running:
    screen.fill((34, 139, 34))  # Green background
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                # Reset with new random location
                print("Resetting to new location...")
                road_system = OSMRoadSystem(
                    center_lat=random.uniform(-40, 50), 
                    center_lon=random.uniform(-120, 120), 
                    radius=1000
                )
                road_bounds = road_system.get_road_bounds()
                camera.set_bounds(*road_bounds)
                
                # Respawn cars
                for car in cars:
                    if not car.crashed:
                        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
                        car.x, car.y, car.angle = spawn_x, spawn_y, spawn_angle
                        car.road_system = road_system
    keys = pygame.key.get_pressed()
    
    # Find best car and update camera (only calculate fitness occasionally for performance)
    alive_cars = 0
    best_car = None
    for car in cars:
        if not car.crashed:
            # Only calculate fitness every 30 frames for performance
            if evolution_timer % 30 == 0:
                car.calculate_fitness()
            if best_car is None or car.fitness > best_car.fitness:
                best_car = car
            alive_cars += 1
    
    # Update camera to follow best car
    if best_car and not camera.manual_mode:
        camera.follow_target(best_car.x, best_car.y)
    camera.update(keys)
    
    # Draw roads
    road_system.draw_roads(screen, camera.x, camera.y, WIDTH, HEIGHT, camera.zoom)
    
    # Update and draw cars (optimize raycasting frequency)
    for car in cars:
        if not car.crashed:
            car.move(keys)
            # Only update raycasts every few frames for performance (except best car)
            is_best = (car == best_car)
            if is_best or evolution_timer % 3 == 0:  # Best car gets real-time raycasts, others every 3 frames
                car.update_raycasts()
            
            car.draw(camera, is_best)
    
    # Evolution logic
    evolution_timer += 1
    if evolution_timer >= evolution_interval or alive_cars == 0:
        cars = evolve_population(cars, population_size, road_system)
        evolution_timer = 0
        generation += 1
        print(f"Generation {generation} started")
    
    # Draw UI
    font = pygame.font.Font(None, 36)
    info_text = f"Gen: {generation} | Alive: {alive_cars} | Time: {evolution_timer // FPS}s"
    if best_car:
        info_text += f" | Best Fitness: {best_car.fitness:.1f}"
    
    text_surface = font.render(info_text, True, WHITE)
    text_rect = text_surface.get_rect()
    text_rect.topleft = (10, 10)
    
    # Draw background for text
    pygame.draw.rect(screen, (0, 0, 0, 128), text_rect.inflate(10, 5))
    screen.blit(text_surface, text_rect)
    
    # Draw controls
    controls = [
        "Controls:",
        "WASD - Manual camera",
        "Q/E - Zoom in/out", 
        "R - New random location",
        "Arrow keys - Manual car control"
    ]
    font_small = pygame.font.Font(None, 24)
    for i, control in enumerate(controls):
        text = font_small.render(control, True, WHITE)
        text_rect = text.get_rect()
        text_rect.topleft = (WIDTH - 250, 10 + i * 25)
        pygame.draw.rect(screen, (0, 0, 0, 128), text_rect.inflate(5, 2))
        screen.blit(text, text_rect)
    
    # Draw neural network (only every 10 frames for performance)
    if best_car and evolution_timer % 10 == 0:
        # draw_neural_network(best_car, camera)
        pass  # Prevent syntax error due to empty block

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
