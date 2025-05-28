import pygame
import random
import math
import torch
import torch.nn as nn
import numpy as np

pygame.init()

WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("asdjnas")

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
            nn.Tanh()
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
        self.width = 12/1.4
        self.height = 20/1.4
        self.checkpoints_passed = []
        self.current_checkpoint = 0
        self.laps_completed = 0
        self.raycast_length = 150
        self.raycast_angles = [-60, -30, 0, 30, 60]
        self.raycast_distances = []
        self.use_ai = use_ai
        self.fitness = 0
        self.time_alive = 0
        self.distance_traveled = 0
        self.crashed = False
        
        if self.use_ai:
            self.ai = CarAI()
            self.randomize_weights()

    def draw(self, walls, is_best=False):
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        if is_best:
            car_surface.fill(self.color)
        else:
            transparent_color = (*self.color, 100)
            car_surface.fill(transparent_color)
        
        rotated_surface = pygame.transform.rotate(car_surface, -self.angle)
        rotated_rect = rotated_surface.get_rect(center=(self.x, self.y))
        screen.blit(rotated_surface, rotated_rect.topleft)
        
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

        prev_x, prev_y = self.x, self.y
        self.x += math.sin(math.radians(self.angle)) * self.speed
        self.y -= math.cos(math.radians(self.angle)) * self.speed

        if self.use_ai:
            distance = math.sqrt((self.x - prev_x)**2 + (self.y - prev_y)**2)
            self.distance_traveled += distance
            self.time_alive += 1

        self.speed *= 0.95

    def check_collision(self, checkpoints):
        if self.current_checkpoint < len(checkpoints):
            checkpoint = checkpoints[self.current_checkpoint]
            car_rect = pygame.Rect(self.x - self.width//2, self.y - self.height//2, 
                                  self.width, self.height)
            
            if car_rect.colliderect(checkpoint):
                self.checkpoints_passed.append(self.current_checkpoint)
                self.current_checkpoint += 1
                
                if self.current_checkpoint >= len(checkpoints):
                    self.laps_completed += 1
                    self.current_checkpoint = 0
                    print(f"Car completed lap {self.laps_completed}!")

    def cast_ray(self, angle, track_points):
        ray_angle = math.radians(self.angle + angle)
        ray_x = self.x
        ray_y = self.y
        
        for distance in range(1, self.raycast_length):
            ray_x = self.x + math.sin(ray_angle) * distance
            ray_y = self.y - math.cos(ray_angle) * distance
            
            if not point_in_track(ray_x, ray_y, outer_points, inner_points):
                return distance, ray_x, ray_y
            
            if ray_x < 0 or ray_x > WIDTH or ray_y < 0 or ray_y > HEIGHT:
                return distance, ray_x, ray_y
        
        return self.raycast_length, ray_x, ray_y
    
    def update_raycasts(self, track_points):
        self.raycast_distances = []
        for angle in self.raycast_angles:
            distance, _, _ = self.cast_ray(angle, track_points)
            self.raycast_distances.append(distance)
    
    def draw_raycasts(self, track_points):
        for angle in self.raycast_angles:
            distance, end_x, end_y = self.cast_ray(angle, track_points)
            pygame.draw.line(screen, (255, 255, 0), (self.x, self.y), (end_x, end_y), 1)
            pygame.draw.circle(screen, (255, 0, 0), (int(end_x), int(end_y)), 2)

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
        checkpoint_bonus = len(self.checkpoints_passed) * 100
        lap_bonus = self.laps_completed * 1000
        distance_bonus = self.distance_traveled * 0.1
        time_bonus = self.time_alive * 0.05
        progress_bonus = self.current_checkpoint * 75
        self.fitness = checkpoint_bonus + lap_bonus + distance_bonus + time_bonus + progress_bonus
        return self.fitness
    
    def check_wall_collision(self, walls):
        car_rect = pygame.Rect(self.x - self.width//2, self.y - self.height//2, 
                              self.width, self.height)
        
        for wall in walls:
            if car_rect.colliderect(wall):
                self.crashed = True
                return True
        
        if (self.x < 0 or self.x > WIDTH or self.y < 0 or self.y > HEIGHT):
            self.crashed = True
            return True
        
        return False

    def clone(self):
        new_car = Car(self.x, self.y, self.color, use_ai=True)
        if self.use_ai:
            new_car.ai.load_state_dict(self.ai.state_dict())
        new_car.checkpoints_passed = []
        new_car.current_checkpoint = 0
        new_car.laps_completed = 0
        return new_car
    
    def mutate(self, mutation_rate=0.1, mutation_strength=0.5):
        if not self.use_ai:
            return
        
        with torch.no_grad():
            for param in self.ai.parameters():
                mutation_mask = torch.rand_like(param) < mutation_rate
                mutation = torch.randn_like(param) * mutation_strength
                param.data += mutation_mask.float() * mutation


def draw_neural_network(car, x_pos, y_pos):
    if not car.use_ai or len(car.raycast_distances) < 5:
        return

    base_x = x_pos
    base_y = y_pos
    y_offset = 200
    x_offset = 0

    layer_spacing = 120
    node_spacing = 30
    node_radius = 8

    inputs = car.raycast_distances.copy()
    normalized_inputs = [d / car.raycast_length for d in inputs]

    input_tensor = torch.tensor(normalized_inputs, dtype=torch.float32)
    with torch.no_grad():
        hidden1 = torch.relu(car.ai.network[0](input_tensor))
        hidden2 = torch.relu(car.ai.network[2](hidden1))
        outputs = torch.tanh(car.ai.network[4](hidden2))

    hidden1_values = hidden1.tolist()[:8]
    hidden2_values = hidden2.tolist()[:8]
    output_values = outputs.tolist()

    layers = [
        ("Inputs", normalized_inputs, (0, 0, 255)),
        ("Hidden 1", hidden1_values, (128, 128, 128)),
        ("Hidden 2", hidden2_values, (128, 128, 128)),
        ("Outputs", output_values, (255, 0, 0))
    ]

    font = pygame.font.Font(None, 24)
    title_text = font.render("Neural Network (Best Car)", True, WHITE)
    screen.blit(title_text, (base_x + 60, base_y + y_offset + 60))

    for layer_idx, (layer_name, values, color) in enumerate(layers):
        layer_x = base_x + layer_idx * layer_spacing
        layer_y_start = base_y + (len(values) - 1) * node_spacing // 2
        
        font_small = pygame.font.Font(None, 18)
        label_text = font_small.render(layer_name, True, WHITE)
        screen.blit(label_text, (layer_x - 20, base_y + 30 + y_offset))
        
        for node_idx, value in enumerate(values):
            node_y = base_y - layer_y_start + node_idx * node_spacing
            intensity = min(255, max(0, int(abs(value) * 255)))
            if value >= 0:
                node_color = (intensity, intensity // 2, 0)
            else:
                node_color = (0, intensity // 2, intensity)
            pygame.draw.circle(screen, node_color, (layer_x, node_y + y_offset), node_radius)
            pygame.draw.circle(screen, WHITE, (layer_x, node_y + y_offset), node_radius, 2)
            if layer_idx < len(layers) - 1:
                next_layer_values = layers[layer_idx + 1][1]
                next_layer_x = base_x + (layer_idx + 1) * layer_spacing
                next_layer_y_start = base_y + (len(next_layer_values) - 1) * node_spacing // 2
                for next_node_idx in range(len(next_layer_values)):
                    next_node_y = base_y - next_layer_y_start + next_node_idx * node_spacing
                    connection_color = (100, 100, 100)
                    pygame.draw.line(screen, connection_color, 
                                    (layer_x + node_radius, node_y + y_offset), 
                                    (next_layer_x - node_radius, next_node_y + y_offset), 1)
        if layer_name == "Outputs":
            labels = ["Accel", "Left", "Right"]
            for i, (label, value) in enumerate(zip(labels, values)):
                text = font_small.render(f"{label}: {value:.2f}", True, WHITE)
                screen.blit(text, (layer_x + 70, base_y - layer_y_start + i * node_spacing - 8 + y_offset))




def create_oval_track():
    track_points = []
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    outer_radius_x, outer_radius_y = 280, 200
    inner_radius_x, inner_radius_y = 160, 120
    outer_points = []
    inner_points = []
    for i in range(64):
        angle = 2 * math.pi * i / 64
        outer_x = center_x + outer_radius_x * math.cos(angle)
        outer_y = center_y + outer_radius_y * math.sin(angle)
        outer_points.append((outer_x, outer_y))
        inner_x = center_x + inner_radius_x * math.cos(angle)
        inner_y = center_y + inner_radius_y * math.sin(angle)
        inner_points.append((inner_x, inner_y))
    return outer_points, inner_points




def point_in_track(x, y, outer_points, inner_points):
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    dx = x - center_x
    dy = y - center_y
    outer_dist = (dx**2 / 280**2) + (dy**2 / 200**2)
    inner_dist = (dx**2 / 160**2) + (dy**2 / 120**2)
    return inner_dist > 1 and outer_dist < 1





def check_wall_collision_track(car, outer_points, inner_points):
    car_points = [
        (car.x, car.y),
        (car.x - car.width//2, car.y - car.height//2),
        (car.x + car.width//2, car.y - car.height//2),
        (car.x - car.width//2, car.y + car.height//2),
        (car.x + car.width//2, car.y + car.height//2),
    ]
    for point in car_points:
        if not point_in_track(point[0], point[1], outer_points, inner_points):
            return True
    return False




def draw_track(outer_points, inner_points):
    pygame.draw.ellipse(screen, (34, 139, 34), 
                       (WIDTH//2 - 280, HEIGHT//2 - 200, 560, 400))
    pygame.draw.ellipse(screen, (64, 64, 64), 
                       (WIDTH//2 - 280, HEIGHT//2 - 200, 560, 400))
    pygame.draw.ellipse(screen, (34, 139, 34), 
                       (WIDTH//2 - 160, HEIGHT//2 - 120, 320, 240))
    pygame.draw.ellipse(screen, BLACK, 
                       (WIDTH//2 - 280, HEIGHT//2 - 200, 560, 400), 3)
    pygame.draw.ellipse(screen, BLACK, 
                       (WIDTH//2 - 160, HEIGHT//2 - 120, 320, 240), 3)

outer_points, inner_points = create_oval_track()






def create_checkpoints():
    checkpoints = []
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    for i in range(8):
        angle = (2 * math.pi * i) / 8
        middle_radius_x = 220
        middle_radius_y = 160
        checkpoint_x = center_x + middle_radius_x * math.cos(angle)
        checkpoint_y = center_y + middle_radius_y * math.sin(angle)
        checkpoint_width = 60
        checkpoint_height = 15
        checkpoint = pygame.Rect(checkpoint_x - checkpoint_width//2, 
                                checkpoint_y - checkpoint_height//2,
                                checkpoint_width, checkpoint_height)
        checkpoints.append(checkpoint)
    return checkpoints





def evolve_population(cars, population_size=60):
    for car in cars:
        car.calculate_fitness()
    cars.sort(key=lambda x: x.fitness, reverse=True)
    print(f"generation evolved! Best fitness: {cars[0].fitness:.2f}")
    elite_count = max(1, population_size // 5)
    elites = cars[:elite_count]
    new_cars = []
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
    while len(new_cars) < population_size:
        parent_index = random.randint(0, min(len(cars) - 1, population_size // 2 - 1))
        parent = cars[parent_index]
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
car_spawn_x, car_spawn_y = WIDTH // 2 + 220, HEIGHT // 2
cars = [Car(car_spawn_x, car_spawn_y, random.choice([RED, GREEN, BLUE]), use_ai=True) for _ in range(20)]
evolution_timer = 0
generation = 1
evolution_interval = 8 * FPS

running = True
while running:
    screen.fill((34, 139, 34))
    draw_track(outer_points, inner_points)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    alive_cars = 0
    best_car = None
    for car in cars:
        if not car.crashed:
            car.calculate_fitness()
            if best_car is None or car.fitness > best_car.fitness:
                best_car = car
    for car in cars:
        if not car.crashed:
            car.move(keys)
            car.update_raycasts(outer_points + inner_points)
            car.check_collision(checkpoints)
            if check_wall_collision_track(car, outer_points, inner_points):
                car.crashed = True
            else:
                is_best = (car == best_car)
                car.draw(outer_points + inner_points, is_best)
                alive_cars += 1
    evolution_timer += 1
    if evolution_timer >= evolution_interval or alive_cars == 0:
        cars = evolve_population(cars)
        evolution_timer = 0
        generation += 1
        print(f"this is generation {generation}")
    font = pygame.font.Font(None, 36)
    best_checkpoint_info = f" | Best Car Checkpoint: {best_car.current_checkpoint}" if best_car else ""
    text = font.render(f"generation: {generation} alive: {alive_cars} time: {evolution_timer // FPS}s{best_checkpoint_info}", True, BLACK)
    screen.blit(text, (10, 10))
    for i, checkpoint in enumerate(checkpoints):
        if best_car is not None and not best_car.crashed:
            color = (255, 255, 0) if i == best_car.current_checkpoint else (255, 165, 0)
        else:
            color = (255, 165, 0)
        pygame.draw.rect(screen, color, checkpoint)
        font_small = pygame.font.Font(None, 24)
        text_num = font_small.render(str(i), True, BLACK)
        screen.blit(text_num, (checkpoint.centerx - 6, checkpoint.centery - 8))
    if best_car is not None:
        draw_neural_network(best_car, 650, 100)
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
