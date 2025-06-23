"""
Main simulation loop for AI Cars Learning on Real Roads
Coordinates all system components and handles the main game loop
"""
import pygame
import random
import math

# Import all our modules
from osm_roads import OSMRoadSystem
from camera import Camera
from pathfinding import AStarPathfinder, CheckpointSystem
from car import Car
from evolution import evolve_population
from constants import *


def initialize_pygame():
    """Initialize pygame and create the main screen"""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Cars Learning on Real Roads")
    clock = pygame.time.Clock()
    return screen, clock


def initialize_systems():
    """Initialize all the main systems (roads, pathfinding, checkpoints, camera)"""
    print("Loading road system...")
    road_system = OSMRoadSystem(
        center_lat=DEFAULT_CENTER_LAT, 
        center_lon=DEFAULT_CENTER_LON, 
        radius=DEFAULT_RADIUS
    )

    print("Building pathfinding graph...")
    pathfinder = AStarPathfinder(road_system)

    print("Setting up checkpoint system...")
    checkpoint_system = CheckpointSystem(
        pathfinder, 
        num_checkpoints=NUM_CHECKPOINTS, 
        min_checkpoint_distance=MIN_CHECKPOINT_DISTANCE
    )

    # Setup camera
    camera = Camera(0, 0, WIDTH, HEIGHT)
    road_bounds = road_system.get_road_bounds()
    camera.set_bounds(*road_bounds)
    
    return road_system, pathfinder, checkpoint_system, camera


def create_initial_population(road_system, pathfinder, checkpoint_system):
    """Create the initial population of cars"""
    cars = []
    for i in range(POPULATION_SIZE):
        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
        color = random.choice([RED, GREEN, BLUE])
        car = Car(
            spawn_x, spawn_y, color, use_ai=True, 
            road_system=road_system, 
            pathfinder=pathfinder, 
            checkpoint_system=checkpoint_system
        )
        cars.append(car)
    return cars


def handle_events():
    """Handle pygame events and return whether to continue running"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                return "reset"
    return True


def reset_to_new_location(cars, camera):
    """Reset simulation to a new random location"""
    print("Resetting to new location...")
    
    # Create new road system at random location
    road_system = OSMRoadSystem(
        center_lat=random.uniform(-40, 50), 
        center_lon=random.uniform(-120, 120), 
        radius=DEFAULT_RADIUS
    )
    
    # Update camera bounds
    road_bounds = road_system.get_road_bounds()
    camera.set_bounds(*road_bounds)
    
    # Rebuild pathfinding and checkpoints
    pathfinder = AStarPathfinder(road_system)
    checkpoint_system = CheckpointSystem(
        pathfinder, 
        num_checkpoints=NUM_CHECKPOINTS, 
        min_checkpoint_distance=MIN_CHECKPOINT_DISTANCE
    )

    # Update all cars with new systems
    for car in cars:
        if not car.crashed:
            spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
            car.x, car.y, car.angle = spawn_x, spawn_y, spawn_angle
            car.road_system = road_system
            car.pathfinder = pathfinder
            car.checkpoint_system = checkpoint_system
            car.current_checkpoint_index = 0
            car._update_pathfinding()
    
    return road_system, pathfinder, checkpoint_system


def update_cars(cars, keys):
    """Update all cars and return the best performing car"""
    alive_cars = 0
    best_car = None
    
    for car in cars:
        if not car.crashed:
            car.move(keys)
            
            # Calculate fitness periodically
            if pygame.time.get_ticks() % 500 == 0:  # Every ~500ms
                car.calculate_fitness()
            
            # Track best car
            if best_car is None or car.fitness > best_car.fitness:
                best_car = car
            
            alive_cars += 1
    
    return best_car, alive_cars


def update_raycasts(cars, evolution_timer, best_car):
    """Update raycasts for cars (optimized to not update all cars every frame)"""
    for car in cars:
        if not car.crashed:
            is_best = (car == best_car)
            # Update raycasts more frequently for best car, less for others
            if is_best or evolution_timer % 3 == 0:
                car.update_raycasts()


def draw_everything(screen, road_system, camera, cars, checkpoint_system, best_car):
    """Draw all visual elements"""
    # Clear screen with grass color
    screen.fill((34, 139, 34))
    
    # Draw roads
    road_system.draw_roads(screen, camera.x, camera.y, WIDTH, HEIGHT, camera.zoom)
    
    # Draw checkpoints
    if checkpoint_system and best_car:
        checkpoint_system.draw_checkpoints(
            screen, camera, 
            getattr(best_car, 'current_checkpoint_index', 0)
        )
    
    # Draw cars
    for car in cars:
        if not car.crashed:
            is_best = (car == best_car)
            car.draw(screen, camera, is_best)


def draw_ui(screen, generation, alive_cars, evolution_timer, best_car):
    """Draw the user interface"""
    font = pygame.font.Font(None, 36)
    time_remaining = (EVOLUTION_INTERVAL_SECONDS * FPS - evolution_timer) // FPS
    
    info_text = f"Gen: {generation} | Alive: {alive_cars} | Time: {time_remaining}s"
    if best_car:
        info_text += f" | Best Fitness: {best_car.fitness:.1f}"
    
    text_surface = font.render(info_text, True, WHITE)
    text_rect = text_surface.get_rect()
    text_rect.topleft = (10, 10)
    
    # Draw background for text
    pygame.draw.rect(screen, (0, 0, 0, 128), text_rect.inflate(10, 5))
    screen.blit(text_surface, text_rect)


def main():
    """Main simulation loop"""
    # Initialize everything
    screen, clock = initialize_pygame()
    road_system, pathfinder, checkpoint_system, camera = initialize_systems()
    cars = create_initial_population(road_system, pathfinder, checkpoint_system)
    
    # Simulation state
    evolution_timer = 0
    generation = 1
    evolution_interval = EVOLUTION_INTERVAL_SECONDS * FPS
    running = True
    
    print(f"Starting simulation with {POPULATION_SIZE} cars...")
    print("Controls: C - manual camera, WASD - move camera, Q/E - zoom, R - new location")
    
    # Main loop
    while running:
        # Handle events
        event_result = handle_events()
        if event_result == False:
            running = False
            continue
        elif event_result == "reset":
            road_system, pathfinder, checkpoint_system = reset_to_new_location(cars, camera)
        
        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        # Update cars
        best_car, alive_cars = update_cars(cars, keys)
        
        # Update camera to follow best car
        if best_car and not camera.manual_mode:
            camera.follow_target(best_car.x, best_car.y)
        camera.update(keys)
        
        # Update raycasts (optimized)
        update_raycasts(cars, evolution_timer, best_car)
        
        # Draw everything
        draw_everything(screen, road_system, camera, cars, checkpoint_system, best_car)
        draw_ui(screen, generation, alive_cars, evolution_timer, best_car)
        
        # Check if evolution is needed
        evolution_timer += 1
        if evolution_timer >= evolution_interval or alive_cars == 0:
            cars = evolve_population(cars, POPULATION_SIZE, road_system, pathfinder, checkpoint_system)
            evolution_timer = 0
            generation += 1
            print(f"Generation {generation} started")
        
        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    print("Simulation ended.")


if __name__ == "__main__":
    main()
