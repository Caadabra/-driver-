"""
Evolution system for genetic algorithm
Handles population evolution and fitness-based selection
"""
import random
from car import Car


def evolve_population(cars, population_size=30, road_system=None, pathfinder=None, checkpoint_system=None):
    """Evolve the car population using genetic algorithm principles"""
    # Calculate fitness for all cars
    for car in cars:
        car.calculate_fitness()
    
    # Sort by fitness (highest first)
    cars.sort(key=lambda x: x.fitness, reverse=True)
    print(f"Generation evolved! Best fitness: {cars[0].fitness:.2f}")
    
    # Print checkpoint progress for best car
    if cars[0].checkpoint_system and hasattr(cars[0], 'current_checkpoint_index'):
        print(f"Best car checkpoint progress: {cars[0].current_checkpoint_index}/{cars[0].checkpoint_system.num_checkpoints}")
        if hasattr(cars[0], 'checkpoint_reached_bonus'):
            print(f"Checkpoint bonus: {cars[0].checkpoint_reached_bonus}")
    
    # Keep elite cars (top performers)
    elite_count = max(1, population_size // 5)
    elites = cars[:elite_count]
    new_cars = []
    
    # Add elites to new population (reset their state)
    for elite in elites:
        new_car = elite.clone()
        _reset_car_state(new_car, road_system)
        new_cars.append(new_car)
    
    # Fill rest of population with mutated offspring
    while len(new_cars) < population_size:
        # Select parent based on fitness (better parents more likely)
        parent_index = random.randint(0, min(len(cars) - 1, population_size // 2 - 1))
        parent = cars[parent_index]
        offspring = parent.clone()
        
        # Adaptive mutation based on parent fitness
        base_mutation_rate = 0.12
        base_mutation_strength = 0.25
        
        if parent.fitness > cars[0].fitness * 0.8:
            # Less mutation for good performers
            offspring.mutate(mutation_rate=base_mutation_rate * 0.7, 
                           mutation_strength=base_mutation_strength * 0.8)
        else:
            # More mutation for poor performers
            offspring.mutate(mutation_rate=base_mutation_rate * 1.3, 
                           mutation_strength=base_mutation_strength * 1.2)
        
        _reset_car_state(offspring, road_system)
        offspring.color = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255)])  # RED, GREEN, BLUE
        new_cars.append(offspring)
    
    return new_cars


def _reset_car_state(car, road_system):
    """Reset car to initial state for new generation"""
    spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point() if road_system else (0, 0, 0)
    car.x = spawn_x
    car.y = spawn_y
    car.angle = spawn_angle
    car.speed = 0
    car.crashed = False
    car.fitness = 0
    car.time_alive = 0
    car.distance_traveled = 0
    car.off_road_time = 0
    car.stationary_timer = 0
    car.last_position = (spawn_x, spawn_y)
    car.frames_since_movement_check = 0
    car.current_path = []
    car.path_index = 0
    car.next_waypoint = None
    car.path_following_accuracy = 0
    car.reached_target_bonus = 0
    car.current_checkpoint_index = 0
    car.checkpoint_reached_bonus = 0
    
    # Reset AI state if it has hidden states
    if hasattr(car.ai, 'hidden_state'):
        car.ai.hidden_state = None
        car.ai.memory_sequence = []
