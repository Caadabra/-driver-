"""


███████ ██    ██  ██████  ██      ██    ██ ████████ ██  ██████  ███    ██    ██████  ██    ██ 
██      ██    ██ ██    ██ ██      ██    ██    ██    ██ ██    ██ ████   ██    ██   ██  ██  ██  
█████   ██    ██ ██    ██ ██      ██    ██    ██    ██ ██    ██ ██ ██  ██    ██████    ████   
██       ██  ██  ██    ██ ██      ██    ██    ██    ██ ██    ██ ██  ██ ██    ██         ██    
███████   ████    ██████  ███████  ██████     ██    ██  ██████  ██   ████ ██ ██         ██    
                                                                                              
                                                                                              

Evolution and population management for the AI driving simulation.
Contains functions for genetic algorithm operations, population saving/loading, and car creation.
"""

import os
import pickle
import random
import math
import torch

from car import Car
from constants import *


def set_raycast_config(num_rays, spread_degrees=180):
    """
    Raycast configuration
    """
    global NUM_RAYCASTS, RAYCAST_SPREAD
    NUM_RAYCASTS = num_rays
    RAYCAST_SPREAD = spread_degrees
    print(f"Raycast config: {NUM_RAYCASTS} rays with {RAYCAST_SPREAD}° spread")


def save_population(cars, generation, filename="population.pkl"):
    """Save the current population to a file"""
    try:
        # Create trained_models directory if it doesn't exist
        os.makedirs("trained_models", exist_ok=True)
        
        # Prepare data to save
        population_data = {
            'generation': generation,
            'cars': []
        }
        
        # Save each car's AI state
        for car in cars:
            if car.use_ai:
                car_data = {
                    'fitness': car.fitness,
                    'ai_state': car.ai.state_dict(),
                    'saved_state': car.saved_state,
                    'destination_reached': car.destination_reached
                }
                population_data['cars'].append(car_data)
        
        # Save to file
        filepath = os.path.join("trained_models", filename)
        with open(filepath, 'wb') as f:
            pickle.dump(population_data, f)
        
        print(f"Population saved to {filepath} - Generation {generation}, {len(population_data['cars'])} cars")
        return True
    except Exception as e:
        print(f"Error saving population: {e}")
        return False


def load_population(filename="population.pkl"):
    """Load a population from a file"""
    try:
        filepath = os.path.join("trained_models", filename)
        if not os.path.exists(filepath):
            print(f"No saved population found at {filepath}")
            return None, 0
        
        with open(filepath, 'rb') as f:
            population_data = pickle.load(f)
        
        print(f"Population loaded from {filepath} - Generation {population_data['generation']}, {len(population_data['cars'])} cars")
        return population_data, population_data['generation']
    except Exception as e:
        print(f"Error loading population: {e}")
        return None, 0


def get_best_car_from_population():
    """Get the best performing car from saved population data"""
    population_data, _ = load_population()
    if population_data is None:
        return None
    
    best_car_data = None
    best_fitness = -float('inf')
    
    for car_data in population_data['cars']:
        if car_data['fitness'] > best_fitness:
            best_fitness = car_data['fitness']
            best_car_data = car_data
    
    return best_car_data


def create_car_from_data(car_data, road_system, pathfinder):
    """Create a new car from saved data"""
    spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
    color = random.choice([RED, GREEN, BLUE])
    car = Car(spawn_x, spawn_y, color, use_ai=True, road_system=road_system, pathfinder=pathfinder)
    car.angle = spawn_angle
    
    # Load AI state
    if car.use_ai and 'ai_state' in car_data:
        try:
            car.ai.load_state_dict(car_data['ai_state'])
        except RuntimeError as e:
            if "size mismatch" in str(e):
                # Attempt graceful upgrade if only the final layer output size changed (3 -> 4)
                print("Model size mismatch detected. Attempting to adapt saved weights to new architecture (adding steer_sens output)...")
                saved_state = car_data['ai_state']
                model_state = car.ai.state_dict()
                # Identify final linear layer keys
                final_weight_key = None
                final_bias_key = None
                for k in saved_state.keys():
                    if k.endswith('.2.weight') or k.endswith('.4.weight') or k.endswith('.6.weight'):
                        # Heuristic: last occurrence will be final layer in Sequential
                        final_weight_key = k
                    if k.endswith('.2.bias') or k.endswith('.4.bias') or k.endswith('.6.bias'):
                        final_bias_key = k
                if final_weight_key and final_bias_key and final_weight_key in saved_state:
                    old_w = saved_state[final_weight_key]
                    old_b = saved_state[final_bias_key]
                    if old_w.shape[0] == 3 and model_state[final_weight_key].shape[0] == 4:
                        # Expand to 4 outputs
                        new_w = model_state[final_weight_key].clone()
                        new_b = model_state[final_bias_key].clone()
                        new_w[:3, :] = old_w
                        new_b[:3] = old_b
                        saved_state[final_weight_key] = new_w
                        saved_state[final_bias_key] = new_b
                        try:
                            car.ai.load_state_dict(saved_state)
                            print("Successfully adapted old model to new 4-output architecture.")
                        except Exception as e2:
                            print(f"Adaptation failed, using fresh network. Error: {e2}")
                    else:
                        print("Final layer shape not as expected for adaptation; using fresh network.")
                else:
                    print("Could not locate final layer for adaptation; using fresh network.")
            else:
                raise e
    
    return car


def create_destination_mode_car(start_pos, end_pos, road_system, pathfinder):
    """Create a car for destination mode using the best AI from training"""
    best_car_data = get_best_car_from_population()
    
    if best_car_data is None:
        print("No saved population found. Using random AI.")
        car = Car(start_pos[0], start_pos[1], RED, use_ai=True, road_system=road_system, pathfinder=pathfinder)
    else:
        car = create_car_from_data(best_car_data, road_system, pathfinder)
        car.x, car.y = start_pos
        car.crashed = False
        car.time_alive = 0
        car.distance_traveled = 0
        car.fitness = 0
    
    # Set custom destination
    path = pathfinder.find_path(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
    if path:
        car.current_path = path
        car.path_waypoints = pathfinder.path_to_waypoints(path)
        # Offset waypoints to left lane for left-hand driving
        car._offset_waypoints_for_left_hand()
        car._resample_waypoints_evenly()
        car._prune_backward_waypoints()
        car.current_waypoint_index = 0
        car.individual_destination = end_pos
        if car.path_waypoints:
            car.target_x, car.target_y = car.path_waypoints[0]
            # Orient towards first waypoint
            try:
                sx, sy = car.x, car.y
                wp_index = 0
                if len(car.path_waypoints) > 1 and math.hypot(car.path_waypoints[0][0]-sx, car.path_waypoints[0][1]-sy) < 5:
                    wp_index = 1
                if wp_index < len(car.path_waypoints):
                    tx, ty = car.path_waypoints[wp_index]
                    dx = tx - sx; dy = ty - sy
                    raw_angle = math.degrees(math.atan2(dx, -dy))
                    car.angle = raw_angle
            except Exception:
                pass
    
    return car


def evolve_population(cars, population_size=POPULATION_SIZE, road_system=None, pathfinder=None, generation=1):
    # Safety: ensure we have cars
    if not cars:
        print("evolve_population: received empty car list, regenerating fresh population")
        cars = []
        for _ in range(population_size):
            sx, sy, sa = road_system.get_random_spawn_point()
            nc = Car(sx, sy, random.choice([RED,GREEN,BLUE]), use_ai=True, road_system=road_system, pathfinder=pathfinder)
            nc.angle = sa
            nc.generate_individual_destination()
            nc.orient_to_first_waypoint()
            cars.append(nc)
    # Calculate fitness for all cars (catch exceptions per car)
    for car in cars:
        try:
            car.calculate_fitness()
        except Exception as e:
            print(f"Fitness calc error for car: {e}")
            car.fitness = -1e9
    # Sort by fitness (best first), prioritizing saved cars; guard if list empty
    if cars:
        cars.sort(key=lambda x: (x.saved_state, x.fitness), reverse=True)
    saved_count = sum(1 for car in cars if car.saved_state)
    best_fit = cars[0].fitness if cars else 0.0
    print(f"Generation evolved! Best fitness: {best_fit:.2f} | Saved cars: {saved_count}")
    
    # Track statistics for graphs
    if cars:
        fitness_history.append(cars[0].fitness)
    generation_history.append(generation)
    saved_cars_history.append(saved_count)
    
    # No need to generate shared destinations - each car handles its own individual destination
    
    # Keep top performers as elites, ensuring saved cars are prioritized
    elite_count = max(1, population_size // 5)
    elites = cars[:elite_count]
    new_cars = []
    
    # Clone elites (reset their state)
    for elite in elites:
        new_car = elite.clone()
        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
        new_car.x = spawn_x
        new_car.y = spawn_y
        new_car.angle = spawn_angle
        new_car.speed = 0
        new_car.crashed = False
        new_car.fitness = 0
        new_car.time_alive = 0
        new_car.distance_traveled = 0
        new_car.off_road_time = 0
        new_car.checkpoint_times = []  # Reset checkpoint times
        new_car.last_checkpoint_time = 0  # Reset checkpoint timing
        new_car.current_checkpoint_start_time = 0  # Reset checkpoint start time
        new_car.checkpoint_streak = 0  # Reset checkpoint streak
        new_car.max_checkpoint_streak = 0  # Reset max streak
        new_car.consecutive_fast_checkpoints = 0  # Reset fast streak
        new_car.pathfinder = pathfinder  # Ensure pathfinder is set
        new_car.destination_reached = False  # Reset destination status
        new_car.saved_state = False  # Reset saved state
        # Each car will generate its own individual destination
        # Attempt to generate a valid path several times
        got_path = False
        for _ in range(5):
            new_car.generate_individual_destination()
            if new_car.path_waypoints:
                got_path = True
                break
        if got_path:
            new_car.orient_to_first_waypoint()
        else:
            # Fallback: leave angle at spawn; mark destination_reached False
            new_car.individual_destination = None
            print("Elite clone failed to get path after retries; using fallback orientation")
        new_cars.append(new_car)
    
    # Create offspring with mutations
    while len(new_cars) < population_size:
        # Select parent from top half (fallback to 0 if small)
        half_index_limit = max(0, population_size // 2 - 1)
        parent_index = random.randint(0, min(len(cars) - 1, half_index_limit))
        parent = cars[parent_index]
        offspring = parent.clone()
        # Apply mutation (stronger for worse performers)
        base_mutation_rate = 0.12
        base_mutation_strength = 0.25
        if cars and parent.fitness > cars[0].fitness * 0.8:
            offspring.mutate(mutation_rate=base_mutation_rate * 0.7, mutation_strength=base_mutation_strength * 0.8)
        else:
            offspring.mutate(mutation_rate=base_mutation_rate * 1.3, mutation_strength=base_mutation_strength * 1.2)
        # Reset offspring state
        spawn_x, spawn_y, spawn_angle = road_system.get_random_spawn_point()
        offspring.x = spawn_x
        offspring.y = spawn_y
        offspring.angle = spawn_angle
        offspring.speed = 0
        offspring.crashed = False
        offspring.fitness = 0
        offspring.time_alive = 0
        offspring.distance_traveled = 0
        offspring.off_road_time = 0
        offspring.checkpoint_times = []
        offspring.last_checkpoint_time = 0
        offspring.current_checkpoint_start_time = 0
        offspring.checkpoint_streak = 0
        offspring.max_checkpoint_streak = 0
        offspring.consecutive_fast_checkpoints = 0
        offspring.color = random.choice([RED, GREEN, BLUE])
        offspring.pathfinder = pathfinder
        offspring.destination_reached = False
        offspring.saved_state = False
        # Attempt to generate path with retries & adaptive distance threshold fallback
        got_path = False
        for attempt in range(6):
            offspring.generate_individual_destination()
            if offspring.path_waypoints:
                got_path = True
                break
            if attempt == 3:
                offspring.x += random.uniform(-20,20)
                offspring.y += random.uniform(-20,20)
        if got_path:
            offspring.orient_to_first_waypoint()
        else:
            offspring.individual_destination = None
            print("Offspring failed to get path after retries; leaving without path")
        new_cars.append(offspring)

    # Final safety: ensure population size and at least one path-having car
    pathful = sum(1 for c in new_cars if c.path_waypoints)
    if pathful == 0:
        print("No cars with valid paths; regenerating entire population fresh")
        new_cars = []
        for _ in range(population_size):
            sx, sy, sa = road_system.get_random_spawn_point()
            nc = Car(sx, sy, random.choice([RED,GREEN,BLUE]), use_ai=True, road_system=road_system, pathfinder=pathfinder)
            nc.angle = sa
            for _ in range(5):
                nc.generate_individual_destination()
                if nc.path_waypoints:
                    nc.orient_to_first_waypoint()
                    break
            new_cars.append(nc)
    if len(new_cars) < population_size:
        for _ in range(population_size - len(new_cars)):
            sx, sy, sa = road_system.get_random_spawn_point()
            nc = Car(sx, sy, random.choice([RED,GREEN,BLUE]), use_ai=True, road_system=road_system, pathfinder=pathfinder)
            nc.angle = sa
            nc.generate_individual_destination()
            if nc.path_waypoints:
                nc.orient_to_first_waypoint()
            new_cars.append(nc)

    print(f"Evolution produced {len(new_cars)} cars ({pathful} with paths)")
    
    # Assign a shared destination/path for the whole new generation for fair evaluation
    try:
        assign_shared_destination(new_cars, pathfinder)
    except Exception as e:
        print(f"assign_shared_destination failed: {e}")
    # Auto-save the population after evolution
    save_population(new_cars, generation)
    
    return new_cars


def assign_shared_destination(cars, pathfinder, min_distance=400, max_attempts=15):
    """Pick one destination and assign the path to all cars (shared route training)."""
    if not cars or not pathfinder:
        return None
    # Use first car's position as canonical start to validate destination
    sx, sy = cars[0].x, cars[0].y
    dest = None
    path0 = None
    for _ in range(max_attempts):
        node, pos = pathfinder.get_random_destination()
        if not node:
            continue
        if math.hypot(pos[0]-sx, pos[1]-sy) < min_distance:
            continue
        p = pathfinder.find_path(sx, sy, pos[0], pos[1])
        if p:
            dest = pos
            path0 = p
            break
    if dest is None:
        # Fallback: take whatever
        node, pos = pathfinder.get_random_destination()
        dest = pos
    # Assign to each car individually from its own start
    for car in cars:
        try:
            car.current_path = pathfinder.find_path(car.x, car.y, dest[0], dest[1]) or []
            car.path_waypoints = pathfinder.path_to_waypoints(car.current_path) if car.current_path else []
            # Keep to the left lane
            car._offset_waypoints_for_left_hand()
            car._resample_waypoints_evenly()
            car.orient_to_first_waypoint()
            car._prune_backward_waypoints()
            car.current_waypoint_index = 0
            car.individual_destination = dest
            if car.path_waypoints:
                car.target_x, car.target_y = car.path_waypoints[0]
                car.initial_destination_distance = math.hypot(car.x - dest[0], car.y - dest[1])
                car.last_destination_distance = None
        except Exception as e:
            print(f"assign_shared_destination: car failed to get path: {e}")
    return dest