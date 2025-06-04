"""
Neural network evolution for autonomous driving using survival-based learning.
"""
import numpy as np
import random
from src.vehicle import Vehicle
from src.road import RoadSystem
from src.raycast import Raycast

# --- Neural Network ---
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b2 = np.zeros(hidden_size)
        self.w3 = np.random.randn(hidden_size, output_size) * 0.1
        self.b3 = np.zeros(output_size)

    def forward(self, x):
        h1 = np.maximum(0, np.dot(x, self.w1) + self.b1)  # ReLU activation
        h2 = np.maximum(0, np.dot(h1, self.w2) + self.b2)  # ReLU activation
        out = np.tanh(np.dot(h2, self.w3) + self.b3)  # Tanh for output
        return out

    def mutate(self, rate=0.1, scale=0.5):
        for param in [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]:
            mask = np.random.rand(*param.shape) < rate
            param[mask] += np.random.randn(*param[mask].shape) * scale

    def copy(self):
        clone = SimpleNN(self.w1.shape[0], self.w1.shape[1], self.w3.shape[1])
        clone.w1 = np.copy(self.w1)
        clone.b1 = np.copy(self.b1)
        clone.w2 = np.copy(self.w2)
        clone.b2 = np.copy(self.b2)
        clone.w3 = np.copy(self.w3)
        clone.b3 = np.copy(self.b3)
        return clone

# Evolutionary Manager
class EvolutionManager:
    def __init__(self, loader, render, road: RoadSystem, batch_size=20, input_size=7, hidden_size=16, output_size=2):
        self.loader = loader
        self.render = render
        self.road = road
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        spawn_point = getattr(road, 'spawn_point', None)
        if spawn_point is None:
            raise ValueError('RoadSystem has no spawn_point set. Make sure roads are loaded and spawn_point is initialized.')
        self.population = [SimpleNN(input_size, hidden_size, output_size) for _ in range(batch_size)]
        self.vehicles = [Vehicle(self.loader, self.render) for _ in range(batch_size)]
        for v in self.vehicles:
            v.model.setPos(spawn_point)
        self.scores = np.zeros(batch_size)
        self.raycast = Raycast()

    def get_inputs(self, vehicle: Vehicle):
        rays = self.raycast.cast(vehicle, self.road, num_rays=5, max_dist=100)
        speed = vehicle.speed / vehicle.max_speed
        center_offset = self.road.lane_offset(vehicle.get_position())
        return np.array(rays + [speed, center_offset])

    def step(self):
        for i, (nn, vehicle) in enumerate(zip(self.population, self.vehicles)):
            if vehicle.crashed:
                continue
            inputs = self.get_inputs(vehicle)
            action = nn.forward(inputs)
            # Output: [steer, throttle], both in [-1, 1]
            steer = float(action[0])
            throttle = float(action[1])
            # Apply throttle and steering to the vehicle
            vehicle.set_control("forward", throttle > 0.1)
            vehicle.set_control("backward", throttle < -0.1)
            vehicle.set_control("left", steer < -0.1)
            vehicle.set_control("right", steer > 0.1)
            # Prevent cars from leaving the road
            if not self.road.is_on_road(vehicle.get_position()):
                vehicle.crashed = True
                continue
            vehicle.update(0.1, self.road.is_on_road(vehicle.get_position()))
            self.scores[i] += self.reward(vehicle)

    def reward(self, vehicle: Vehicle):
        # Reward for moving forward, staying left, not crashing
        if vehicle.crashed:
            return -100
        reward = 0.1  # alive bonus
        reward += vehicle.speed  # encourage speed
        # Encourage staying on left side of lane
        offset = self.road.lane_offset(vehicle.get_position())
        reward -= abs(offset - self.road.lane_width * -0.25) * 0.5
        # Penalize going off road
        if not self.road.is_on_road(vehicle.get_position()):
            reward -= 10
        return reward

    def evolve(self):
        # Select top performers
        idx = np.argsort(self.scores)[-self.batch_size//2:]
        survivors = [self.population[i].copy() for i in idx]
        # Mutate to fill population
        new_population = []
        for nn in survivors:
            new_population.append(nn)
            clone = nn.copy()
            clone.mutate()
            new_population.append(clone)
        self.population = new_population[:self.batch_size]
        # Reset vehicles and scores
        spawn_point = getattr(self.road, 'spawn_point', None)
        self.vehicles = [Vehicle(self.loader, self.render) for _ in range(self.batch_size)]
        for v in self.vehicles:
            v.model.setPos(spawn_point)
        self.scores = np.zeros(self.batch_size)

    def run_generation(self, steps=500):
        for step in range(steps):
            self.step()
            # Reset every 10 seconds
            if step > 0 and step % 100 == 0:
                self.evolve()

    def best_vehicle(self):
        idx = np.argmax(self.scores)
        return self.vehicles[idx], self.population[idx]

def test_neural_network():
    """Test the neural network's input-output behavior and integration."""
    # Create a dummy road and vehicle
    class DummyRoad:
        def is_on_road(self, position):
            return True

        def lane_offset(self, position):
            return 0.0

        lane_width = 3.5

    class DummyVehicle:
        def __init__(self):
            self.x = 0
            self.y = 0
            self.speed = 10
            self.max_speed = 20
            self.crashed = False

        def get_position(self):
            return (self.x, self.y)

    # Initialize components
    road = DummyRoad()
    vehicle = DummyVehicle()
    nn = SimpleNN(input_size=14, hidden_size=16, output_size=2)  # 12 rays + speed + offset

    # Simulate raycasting inputs
    raycast = Raycast()
    rays = raycast.cast(vehicle, road, num_rays=12, max_dist=100)
    speed = vehicle.speed / vehicle.max_speed
    center_offset = road.lane_offset(vehicle.get_position())
    inputs = np.array(rays + [speed, center_offset])

    # Forward pass through the network
    outputs = nn.forward(inputs)

    # Validate outputs
    assert len(outputs) == 2, "Network should output two values: [steer, throttle]"
    assert -1 <= outputs[0] <= 1, "Steering output should be in range [-1, 1]"
    assert -1 <= outputs[1] <= 1, "Throttle output should be in range [-1, 1]"

    print("Neural network test passed.")

# Run the test
test_neural_network()
