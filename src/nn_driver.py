import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import numpy as np
from src.vehicle import Vehicle
from src.road import RoadSystem

class DrivingNN(nn.Module):
    """
    Simple feedforward network that outputs throttle and steering commands in [-1,1].
    """
    def __init__(self, input_size, hidden_size=64):
        super(DrivingNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # throttle, steer
        return torch.tanh(self.fc3(x))

class DriverIndividual:
    """
    Wraps a DrivingNN and its fitness score.
    """
    def __init__(self, model):
        self.model = model
        self.fitness = 0.0

    def mutate(self, rate=0.02):
        """Add random noise to weights."""
        for param in self.model.parameters():
            if len(param.shape) > 1:
                noise = torch.randn_like(param) * rate
                param.data += noise

    @staticmethod
    def crossover(parent1, parent2):
        """Combine weights from two parents."""
        child = copy.deepcopy(parent1)
        for p_c, p1, p2 in zip(child.model.parameters(), parent1.model.parameters(), parent2.model.parameters()):
            mask = torch.rand_like(p1) < 0.5
            p_c.data.copy_(torch.where(mask, p1.data, p2.data))
        return child

class SimulationEnv:
    """
    Runs a simulation for one individual and returns fitness.
    """
    def __init__(self, renderer, noise_texture, car_model, map_center, radius=500, max_steps=1000):
        self.road_system = RoadSystem(renderer, noise_texture, car_model.model.getBounds().getMax().getX())
        # load a generic road area around map_center
        self.road_system.load_roads_from_osm(*map_center, radius=radius, park_car_model=None)
        self.max_steps = max_steps
        self.car_model = car_model
        self.renderer = renderer

    def reset(self):
        # instantiate a new vehicle
        self.vehicle = Vehicle(self.car_model.loader, self.renderer)
        if self.road_system.spawn_point:
            p = self.road_system.spawn_point
            self.vehicle.model.setPos(p)
        self.steps = 0
        self.total_reward = 0.0
        self.last_side = None

    def get_inputs(self):
        # sensor: distance to left/right road edges, angle to direction
        pos = self.vehicle.get_position()
        # distance to road centerline
        dist_center = self.road_system.point_line_distance(pos, *self.road_system.get_closest_road_segment(pos)[0][:2])
        # which side (-1 left, 0 center, 1 right)
        side = 0
        s = self.road_system.get_road_side(pos)
        if s == 'left': side = -1
        elif s == 'right': side = 1
        # velocity normalized
        vel = self.vehicle.velocity / self.vehicle.max_speed
        angle = np.deg2rad(self.vehicle.model.getH() % 360)
        return torch.tensor([dist_center / (self.road_system.road_width/2), side, vel, np.cos(angle), np.sin(angle)], dtype=torch.float32)

    def step(self, individual):
        inputs = self.get_inputs()
        with torch.no_grad():
            out = individual.model(inputs)
        throttle, steer = out.numpy()
        # apply controls
        individual_controls = self.vehicle.controls
        individual_controls['forward'] = throttle > 0
        individual_controls['backward'] = throttle < 0
        individual_controls['left'] = steer > 0.1
        individual_controls['right'] = steer < -0.1
        pos_before = self.vehicle.get_position()
        self.vehicle.update(0.1, self.road_system.is_on_road(pos_before))
        pos_after = self.vehicle.get_position()
        # reward: forward progress
        progress = (pos_after - pos_before).length()
        reward = progress
        # penalty for leaving road
        if not self.road_system.is_on_road(pos_after): reward -= 1.0
        # reward for staying on left side
        side = self.road_system.get_road_side(pos_after)
        if side == 'left': reward += 0.1
        elif side == 'right': reward -= 0.1
        # discourage stopping
        if progress < 0.01: reward -= 0.5
        individual.fitness += reward
        self.steps += 1
        return self.steps < self.max_steps

    def run(self, individual):
        """Evaluate individual over max_steps."""
        self.reset()
        while self.step(individual): pass
        return individual.fitness

class GeneticTrainer:
    """
    Manages population, selection, crossover, mutation and evolution.
    """
    def __init__(self, env, population_size=50, retain_frac=0.2, random_select=0.05, mutate_rate=0.02):
        self.env = env
        self.pop_size = population_size
        self.retain = retain_frac
        self.random_select = random_select
        self.mutate_rate = mutate_rate
        self.population = [DriverIndividual(DrivingNN(input_size=5)) for _ in range(self.pop_size)]

    def evolve(self, generations=100):
        for gen in range(generations):
            # evaluate
            for indiv in self.population:
                indiv.fitness = self.env.run(indiv)
            # sort by fitness
            graded = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            retain_length = int(len(graded)*self.retain)
            parents = graded[:retain_length]
            # random add
            for indiv in graded[retain_length:]:
                if random.random() < self.random_select:
                    parents.append(indiv)
            # crossover
            desired_length = self.pop_size - len(parents)
            children = []
            while len(children) < desired_length:
                p1, p2 = random.sample(parents, 2)
                child = DriverIndividual.crossover(p1, p2)
                child.mutate(self.mutate_rate)
                children.append(child)
            self.population = parents + children
            best = self.population[0]
            print(f"Gen {gen+1}: Best fitness = {best.fitness:.2f}")
        return self.population[0]

if __name__ == '__main__':
    # Example usage: instantiate Panda3D renderer, road, and trainer
    from direct.showbase.ShowBase import ShowBase
    base = ShowBase()
    # load a placeholder box as car model
    car_model = base.loader.loadModel('models/box')
    trainer = GeneticTrainer(SimulationEnv(base.render, None, car_model, map_center=(-36.89,174.94)))
    best_driver = trainer.evolve(generations=50)
    # save best model
    torch.save(best_driver.model.state_dict(), 'best_driver.pth')
