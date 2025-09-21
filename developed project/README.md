# Autonomous Driver Simulation

This project is a simulation environment for autonomous driving, featuring AI models, pathfinding algorithms, and 3D visualization.

## Features

- AI-driven car simulation
- Pathfinding using Dijkstra's algorithm
- 3D visualization with Panda3D
- Evolutionary algorithms for optimization
- Fluid dynamics simulation
- Camera controls and HUD

## Requirements

- Python 3.10+
- Required packages: Install via `pip install -r requirements.txt` (if available)

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main simulation: `python main.py`

## Usage

- Start the simulation by running `main.py`
- Use the UI controls to adjust parameters
- View 3D simulation in the browser via `3dsim.html`

## Project Structure

- `main.py`: Entry point for the simulation
- `ai_models.py`: AI model implementations
- `car.py`: Car simulation logic
- `simulation_core.py`: Core simulation engine
- `visual3d_server.py`: 3D visualization server
- `ui/`: User interface components
- `trained_models/`: Saved AI models
- `poparchive/`: Population archives for evolution

## Contributing

Feel free to contribute by submitting pull requests or issues.

## License

[Add license information here]