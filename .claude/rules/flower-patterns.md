# Flower Patterns

- Separate **client app**, **server app**, and **strategy** modules per baseline or phase.
- Keep dataset loading and model definitions out of strategy classes.
- Use Flower's recommended `ClientApp` / `ServerApp` patterns for simulation.
- Hydra configs select algorithm and dataset; avoid hardcoding hyperparameters in Python.
