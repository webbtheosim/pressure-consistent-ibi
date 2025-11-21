#!/usr/bin/env python3
import pdb

class IBISimulation:
    def __init__(self, config_path):
        from .initialization import Initializer
        import yaml
        import matplotlib.pyplot as plt

        self.initializer = Initializer(config_path)

    def run_simulation(self, start=1):
        from .simulation import Simulator
        Simulator(self.initializer).run(start=start)


def main():
    import argparse

    try:
        parser = argparse.ArgumentParser(description="Run the simulation wrapper.")
        parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
        parser.add_argument("--start", type=int, default=1, help="Starting iteration number.")
        args = parser.parse_args()

        print("Starting simulation with config:", args.config, flush=True)
        simulation = IBISimulation(args.config)
        print("Running simulation...", flush=True)
        simulation.run_simulation(args.start)
        print("Simulation completed successfully.", flush=True)
    except Exception as e:
        print(e)
        pdb.post_mortem()

    # simulation.run_simulation()

def plot():
    import argparse
    parser = argparse.ArgumentParser(description="Run the simulation wrapper.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    simulation = IBISimulation(args.config)