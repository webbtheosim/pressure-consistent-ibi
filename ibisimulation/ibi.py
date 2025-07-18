#!/usr/bin/env python3
import pdb

class IBISimulation:
    def __init__(self, config_path):
        from .initialization import Initializer
        import yaml
        import matplotlib.pyplot as plt

        self.initializer = Initializer(config_path)

    def run_simulation(self):
        from .simulation import Simulator
        Simulator(self.initializer).run()


def main():
    import argparse

    try:
        parser = argparse.ArgumentParser(description="Run the simulation wrapper.")
        parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
        args = parser.parse_args()

        print("Starting simulation with config:", args.config)
        simulation = IBISimulation(args.config)
        print("Running simulation...")
        simulation.run_simulation()
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