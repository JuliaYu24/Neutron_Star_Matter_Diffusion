#!/usr/bin/env python3
from eos_diffusion.train import train

if __name__ == "__main__":
    train(
        data_path="eos_training_curves/data",
        checkpoint_dir="checkpoints",
        save_every=25,
    )
