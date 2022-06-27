#!/usr/bin/env python3
import os

flow_seed = int(os.environ['VAE_SEED'])
grid_seed = flow_seed + 1_678_498

flow_seed = str(flow_seed)
grid_seed = str(grid_seed)

print("Setting seeds", flow_seed, grid_seed)

with open("examples/vae/config_base.yml") as f:
    config = f.read()

config_out = config.replace("$FLOW_ENGINE_SEED", flow_seed).replace("$GRID_SELECTION_SEED", grid_seed)

# check somethiing changed correctly
assert config_out != config

with open(f"examples/vae/config_{flow_seed}.yml", "w") as f:
    f.write(config_out)
