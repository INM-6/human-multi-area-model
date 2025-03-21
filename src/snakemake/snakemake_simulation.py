"""
Helper script that simulates the network using an `experiment.py` file
and the network hash. Writes the simulation hash to a simulation_hash file.

Mainly evoked by snakemake
"""

import os
import sys
import importlib.util

from .snakemake_helpers import nested_dict_update, get_git_revision_hash
from ..params.default_sim_params import params as sim_params
from ..params.default_net_params import params as net_params
from ..network import networkDictFromDump
from ..simulation import Simulation

# Load script from specified path sys.argv[1] (snakemake {input})
# Does nothing else than `import path/to/script as exp` would do
conf_name, _ = os.path.splitext(os.path.basename(sys.argv[1]))
conf_path = os.path.join(os.getcwd(), sys.argv[1])
spec = importlib.util.spec_from_file_location(conf_name, conf_path)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

# Update default parameters with parameters specified in exp
nested_dict_update(net_params, exp.net_params)
nested_dict_update(sim_params, exp.sim_params)

outpath = net_params['outpath']

# Read network hash
with open(sys.argv[2], 'r') as f:
    net_hash = f.read()

# Read network dict
net_folder = os.path.join(outpath, net_hash)
net_dict = networkDictFromDump(net_folder)

# Create Simulation class, export it and calculate the hash
sim = Simulation(sim_params, net_dict)
sim.dump(net_folder)
sim_hash = sim.getHash()


# Set output directory according to hashes, instantiate the Network
# and run the simulation
data_path = os.path.join(outpath, net_hash, sim_hash)
num_threads = int(sys.argv[4])
sim.setup(data_path, num_threads)
sim.simulate()

# Save the network hash
hash_file = sys.argv[3]
hash_fn = os.path.join(os.getcwd(), hash_file)
with open(hash_fn, 'w') as f:
    f.write(sim_hash)

# Save the humam repository hash
sim_folder = os.path.join(net_folder, sim_hash)
git_fn = os.path.join(sim_folder, "git_hash.txt")
with open(git_fn, 'w') as f:
    f.write(get_git_revision_hash())
