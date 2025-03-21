"""
Helper script that simulates the network using an `experiment.py` file
and the network hash. Writes the simulation hash to a simulation_hash file.

Mainly evoked by snakemake
"""

import os
import sys
import importlib.util

from ..network import networkDictFromDump
from ..simulation import simulationDictFromDump
from ..analysis import Analysis
from .snakemake_helpers import nested_dict_update, get_git_revision_hash
from ..params.default_ana_params import params as ana_params
from ..params.default_net_params import params as net_params

# Load script from specified path sys.argv[1] (snakemake {input})
# Does nothing else than `import path/to/script as exp` would do
conf_name, _ = os.path.splitext(os.path.basename(sys.argv[1]))
conf_path = os.path.join(os.getcwd(), sys.argv[1])
spec = importlib.util.spec_from_file_location(conf_name, conf_path)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

# Update default parameters with parameters specified in exp
nested_dict_update(net_params, exp.net_params)
nested_dict_update(ana_params, exp.ana_params)

outpath = net_params['outpath']

# Read network hash
with open(sys.argv[2], 'r') as f:
    net_hash = f.read()

# Read simulation hash
with open(sys.argv[3], 'r') as f:
    sim_hash = f.read()

# Read network dict
net_folder = os.path.join(outpath, net_hash)
net_dict = networkDictFromDump(net_folder)

# Read simulation dict
sim_folder = os.path.join(outpath, net_hash, sim_hash)
sim_dict = simulationDictFromDump(sim_folder)

# Create Analysis class and export it
base_path = sys.argv[5]
ana = Analysis(ana_params, net_dict, sim_dict, sim_folder, base_path)
ana.dump(sim_folder)

# Do the analysis
ana.fullAnalysis()

# Save the analysis hash
hash_file = sys.argv[4]
hash_fn = os.path.join(os.getcwd(), hash_file)
with open(hash_fn, 'w') as f:
    f.write(ana.getHash())

# Save the humam repository hash
ana_folder = os.path.join(sim_folder, ana.getHash())
git_fn = os.path.join(ana_folder, "git_hash.txt")
with open(git_fn, 'w') as f:
    f.write(get_git_revision_hash())
