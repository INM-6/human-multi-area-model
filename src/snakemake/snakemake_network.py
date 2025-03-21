"""
Helper script that creates the network using an `experiment.py` file,
dumps the network and writes the network hash to a network_hash file.

Mainly evoked by snakemake
"""

import os
import sys
import importlib.util

from .snakemake_helpers import nested_dict_update, get_git_revision_hash
from ..params.default_net_params import params as net_params
from ..data_preprocessing.cytoarchitecture import NeuronNumbers
from ..data_preprocessing.connectivity import SynapseNumbers
from ..network import Network


# Load script from specified path sys.argv[1] (snakemake {input})
# Does nothing else than `import path/to/script as exp` would do
conf_name, _ = os.path.splitext(os.path.basename(sys.argv[1]))
conf_path = os.path.join(os.getcwd(), sys.argv[1])
spec = importlib.util.spec_from_file_location(conf_name, conf_path)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

# Update default parameters with parameters specified in exp
nested_dict_update(net_params, exp.net_params)
outpath = net_params['outpath']

# Create Network class
NN = NeuronNumbers(
    surface_area=net_params['surface_area'],
    **net_params['cytoarchitecture_params']
)
SN = SynapseNumbers(
    NN=NN,
    **net_params['predictive_connectomic_params']
)
net = Network(NN, SN, net_params)

# Export the network
net.dump(outpath)

# Save the network hash
hash_file = sys.argv[2]
hash_fn = os.path.join(os.getcwd(), hash_file)
with open(hash_fn, 'w') as f:
    f.write(net.getHash())

# Save the humam repository hash
net_folder = os.path.join(outpath, net.getHash())
git_fn = os.path.join(net_folder, "git_hash.txt")
with open(git_fn, 'w') as f:
    f.write(get_git_revision_hash())
