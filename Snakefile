import os
import json

configfile: 'config.yaml'

# Read in all .py files in 'experiments' as EXPERIMENTS
EXPERIMENTS = []
for f in os.listdir('experiments'):
    fn, ext = os.path.splitext(f)
    if ext == '.py':
        EXPERIMENTS.append(fn)

if 'BASE_PATH' in config:
    BASE_PATH = config['BASE_PATH']
else:
    BASE_PATH = os.getcwd()

NEST_VARS_PATH = os.path.join(config['NEST_PATH'], 'bin', 'nest_vars.sh')

with open('cluster.json', 'r') as f:
    THREADS_SIMULATE = json.load(f)['simulateNetwork']['cpus-per-task']

# Load necessary modules for MPI and NEST before executing
RUN_CREATE = 'python'
if 'MPI_MODULES' in config:
    RUN_CREATE = '{}; {}'.format(config['MPI_MODULES'], RUN_CREATE)

RUN_SIMULATE = 'python'
if 'MPIRUN' in config:
    RUN_SIMULATE = '{} {}'.format(config['MPIRUN'], RUN_SIMULATE)
RUN_SIMULATE = 'source {}; {}'.format(NEST_VARS_PATH, RUN_SIMULATE)
if 'MPI_MODULES' in config:
    RUN_SIMULATE = '{}; {}'.format(config['MPI_MODULES'], RUN_SIMULATE)

RUN_ANALYSIS = 'source {}; python'.format(NEST_VARS_PATH)
if 'MPI_MODULES' in config:
    RUN_ANALYSIS = '{}; {}'.format(config['MPI_MODULES'], RUN_ANALYSIS)

rule all:
    input:
        expand('experiments/{experiment}.analysis_hash', experiment=EXPERIMENTS)

rule createNetwork:
    input:
        'experiments/{experiment}.py'
    output:
        'experiments/{experiment}.network_hash'
    conda:
        'humam.yml'
    log:
        'out/log/{experiment}_createNetwork.log'
    shell:
        '{RUN_CREATE} src/snakemake/snakemake_network.py {input} {output}'

rule simulateNetwork:
    input:
        'experiments/{experiment}.py',
        'experiments/{experiment}.network_hash'
    output:
        'experiments/{experiment}.simulation_hash'
    conda:
        'humam.yml'
    threads:
        THREADS_SIMULATE
    log:
        'out/log/{experiment}_simulateNetwork.log'
    shell:
        '{RUN_SIMULATE} src/snakemake/snakemake_simulation.py {input} {output} {THREADS_SIMULATE}'

rule analyzeNetwork:
    input:
        'experiments/{experiment}.py',
        'experiments/{experiment}.network_hash',
        'experiments/{experiment}.simulation_hash'
    output:
        'experiments/{experiment}.analysis_hash'
    conda:
        'humam.yml'
    log:
        'out/log/{experiment}_analyzeNetwork.log'
    shell:
        '{RUN_ANALYSIS} src/snakemake/snakemake_analysis.py {input} {output} {BASE_PATH}'
