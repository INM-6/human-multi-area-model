# -*- coding: utf-8 -*-

"""
Some helper functions for snakemake.
"""

__all__ = [
    "get_git_revision_hash",
    "nested_dict_update",
]

def get_git_revision_hash():
    import subprocess as subprocess

    git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
            ).decode('ascii').strip()
    return git_hash


def nested_dict_update(default_dict, specific_dict):
    """
    Updates the values of default_dict by the values in specific_dict
    recursively.
    """
    for key in specific_dict:
        # Fully replace neuron_params_X or neuron_param_dist_X dictionaries
        if isinstance(key, str) and key.startswith('neuron_param'):
            print('Specified {0} dict to {1}'.format(
                key, specific_dict[key]
            ))
            default_dict[key] = specific_dict[key]
        elif isinstance(specific_dict[key], dict):
            nested_dict_update(default_dict[key], specific_dict[key])
        else:
            if key in default_dict.keys():
                if isinstance(specific_dict[key], type(default_dict[key])) or default_dict[key] is None:
                    default_dict[key] = specific_dict[key]
                else:
                    raise TypeError(
                        'Type of specific_dict[{0}]={1} and '
                        'default_dict[{0}]={2} dont match.'.format(
                            key,
                            specific_dict[key],
                            default_dict[key]
                        )
                    )
            else:
                print('Adding entry default_dict[{0}]={1}'.format(
                    key, specific_dict[key]
                ))
                default_dict[key] = specific_dict[key]
