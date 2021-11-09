import os
import json
import datetime
from enum import Enum

class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def create_env_folder(env_name, network_class, test=False):
    # might have diambiguation problems, but very unlikely to have match to the millisecond
    wrapper_folder = f'{env_name}-{datetime.datetime.now().strftime("%m-%d-%Y")}'
    inner_folder = f'{env_name}-{network_class}-{datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S-%f")}'
    if test:
        wrapper_folder += '-test'
        inner_folder += '-test'
    folder_path = os.path.join('logs', wrapper_folder, inner_folder)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_kwargs(kwargs_dict, dir):
    with open(os.path.join(dir, "variant.json"), "w") as f:
        json.dump(kwargs_dict, f, indent=2, sort_keys=True, cls=MyEncoder)
