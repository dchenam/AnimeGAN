import json
from bunch import Bunch
import os
import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'test'],
        help='choose training or test mode')
    argparser.add_argument(
        '--resume',
        default=None,
        action='store_true',
        help='resume from checkpoint')
    args = argparser.parse_args()

    return args


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(jsonfile):
    config, _ = get_config_from_json(jsonfile)
    config.model_dir = os.path.join("./experiments", config.exp_name)
    config.sample_dir = os.path.join("./experiments", config.exp_name, "sample")
    config.summary_dir = os.path.join("./experiments", config.exp_name, "summary")
    config.checkpoint_dir = os.path.join("./experiments", config.exp_name, "checkpoint")
    return config


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def print_network(model, name=None):
    """Print out the network information."""
    num_params = 0
    train_params = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            train_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))
    print("The number of trainable parameters: {}".format(train_params))
