from utils.config import Config

args = Config()
stroke_config = Config()

def update_args(args_):
    args.update(args_)

def update_config(config_):
    stroke_config.update(config_)