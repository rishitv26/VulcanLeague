from errors import *
import shutil
from config import Config
import os

def main(cmd_list: list[str], config: Config, detector):
    if len(cmd_list) != 2:
        illegal_argument(2)
    else:
        try:
            shutil.rmtree(os.path.join(config.get('base_path'), 'data/vesuvius-challenge-ink-detection/test/' + cmd_list[1]))
        except FileNotFoundError:
            segment_doesnt_exist()
