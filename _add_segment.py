from errors import *
import shutil
import os
from config import Config

def main(cmd_list: list[str], config: Config, detector):
    if len(cmd_list) != 2:
        illegal_argument(2)
    else:
        try:
            destination_folder = os.path.join(config.get("base_path") + '\\data\\vesuvius-challenge-ink-detection\\test\\', os.path.basename(cmd_list[1]))
            shutil.copytree(cmd_list[1], destination_folder)
        except FileExistsError:
            segment_exists()