from errors import *
import shutil
import os
from config import Config

def main(cmd_list: list[str], config: Config, detector):
    if len(cmd_list) != 2:
        illegal_argument(2)
    else:
        # check if the segment has the mask.png file and the surface_volume folder
        if not os.path.exists(os.path.join(cmd_list[1], 'mask.png')):
            segment_not_formated_correctly()
            return
        if not os.path.exists(os.path.join(cmd_list[1], 'surface_volume')):
            segment_not_formated_correctly()
            return
        try:
            destination_folder = os.path.join(os.path.join(config.get("base_path"), '/data/vesuvius-challenge-ink-detection/test/'), os.path.basename(cmd_list[1]))
            shutil.copytree(cmd_list[1], destination_folder)
        except FileExistsError:
            segment_exists()