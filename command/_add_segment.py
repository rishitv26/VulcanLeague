from .errors import *
import shutil
from ..config import Config

def main(cmd_list: list[str], config: Config, detector):
    if len(cmd_list) != 2:
        illegal_argument(2)
    else:
        try:
            shutil.copytree(cmd_list[1], config.get("base_path") + '\\data\\test\\')
        except FileExistsError:
            segment_exists()