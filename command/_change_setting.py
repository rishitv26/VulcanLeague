from .errors import *
from ..config import Config


def main(cmd_list: list[str], config: Config, detector):
    if len(cmd_list) != 3:
        illegal_argument(3)
    else:
        try:
            config.edit(cmd_list[1], cmd_list[2])
        except KeyError:
            setting_doesnt_exist()
            return None
        print(cmd_list[1], config.get(cmd_list[1]))