from errors import *
from config import Config
from ai import AI

def main(cmd_list: list[str], config: Config, detector: AI):
    if len(cmd_list) != 2:
        illegal_argument(2)
    else:
        try:
            print(config.get(cmd_list[1]))
        except KeyError:
            setting_doesnt_exist()