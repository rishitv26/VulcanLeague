from errors import *
from config import Config
from ai import AI

def main(cmd_list: list[str], config: Config, detector: AI):
    for setting in config.SETTINGS:
        print(str(setting) + ": " + str(config.SETTINGS[setting]))