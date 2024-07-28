from .errors import *
from .. import util
from ..config import Config
from AI.ai import AI


def main(cmd_list, config: Config, detector: AI):
    print("Preparing for ink evaluation...\n")
    print("DO NOT let machine turn off or sleep during this process.")
    print("Close all other programs for best performance.")
    print("Press enter when you are ready to continue.")
    util.pause()
    util.clear()
    try:
        detector.eval_model(float(config.get("threshold")))
    except:
        print("There was an error during the evaluation proccess. Please ensure all settings are valid and correct.")
        print("If error persists, contact the developers or start an issue at our repository.")
