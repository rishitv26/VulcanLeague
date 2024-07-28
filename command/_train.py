from .errors import *
import os
from .. import util
from ..config import Config
from AI.ai import AI

def main(cmd_list, config: Config, detector: AI):
    if not os.path.isdir("data"):
        reply = data_not_downloaded()
        if reply:
            return None
    print("Preparing to train this model...\n")
    print("DO NOT let machine turn off or sleep during this process.")
    print("Close all other programs for best performance.")
    print("Press enter when you are ready to continue.")
    util.pause()
    util.clear()
    try:
        detector.load_model([i for i in config.get("training_data").split(",")])
    except:
        print("There was an error in loading the model. Please check your settings to ensure they are correct.")
        print("If error persists, contact the developers or start an issue at our repository.")
