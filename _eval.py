from errors import *
import util
from config import Config
from ai import AI


def main(cmd_list, config: Config, detector: AI):
    print("Preparing for ink evaluation...\n")
    print("DO NOT let machine turn off or sleep during this process.")
    print("Close all other programs for best performance.")
    print("Press enter when you are ready to continue.")
    util.pause()
    util.clear()
    try:
        config = Config()
        
        # find all thresholds:
        thresholds = []
        ts = config.get("threshold")
        for i in ts.split(","):
            thresholds.append(float(i))
        
        # run on all thresholds:
        for threshold in thresholds:
            print(f"Running using {threshold*100}% threshold...")
            detector.eval_model(threshold)
            util.clear()

    except Exception as e:
        print("There was an error during the evaluation proccess. Please ensure all settings are valid and correct.")
        print("If error persists, contact the developers or start an issue at our repository.")
        print(f"Error: {e}")
