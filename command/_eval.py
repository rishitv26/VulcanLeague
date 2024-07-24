
def main():
    print("Preparing for ink evaluation...\n")
    print("DO NOT let machine turn off or sleep during this process.")
    print("Close all other programs for best performance.")
    print("Press enter when you are ready to continue.")
    util.pause()
    util.clear()
    detector.eval_model(float(util.get_setting("threshold")))