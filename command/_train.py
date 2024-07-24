
def main():
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
    detector.load_model([i for i in util.get_setting("training_data").split(",")])