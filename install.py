import os
import util
import time
from config import Config

def setup_configs():
    config = Config()
    config.add("setup", "true")
    config.add("base_path", os.getcwd())
    config.add("trained", "false")
    config.add("batch_size", "32")
    config.add("training_steps", "60000")
    config.add("learning_rate", "1e-3")
    config.add("training_data", "1")
    config.add("threshold", "0.4")
    config.save()


def main():
    print("Welcome to the VLAE 1.0.0 (Vulcan League AI Engine)")
    print("Starting initial setup... Press enter to continue.")
    util.pause()
    util.clear()
    ################# import dependencies:

    import os
    if not util.is_windows():
        os.system("python3 -m pip install --upgrade pip")
    else:
        os.system("python -m pip install --upgrade pip")

    print("Installing required libraries...")
    os.system("pip3 install -r requirements.txt")
    print("Dependency installation complete! initializing settings...")
    
    try:
        setup_configs()
    except KeyError:
        print("Using existing settings...")
        reply = util.ask("Reset Settings?")
        if reply:
            os.remove("config.txt")
            setup_configs()

    print("Setup complete!")
    util.clear()
    util.pause()
    
    print("Please reopen the application for use.")
    time.sleep(1)
    exit(0)
