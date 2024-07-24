
import util


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
    
    # TODO: Config CLASS

    # load_settings()
    # modify_setting("setup", "true")
    # modify_setting("base_path", os.getcwd())
    # modify_setting("trained", "false")
    # modify_setting("batch_size", "32")
    # modify_setting("training_steps", "60000")
    # modify_setting("learning_rate", "1e-3")
    # modify_setting("training_data", "1")
    # modify_setting("threshold", "0.4")

    print("Setup complete!")
    print("Please reopen the application to use the AI.")
    util.pause()
    
    import run
    run.main()

