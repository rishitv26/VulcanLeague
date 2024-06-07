#
# Main file that runs tensorflow network. Run this in server.
#

from util import *
import commands

def main():
    print("Welcome to the VLAE (Vulcan League AI Engine) 1.0.0")
    print("type 'help' to see the list of commands.")
    print("type 'manual' for a basic tutorial on what to do.")

    commands.main()

def first_setup():
    print("Welcome to the VLAE 1.0.0 (Vulcan League AI Engine)")
    print("Starting initial setup... Press enter to continue.")
    pause()
    clear()
    ################# import dependencies:

    import os
    os.system("pip3 install --upgrade pip")
    clear()

    if not is_mac():
        reply = ask("Do you have an NVIDIA GPU setup on this machine?> ")
        if reply:
            reply = ask("Install using CUDA version 11.8? Will install 12.1 if not> ")
            if reply:
                # install CUDA 11.8
                os.system("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            else:
                # install CUDA 12.1
                os.system("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        else:
            if not is_windows():
                # linux CPU
                os.system("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            else:
                # windows default
                os.system("pip3 install torch torchvision")
    else:
        # MAC install:
        os.system("pip3 install torch torchvision")

    print("Installing rest of dependencies...")
    os.system("pip3 install matplotlib numpy pandas scikit-learn tqdm opendatasets")
    clear()
    print("Dependency installation complete! initializing settings...")

    load_settings()
    modify_setting("setup", "true")
    modify_setting("base_path", os.getcwd())
    modify_setting("trained", "false")
    modify_setting("batch_size", "32")
    modify_setting("training_steps", "60000")
    modify_setting("learning_rate", "1e-3")
    modify_setting("training_data", "1,2,3")
    save_settings()

    print("Settings initialized. Installing training and testing data...")
    print("Press enter to continue.")
    pause()

    import opendatasets as od
    
    print("Starting Download...")
    od.download("https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/data", "data/")
    print("Download Complete...")

    print("Setup complete!")
    print("Please reopen the application to use AI.")
    pause()
    exit_routine()

if __name__ == "__main__":
    clear()
    file = open("mem.txt", 'r')
    content = file.read()
    file.close()

    try:
        if content == "":
            first_setup()
        else:
            main()
    except KeyboardInterrupt:
        exit_routine()
