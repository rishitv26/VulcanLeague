
import os
from config import Config
import util
import shutil
import os
import AI.ai as ai
from command import _help, _manual, _change_setting, _get_setting, _add_segment, _rm_segment, _eval, _train

def illegal_argument(x: int):
    print("ERROR: Invalid Argument Call")
    print(f"Expected {x} arguments.")

def segment_exists():
    print("ERROR: This segment already exists with the same name.")

def segment_doesnt_exist():
    print("ERROR: This segment does not exist.")

def setting_doesnt_exist():
    print("ERROR: This setting has not been defined or doesnt exist.")

def data_not_downloaded():
    print("ERROR: The data required to train the model is not yet installed.\n")
    reply = util.ask("Install them now?> ")
    if reply:
        ai.download_data()
        return False
    else:
        return True

def main():
    print("Welcome to the VLAE (Vulcan League AI Engine) 1.0.0")
    print("type 'help' to see the list of commands.")
    print("type 'manual' for a basic tutorial on what to do.")
    
    # detector = ai.AI(
    #     int(util.get_setting("batch_size")),
    #     int(util.get_setting("training_steps")),
    #     float(util.get_setting("learning_rate")),
    #     bool(util.get_setting("trained"))
    # )
    # detector.set_basepath(util.get_setting("base_path"))
    
    while True:
        query = input(">>> ")
        cmd_list = query.lower().split()
        if len(cmd_list) == 0:
            continue
        cmd = cmd_list[0]

        if cmd == "help":
            _help.main()
        elif cmd == "clear" or cmd == "cls":
            util.clear()
        elif cmd == "exit":
            util.exit_routine()
        elif cmd == "manual":
            _manual.main()
        elif cmd == "change-setting":
            _change_setting.main()
        elif cmd == "get-setting":
            _get_setting.main()
        elif cmd == "add-segment":
            _add_segment.main()
        elif cmd == "rm-segment":
            _rm_segment.main()
        elif cmd == "train":
            _train.main()
        elif cmd == "eval":
            _eval.main()
        else:
            print("ERROR: Could not recongnize command: '" + cmd + "'.")
