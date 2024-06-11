#
# Contains the interpreter for the commands to run...
# 
import util
import shutil
import os
try:
    import ai
except:
    print("WARNING: Dependencies (probably torch) not installed correctly.")
    print("press enter to acknoledge")
    util.pause()

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
    util.load_settings()
    while True:
        query = input(">>> ")
        cmd_list = query.lower().split()
        if len(cmd_list) == 0:
            continue
        cmd = cmd_list[0]

        if cmd == "help":
            # Give a list of each command and what it does.
            print("clear/cls - clear the console output")
            print("exit - stop the VLAE routine")
            print("manual - basic instructions for running VLAE")
            print("change-setting <setting> <new value> - changes a setting variable manually.")
            print("get-setting <setting> - gets the value of a given setting.")
            print("add-segment <dir> - copy a segment into 'test' subfolder to run ink detetction on.")
            print("rm-segment <name> - delete a segment from 'test' subfolder")
            print("train - train the model with data configured by training_data setting. Must be a comma seperated list containing only 1, 2, or 3; NO REPEATS.")
            print("eval - Run the model on data from 'test' subfolder.")
        elif cmd == "clear" or cmd == "cls":
            util.clear()
        elif cmd == "exit":
            util.exit_routine()
        elif cmd == "manual":
            print("1. Choose between training data combinations of the following: 1; 1,2; 1,2,3. DO NOT PUT ANYTHING ELSE")
            print("2. When chosen, set it as value for training_data setting using change-setting")
            print("3. train the model by train command")
            print("4. Add all segments using add-segment command for ink evaluation.")
            print("5. Remove any segments if neccessary.")
            print("6. run the ink evaluation by eval command")
            print("7. snaphot output and save it useful")
            print("8. Run it through letter detector. (TODO)")
            print("9. Profit!")
            pass
        elif cmd == "change-setting":
            if len(cmd_list) != 3:
                illegal_argument(3)
            else:
                util.modify_setting(cmd_list[1], cmd_list[2])
        elif cmd == "get-setting":
            if len(cmd_list) != 2:
                illegal_argument(2)
            else:
                try:
                    print(util.get_setting(cmd_list[1]))
                except KeyError:
                    setting_doesnt_exist()
        elif cmd == "add-segment":
            if len(cmd_list) != 2:
                illegal_argument(2)
            else:
                try:
                    shutil.copytree(cmd_list[1], 'data/test')
                except FileExistsError:
                    segment_exists()
        elif cmd == "rm-segment":
            if len(cmd_list) != 2:
                illegal_argument(2)
            else:
                try:
                    shutil.rmtree('data/test/' + cmd_list[1])
                except FileNotFoundError:
                    segment_doesnt_exist()
        elif cmd == "train":
            if not os.path.isdir("data"):
                reply = data_not_downloaded()
                if reply:
                    continue
            print("Preparing to train this model...\n")
            print("DO NOT let machine turn off or sleep during this process.")
            print("Close all other programs for best performance.")
            print("Press enter when you are ready to continue.")
            util.pause()
            util.clear()
            ai.load_model()
        elif cmd == "eval":
            print("Preparing for ink evaluation...\n")
            print("DO NOT let machine turn off or sleep during this process.")
            print("Close all other programs for best performance.")
            print("Press enter when you are ready to continue.")
            util.pause()
            util.clear()
            ai.eval_model()
        else:
            print("ERROR: Could not recongnize command: '" + cmd + "'.")
        
        #### TODO....
