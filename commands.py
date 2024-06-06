#
# Contains the interpreter for the commands to run...
# 
import util
import shutil

def illegal_argument(x: int):
    print("ERROR: Invalid Argument Call")
    print(f"Expected {x} arguments.")

def segment_exists():
    print("ERROR: This segment already exists with the same name.")

def segment_doesnt_exist():
    print("ERROR: This segment does not exist.")

def setting_doesnt_exist():
    print("ERROR: This setting has not been defined or doesnt exist.")

def main():
    util.load_settings()
    while True:
        query = input(">>> ")
        cmd_list = query.lower().split()
        cmd = cmd_list[0]

        if cmd == "help":
            # Give a list of each command and what it does.
            pass
        elif cmd == "clear" or cmd == "cls":
            util.clear()
        elif cmd == "exit":
            util.exit_routine()
        elif cmd == "manual":
            # Give a list of instructions to do.
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
                    shutil.rmtree('data/test/' + str(cmd_list[1]))
                except FileNotFoundError:
                    segment_doesnt_exist()
        else:
            print("ERROR: Could not recongnize command: '" + cmd + "'.")
        
        #### TODO....
