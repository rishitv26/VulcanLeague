#
# Contains the interpreter for the commands to run...
# 
import util
import shutil

def illegal_argument(x: int):
    print("ERROR: Invalid Argument Call")
    print(f"Expected {x} arguments.")

def segment_exists():
    print("Error: This segment already exists with the same name.")

def segment_doesnt_exist():
    print("Error: This segment does not exist.")

def main():
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
                util.modify_setting(cmd[1], cmd[2])
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
        
        #### TODO....
