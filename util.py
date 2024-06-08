#
# Includes utility functions... contains trivial tasks like user input
#
import os
import platform

# check if platform is windows:
def is_windows():
    if platform.system() == "Windows":
        return True
    else:
        return False

def is_mac():
    if platform.system() == "Darwin":
        return True
    else:
        return False

# clear the screen
def clear():
    if is_windows():
        os.system("cls")
    else:
        os.system("clear")
    
def pause():
    input()

def ask(string: str):
    ans = input(string)
    if ans.lower() == "yes" or ans.lower() == "y":
        return True
    else:
        return False

SETTINGS = {}

def load_settings():
    file = open("mem.txt", "r")
    for line in file.readlines():
        data = line.split("=")
        SETTINGS[data[0]] = data[1]
    file.close()

def modify_setting(setting: str, new_data: str):
    SETTINGS[setting] = new_data

def get_setting(setting: str):
    return SETTINGS[setting]

def save_settings():
    file = open("mem.txt", 'w')
    lines = []
    for key in SETTINGS:
        lines.append(key + "=" + SETTINGS[key] + "\n")
    file.writelines(lines)
    file.close()

def exit_routine():
    # special routine to exit.
    save_settings()
    exit(0)
