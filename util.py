#
# Includes utility functions... contains trivial tasks like user input
#
import os
import platform
import re

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
        for i in range(len(data)):
            data[i].replace('\n', '')
            
        SETTINGS[data[0]] = data[1]
    
    file.close()
    
    for key in SETTINGS:
        SETTINGS[key].replace('\n', '')

def modify_setting(setting: str, new_data: str):
    SETTINGS[setting] = new_data

def get_setting(setting: str):
    print(SETTINGS)
    if not (setting in SETTINGS):
        print("ERROR: setting '" + setting + "' does not exist.")
        return None
    return SETTINGS[setting]

def save_settings():
    file = open("mem.txt", 'w')
    file.write("")
    file.close()
    file = open("mem.txt", "a")
    for key in SETTINGS:
        file.write(key + "=" + SETTINGS[key].rstrip('\n') + "\n")
    file.close()

def exit_routine():
    # special routine to exit.
    save_settings()
    exit(0)
