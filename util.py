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
    data_pattern = re.compile(r'^[^\s=]+=[^\s=]+\n$')
    file = open("mem.txt", "r")
    
    for line in file.readlines():
        if data_pattern.match(line):
            
            data = line.split("=")
            for i in range(len(data)):
                data[i].replace('\n', '')
                data[i].replace(' ', '')
            
            SETTINGS[data[0]] = data[1]
    
    file.close()

def modify_setting(setting: str, new_data: str):
    if not (setting in SETTINGS):
        print("ERROR: setting '" + setting + "' does not exist.")
        return None
    SETTINGS[setting] = new_data

def get_setting(setting: str):
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
        file.write(key + "=" + SETTINGS[key] + "\n")
    file.close()

def exit_routine():
    # special routine to exit.
    save_settings()
    exit(0)
