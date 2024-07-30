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

# prompts the user to press enter to continue.   
def pause():
    print("Press enter to continue...")
    input()

# asks a user from a given prompt. will return true if user said yes, and false if no.
def ask(string: str):
    ans = input(string)
    if ans.upper() == "YES" or ans.lower() == "y":
        return True
    else:
        return False

# Checks if installation of VLAE has been done properly, and return true or false.
def is_installed():
    # TODO: explicitly check if libraries are working, and if not, rerun installation and fix issues.
    return os.path.exists("config.txt")