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

# clear the screen
def clear():
    if is_windows():
        os.system("cls")
    else:
        os.system("clear")
