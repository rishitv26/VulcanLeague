#
# Main file that runs tensorflow network. Run this in server.
#
from util import *

def main():
    print("Welcome to the VLAE (Vulcan League AI Engine)")
    print("type 'start' to begin extraction.")

def first_setup():
    print("Welcome to the VLAE (Vulcan League AI Engine)")
    

try:
    # import tenserflow:
    import tensorflow
except:
    # if import fails, install tenserflow: 
    import pip
    
    pip.main(["install", "--upgrade", "pip"])
    pip.main(["install", "tensorflow"])
    
    
if __name__ == "__main__":
    clear()
    file = open("mem.txt", 'r')
    if file.read() == "":
        first_setup()
    else:
        main()
