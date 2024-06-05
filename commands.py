#
# Contains the interpreter for the commands to run...
#
import util

def main():
    while True:
        query = input(">>> ")
        if query.lower() == "help":
            # Give a list of each command and what it does.
            pass
        elif query.lower() == "clear" or query.lower() == "cls":
            util.clear()
        elif query.lower() == "exit":
            util.exit_routine()
        
        #### TODO....
