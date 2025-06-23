#
# Main file that runs everything. Run this in server.
#

# depending on version and computer, program may take some time to start
print("Loading...")


import util
import install



def main():
    if util.is_installed():
        try:
            import run
            run.main()
        except ModuleNotFoundError:
            install.main()
    else:
        install.main()

main()

