#
# Main file that runs everything. Run this in server.
#

# depending on version and computer, program may take some time to start
print("Loading...")


import util
import install
try:
    import run
except ModuleNotFoundError:
    install.main()


def main():
    if util.is_installed():
        run.main()
    else:
        install.main()

main()

