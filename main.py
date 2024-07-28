#
# Main file that runs everything. Run this in server.
#

# depending on version and computer, program may take some time to start
print("Loading...")


import util

def main():
    #try:
        if util.is_installed():
            import run
            run.main()
        else:
            import install
            install.main()
    #except:
    #    exit(0)

main()

