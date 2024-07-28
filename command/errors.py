from .. import util
from AI import ai

def illegal_argument(x: int):
    print("ERROR: Invalid Argument Call")
    print(f"Expected {x} arguments.")

def segment_exists():
    print("ERROR: This segment already exists with the same name.")

def segment_doesnt_exist():
    print("ERROR: This segment does not exist.")

def setting_doesnt_exist():
    print("ERROR: This setting has not been defined or doesnt exist.")

def data_not_downloaded():
    print("ERROR: The data required to train the model is not yet installed.\n")
    reply = util.ask("Install them now?> ")
    if reply:
        try:
            ai.download_data()
        except:
            print("There was an error in downloading the data, are you connected to the internet?")
            print("If error persists, contact the developers or start an issue at our repository.")
            return True
            
        return False
    else:
        return True
    