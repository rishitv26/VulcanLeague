

def main():
    print("clear/cls - clear the console output")
    print("exit - stop the VLAE routine")
    print("manual - basic instructions for running VLAE")
    print("change-setting <setting> <new value> - changes a setting variable manually.")
    print("get-setting <setting> - gets the value of a given setting.")
    print("add-segment <dir> - copy a segment into 'test' subfolder to run ink detetction on.")
    print("rm-segment <name> - delete a segment from 'test' subfolder")
    print("train - train the model with data configured by training_data setting. Must be a comma seperated list containing only 1, 2, or 3; NO REPEATS.")
    print("eval - Run the model on data from 'test' subfolder.")