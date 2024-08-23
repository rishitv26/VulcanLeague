
from config import Config
import util
import ai as ai
import _help, _manual, _change_setting, _get_setting, _add_segment, _rm_segment, _eval, _train

def main():
    print("Welcome to the VLAE (Vulcan League AI Engine) 1.0.0")
    print("type 'help' to see the list of commands.")
    print("type 'manual' for a basic tutorial on what to do.")
        
    config = Config()
    trained = True
    if config.get("trained") == "true":
        trained = False
    
    detector = ai.AI(
        int(config.get("batch_size")),
        int(config.get("training_steps")),
        float(config.get("learning_rate")),
        trained
    )
    detector.set_basepath(config.get("base_path"))
    
    while True:
        query = input(">>> ")
        cmd_list = query.split()
        if len(cmd_list) == 0:
            continue
        cmd = cmd_list[0]

        if cmd == "help":
            _help.main(cmd_list, config, detector)
        elif cmd == "clear" or cmd == "cls":
            util.clear()
        elif cmd == "exit":
            config.save()
            exit(0)
        elif cmd == "manual":
            _manual.main(cmd_list, config, detector)
        elif cmd == "change-setting":
            _change_setting.main(cmd_list, config, detector)
        elif cmd == "get-setting":
            _get_setting.main(cmd_list, config, detector)
        elif cmd == "add-segment":
            _add_segment.main(cmd_list, config, detector)
        elif cmd == "rm-segment":
            _rm_segment.main(cmd_list, config, detector)
        elif cmd == "train":
            _train.main(cmd_list, config, detector)
        elif cmd == "eval":
            _eval.main(cmd_list, config, detector)
        else:
            print("ERROR: Could not recongnize command: '" + cmd + "'.")

        config.save()


