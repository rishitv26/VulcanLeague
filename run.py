from config import Config
from itertools import groupby
import util
import ai as ai
import _help, _manual, _change_setting, _get_setting, _add_segment, _rm_segment, _eval, _train

def dispatch(cmd_list, config, detector):
    """Execute a single parsed command. Returns False if the shell should exit."""
    if len(cmd_list) == 0:
        return True

    cmd = cmd_list[0]

    if cmd == "help":
        _help.main(cmd_list, config, detector)
    elif cmd == "clear" or cmd == "cls":
        util.clear()
    elif cmd == "exit":
        config.save()
        return False
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

    return True


def main():
    print("Welcome to the VLAE (Vulcan League AI Engine) 2.0.1")
    print("type 'help' to see the list of commands.")
    print("type 'manual' for a basic tutorial on what to do.")

    config = Config()
    detector = ai.AI(
        int(config.get("batch_size")),
        int(config.get("training_steps")),
        float(config.get("learning_rate")),
    )
    detector.set_basepath(config.get("base_path"))

    while True:
        query    = input(">>> ")
        cmd_list = query.split()

        if len(cmd_list) == 0:
            continue

        sub_commands = [
            list(g)
            for k, g in groupby(cmd_list, key=lambda x: x == "&&")
            if not k
        ]

        for sub in sub_commands:
            should_continue = dispatch(sub, config, detector)
            config.save()
            if not should_continue:
                return