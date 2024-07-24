
def main():
    if len(cmd_list) != 3:
        illegal_argument(3)
    else:
        util.modify_setting(cmd_list[1], cmd_list[2])