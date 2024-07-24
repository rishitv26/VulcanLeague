
def main():
    if len(cmd_list) != 2:
        illegal_argument(2)
    else:
        try:
            print(util.get_setting(cmd_list[1]))
        except KeyError:
            setting_doesnt_exist()