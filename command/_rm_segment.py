
def main():
    if len(cmd_list) != 2:
        illegal_argument(2)
    else:
        try:
            shutil.rmtree('data/test/' + cmd_list[1])
        except FileNotFoundError:
            segment_doesnt_exist()