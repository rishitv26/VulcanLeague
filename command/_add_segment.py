
def main():
    if len(cmd_list) != 2:
        illegal_argument(2)
    else:
        try:
            shutil.copytree(cmd_list[1], 'data/test')
        except FileExistsError:
            segment_exists()