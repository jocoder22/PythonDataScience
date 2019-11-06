def print2(*args):
    for memb in args:
        print(memb.head(), memb.info, memb.describe(), sep = '\n\n', end = '\n\n')