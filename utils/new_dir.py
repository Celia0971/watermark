import os

def new_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
