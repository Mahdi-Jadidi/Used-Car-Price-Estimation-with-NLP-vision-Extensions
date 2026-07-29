import os
def get_main_dir():
    return os.path.dirname(os.path.abspath(__file__))
print(get_main_dir())