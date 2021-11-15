import os

def find_config_recursively(file_name):
    root_limit = os.path.expanduser('~')
    found_file = {'f': None}
    def find(here):
        file_names = os.listdir(here)
        if file_name in file_names:
            found_file['f'] = os.path.join(here, file_name)
            return True
        return False

    here = os.getcwd()
    while(not find(here)):
        if here == root_limit:
            return False
        here = os.path.abspath(os.path.join(here, '..'))
    return found_file['f']


found_file = find_config_recursively('.profile')
print(found_file)






