import os
import pickle
import json
from typing import List


def subdirs(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)


def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def check_dir(dir):
    if not os.path.exists(dir):
        mkdir(dir)


def mkdir(path):
    '''
    make new directory
    '''
    path=path.strip()
    path=path.rstrip("\\")
    path=path.rstrip("/")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print(path+' has been maken successful!')
        return True
    else:
        print(path+' exsisted.')
        return False

# def get_filename_number(filename, num):
#     # Extract numbers from filenames
#     return int(re.findall("\d+",filename)[num])

def get_case_number(filename):

    return int((filename.split('/')[-1].split('_')[1]))

def get_filename_info(filefullname):
    # Extract folder path, file name, file short name, file extension of file name
    (folderpath,filename) = os.path.split(filefullname)
    (shotname,extension) = os.path.splitext(filename)
    return folderpath, filename, shotname, extension
