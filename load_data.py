import pandas as pd # type: ignore
import os

root_path = './dataset/'
data_directories = []

def get_files_from_dir(dir):
    directories = []
    files = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        files.append(filenames)
        directories.extend(dirnames)

    for dirnames in directories :
        data_directories.append(dirnames)
    return files

def get_file_name(file):
    return os.path.splitext(file)[0]

def get_device_name_and_feature(filename):
    device_name = ''
    device_feature = ''
    for c in filename:
        if c.islower():
            device_name += c
        else:
            device_feature += c
    return device_name, device_feature

def load_data(files):
    dataframes = []
    i = -1
    for dir in files:
        s = []
        for file in dir:
            file_path = root_path + data_directories[i] + "/" + file
            df = pd.read_csv(file_path)
            file_name = get_file_name(file)[:-2]
            device_name, device_feature = get_device_name_and_feature(file_name)
            df['device_name'] = device_name
            df['device_feature'] = device_feature
            s.append(df)
        if(i >= 0):
            dataframes.append(pd.concat(s))
        i += 1
    return dataframes

        
    