import pandas as pd # type: ignore
import os

path = './dataset/'
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

def load_data(files):
    dataframes = []
    i = -1
    for dir in files:
        for file in dir:
            file_path = path + data_directories[i] + "/" + file
            df = pd.read_csv(file_path)
            df['device_name'] = get_file_name(file)[:-2]
            dataframes.append(df)
        i += 1
    return pd.concat(dataframes)

print(load_data(get_files_from_dir(path)))

        
    