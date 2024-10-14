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
        if c.islower() or c.isnumeric():
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
            home_automation = ["yutron1", "yutron2","teckin2","teckin1","smartboard","roomba","philipshue","heimvisionlamp", "heimvision","gosundcenter","globelamp","eufyhomebase","atomicoffeemaker","amazonplug"]
            camera = ["simcam", "netatmocam","nestcam","homeeyecam","heimvisioncam","dlinkcam","boruncam","arloqcam","arlobasestationcam","amcrestcam","luohecam"]
            audio = ["sonosone","nestmini","echostudio","echospot","echodot1","echodot2","echodot3"]
            file_path = root_path + data_directories[i] + "/" + file
            df = pd.read_csv(file_path)
            file_name = get_file_name(file)[:-2]
            device_name, device_feature = get_device_name_and_feature(file_name)
            df['device_name'] = device_name
            if device_name in home_automation:
                df['device_category'] = "home_automation"
            elif device_name in camera:
                df['device_category'] = "camera"
            elif device_name in audio:
                df['device_category'] = "audio"
            else:
                df['device_category'] = "none"
                print("set : none ("+device_name+")")
            s.append(df)
        if(i >= 0):
            dataframes.append(pd.concat(s))
        i += 1
    return dataframes

        
    