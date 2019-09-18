import cv2
import os
import glob
import random
from shutil import copyfile
import sys
from sys import exit
from os.path import join, basename, dirname, exists

root_folder = 'superbeings'

def extract_frames(movie_file):
    folder = movie_file.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(movie_file)
    success, image = vidcap.read()

    if not success:
        exit(2)

    dir_path = os.path.join(root_folder, folder, 'all')

    if not exists(dir_path):
        os.makedirs(dir_path)
    
    count = 0
    while success:
        path = join(dir_path, 'frame{}.jpg'.format(count))
        cv2.imwrite(path, image)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 250))  # every 1/4 sec
        success, image = vidcap.read()

        if not success:
            break

        count += 1
    print('Images count', count)
    return dir_path

def copy_files(source_filenames, dest_path):
    for source_file in source_filenames:
        dest_file = join(dest_path, basename(source_file))

        try:
            copyfile(source_file, dest_file)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(2)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(2)
    

def create_train_val_test(all_folder):
    image_files = []
    for image_file in glob.glob(all_folder + '/*jpg'):
        image_files.append(image_file)
    random.shuffle(image_files)
    
    split_1 = int(0.7 * len(image_files))
    split_2 = int(0.9 * len(image_files))
    train_filenames = image_files[:split_1]
    print('Total train files', len(train_filenames))
    val_filenames = image_files[split_1:split_2]
    print('Total val files', len(val_filenames))
    test_filenames = image_files[split_2:]
    
    train_path = join(root_folder, 'train',
                      basename(dirname(all_folder)))
    val_path = join(root_folder, 'val',
                    basename(dirname(all_folder)))
    print(train_path)
    print(val_path)

    if not exists(train_path):
        os.makedirs(train_path)
    copy_files(train_filenames, train_path)

    if not exists(val_path):
        os.makedirs(val_path)
    copy_files(val_filenames, val_path)

    
dir_path = extract_frames('videos/superman.MOV')
create_train_val_test(dir_path)
dir_path = extract_frames('videos/catwoman.MOV')
create_train_val_test(dir_path)

