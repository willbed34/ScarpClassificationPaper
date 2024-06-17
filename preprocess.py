import numpy as np
import os
#maybe only import one function
import operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
print("TensorFlow version:", tf. __version__) 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from skimage import io, transform
from os.path import exists
import time
import re
import cv2
import constants

#may need to change this part with introduction of full dataset
#example of full path
# "I:\dem\Homogeneous_Results\D3_coh1.0e+05_ten1.0e+05_dip20_FS0.5\video\contacts\movie_contacts0001.png"
#HETERO PATH:
# I:\dem\Heterogeneous_Results\D3_A_dip20_FS0.5\video\contacts\movie_contacts0001.png"

#UNIX/MAC
data_path_unix = r"/Users/william 1/Undergrad/research/HarvardSeismo/data"
#WINDOWS
data_path_windows = r"I:\dem\Homogeneous_Results"
#since we are doing hetero
data_path_windows = r"I:\dem\Heterogeneous_Results"
hetero_path = r"I:\dem\Heterogeneous_Results"
sub_folder = r"video\contacts"
file_prefix = "movie_contacts"
file_suffix = ".png"
#in order to map timestep to image


def get_data(num_imgs_ret, image_dict, trial_path, show_first_image = False):
    if os.name == "nt":
        dir_path = os.path.join(data_path_windows, trial_path, sub_folder)
    elif os.name == "posix":
        dir_path = os.path.join(data_path_unix, trial_path)
    try:
        sorted_files = sorted(filename for filename in os.listdir(dir_path) if filename.lower().endswith(".png"))
    except Exception as e:
        error_type = type(e).__name__
        print("An error of type of", error_type, "occurred while listing directory contents of ", trial_path)
        return 0
    start_time = time.time()
    #try hardcoded
    # sorted_files = sorted(os.listdir(dir_path))
    last_file = sorted_files[-1]
    # print("Last file: ", last_file)
    #CHANGE
    start = 1 #ignore the basic images w little motion
    stop = int(re.findall(r'\d+', last_file)[0])
    filenum_list = np.linspace(start, stop, num_imgs_ret, endpoint = True)
    filenum_list = np.around(filenum_list).astype(int)

    for filenum in filenum_list:
        four_char_filenum = '{:04d}'.format(filenum)
        full_file_name = file_prefix + four_char_filenum + file_suffix
        fullpath = os.path.join(dir_path, full_file_name)
        if(exists(fullpath)):
            #original size: (2160, 3840, 3)
            # print("full path: ", fullpath)
            image_array = cv2.imread(fullpath)
            if show_first_image:
                operations.show_image(image_array, "Original Image")
            image_array = image_array[constants.LOWER_Y_CROP:constants.UPPER_Y_CROP, constants.LOWER_X_CROP:constants.UPPER_X_CROP]
            image_array = operations.remove_ruler(image_array)
            left_crop, right_crop = operations.find_leftmost_rightmost_pixel(image_array)
            #makes it centered
            cushion = min(constants.MAX_CUSHION, left_crop - 5, right_crop - 5)
            image_array = image_array[:, left_crop - cushion:right_crop + cushion]
            if show_first_image:
                operations.show_image(image_array, "Image with no Ruler")
            show_first_image = False
            # print("new size: ", image_array.shape)
            # image_dict[fullpath] = np.array(image_array)
            image_dict[fullpath] = image_array
    end_time = time.time()
    execution_time = end_time - start_time
    # print("Time to get ", num_imgs_ret," images: ", execution_time, "seconds. On average: ", str(execution_time / num_imgs_ret), " s/img")
    return stop


def get_directories(directory):
    """
    Retrieves the names of directories directly under a specific directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        list: A list of directory names.

    Raises:
        Any specific exceptions that may occur during the execution.
    """
    directories = []

    # Iterate over each entry in the directory
    with os.scandir(directory) as entries:
        for entry in entries:
            # Check if the entry is a directory
            if entry.is_dir():
                # Append the directory name to the list
                directories.append(entry.name)

    return directories

def get_data_from_trials(num_imgs_ret, use_all_trials = False, list_of_trials = [], show_first_image = False):
    """
    Retrieves image data from multiple trials. If desire to use all trials, s, use os

    Args:
        num_imgs_ret (int): Number of images to retrieve from each trial.
        resize_shape (tuple): Desired shape for resizing the images.
        use_all_trials (bool): whether or not to use all trials
        list_of_trials (list): List of trial names or identifiers.

    Returns:
        image_dict: A dictionary mapping file paths to image arrays.
        trial_size_dict: A dictionary mapping directories to the number of images they contain. This will be used
        calculate slip as a function of image number
    """
    image_dict = {}
    trial_size_dict = {}
    if use_all_trials:
        if os.name == "nt":
            list_of_trials = get_directories(r"I:\dem\Heterogeneous_Results")
        elif os.name == "posix":
            list_of_trials = get_directories(r"/Users/william 1/Undergrad/research/HarvardSeismo/data")
    for trial in list_of_trials:
        trial_size = get_data(num_imgs_ret, image_dict, trial, show_first_image)
        trial_size_dict[trial] = trial_size
    return image_dict, trial_size_dict


