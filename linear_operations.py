import cv2
import numpy as np
import os
import re
import csv
import constants
import operations
from sklearn.metrics import r2_score

from scipy.signal import savgol_filter

def savitzky_golay_smooth(signal, window_size, polynomial_order):
    smoothed_signal = savgol_filter(signal, window_size, polynomial_order)
    return smoothed_signal

def get_highest_pixel_rows(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the height and width of the image
    height, width = gray.shape

    # Initialize an empty list to store the highest pixel rows
    highest_pixel_rows = []

    # Iterate through each column
    for col in range(width):
        # Get the column
        column = gray[:, col]

        # Find the indices of non-zero pixels in the column
        nonzero_indices = np.nonzero(column)[0]

        # Check if there are any non-zero pixels in the column
        if len(nonzero_indices) > 0:
            # Find the row index of the highest non-zero pixel
            highest_row = np.min(nonzero_indices)
            highest_pixel_rows.append(height - highest_row)
        else:
            # If no non-zero pixels found, append None
            highest_pixel_rows.append(0)

    return highest_pixel_rows

def trim_zeros(lst):
    # Find the index of the first non-zero element from the left
    first_nonzero_index = next((i for i, val in enumerate(lst) if val != 0), None)

    if first_nonzero_index is None:
        # If all elements are zeros, return an empty list and 0 leading zeros
        return [], 0

    # Find the index of the first non-zero element from the right
    last_nonzero_index = next((i for i, val in enumerate(reversed(lst)) if val != 0), None)

    # Calculate the index from the right side
    last_nonzero_index = len(lst) - last_nonzero_index - 1

    # Trim the list from both ends using the first and last non-zero indices
    trimmed_list = lst[first_nonzero_index:last_nonzero_index + 1]

    # Calculate the number of leading zeros
    num_leading_zeros = first_nonzero_index

    return trimmed_list, num_leading_zeros

def find_deviation(array, target, difference, traverse_left=True):
    if traverse_left:
        for i in range(len(array)):
            if abs(array[i] - target) >= difference:
                return (i-1, int(array[i-1]))
    else:
        for i in range(len(array) - 2, -1, -1):
            if abs(array[i] - target) >= difference:
                return (i+1, int(array[i+1]))
    return (-1, -1)

def find_rightmost_pixel(image, begin_index, end_index):
    
    # Load the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the rightmost pixel in each row
    rightmost_pixels = []
    for row in gray:
        rightmost_pixel = np.max(np.where(row > 0, np.arange(len(row)), -1))
        rightmost_pixels.append(rightmost_pixel)

    return rightmost_pixels[begin_index:end_index], begin_index, end_index

def find_leftmost_pixel(image):
    begin_index = operations.find_highest_point(image)[1]
    end_index = operations.find_top_left_point(image)[1] - 5
    # Load the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the leftmost non-zero pixel in each row
    leftmost_pixels = []
    for row in gray:
        leftmost_pixel = np.min(np.where(row > 0, np.arange(len(row)), len(row)))
        leftmost_pixels.append(leftmost_pixel)

    return leftmost_pixels[begin_index:end_index], begin_index, end_index


def find_index_difference_less_than_amt(lst, amt):
    for i in range(len(lst) - 1):
        if abs(lst[i] - lst[i+1]) < amt:
            return (lst[i], i)
    return (-1,-1)

def find_index_difference_more_than_amt(lst, amt):
    for i in range(len(lst) - 2,0,-1):
        if abs(lst[i] - lst[i+1]) > amt:
            return (lst[i], i)
    return (-1,-1)

def find_decrease(lst, avg_length, threshold=0, return_list=False):
    n = len(lst)
    decrease_indices = []

    for i in range(n - (2 * avg_length - 1)):
        first_range = lst[i:i + avg_length]
        next_range = lst[i + avg_length:i + 2 * avg_length]

        first_avg = sum(first_range) / avg_length
        next_avg = sum(next_range) / avg_length

        if first_avg - next_avg >= threshold:
            if return_list:
                decrease_indices.append((i, lst[i]))
            else:
                return (i, lst[i])

    if return_list:
        return decrease_indices

    return (-1, -1)

def find_increase(lst, avg_length, threshold = 0):
    n = len(lst)
    for i in range(n - (2*avg_length - 1)):
        first_range = lst[i:i + avg_length]
        next_range = lst[i + avg_length:i + 2 * avg_length]

        first_avg = sum(first_range) / avg_length
        next_avg = sum(next_range) / avg_length

        if next_avg - first_avg >= threshold:
            return (i, lst[i])

    return (-1,-1)

def find_point_difference(signal, window_size, threshold, decrease_or_increase = True):
    n = len(signal)
    for i in range(n - window_size + 1):
        window = signal[i:i + window_size]
        max_diff = np.max(window) - np.min(window)
        max_index = np.argmax(window)
        min_index = np.argmin(window)
        if decrease_or_increase or max_index < min_index:
            if max_diff > threshold:
                index = i + max_index  # Index of the point with the maximum value in the window
                value = signal[index]
                return (index, int(value))
    return (-1, -1)

def find_element_with_threshold(y_values, threshold):
    # Get the rightmost value in the list
    rightmost_value = y_values[-1]
    
    # Iterate through the list in reverse order
    for i in range(len(y_values) - 2, -1, -1):
        if greater:
            if abs(rightmost_value - y_values[i]) > threshold:
                return (i, y_values[i])
    
    # If no element meets the condition, return None
    return -1

def get_squared_value(first_point, second_point, true_y_values, num_leading_zeros, top_begin_index, top_end_index): # Extract the y-values between dz_begin_1 and dz_end_1
    begin_index_r = int(first_point[0] - num_leading_zeros)
    end_index_r = int(second_point[0] - num_leading_zeros)
    y_values_range_r = true_y_values[begin_index_r:end_index_r]

    # Fit a straight line between first_point and second_point
    x_values_range_r = np.arange(begin_index_r, end_index_r)
    A_r = np.vstack([x_values_range_r, np.ones(len(x_values_range_r))]).T
    m_r, c_r = np.linalg.lstsq(A_r, y_values_range_r, rcond=None)[0]

    # Calculate the line of best fit
    line_of_best_fit = m_r * x_values_range_r + c_r

    x_values_cropped_1 = np.arange(top_begin_index, top_end_index)

    # # plt.plot(y_values_1, label='y_values_1')
    # plt.plot(x_values_cropped_1, true_y_values, label='true_y_values')
    # # undoing the indexing, took a bit to figure out but p sure its right 
    # plt.plot([first_point[0] - num_leading_zeros], [first_point[1]], marker='o', markersize=8, color="red", label='begin_dz_1')
    # plt.plot([second_point[0] - num_leading_zeros], [second_point[1]], marker='o', markersize=8, color="green", label='end_dz_1')
    # plt.plot(x_values_range_r, line_of_best_fit, label = "line of best fit")
    # plt.legend()
    # plt.show()

    # Calculate R-squared value
    r_squared_lobf = r2_score(y_values_range_r, line_of_best_fit)

    # print("R-squared value:", r_squared_lobf)
    return r_squared_lobf

#VIZ
def find_and_draw_rectangles(image, objects_to_remove = 0, min_area = 3000, max_area = float('inf'), max_height = float('inf')):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to create a binary image
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

    # Sort contours based on their areas (in descending order)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    #sets min boundary
    filtered_contours = []
    for contour in sorted_contours:
        # Find the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        bottom_edge = y + h
    
        # Check if the bottom edge of the contour is below the max height
        if bottom_edge <= max_height:
            filtered_contours.append(contour)

    # Draw rectangles around the contours
    objects_removed = 0
    for contour in filtered_contours:
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:  # Adjust the minimum contour area as needed, make a constant TODO
            #REMOVES HANGING AND FOOTWALL
            if objects_removed < objects_to_remove:
                objects_removed += 1
            else:
                return contour
    
    return None

def find_simple_dz_begin(image, scarp_point, fault_dip):
    bottom_right_point = operations.find_bottom_right_point(image)
    bottom_left_point = operations.find_bottom_left_point(image)
    foot_wall_point = operations.find_first_nonzero_pixel(image, bottom_right_point[1])

    #trigonometric calculation of hanging wall point
    vertical_uplift = bottom_right_point[1] - bottom_left_point[1]
    hanging_wall_x_theoretical = operations.find_hanging_wall(foot_wall_point[0], vertical_uplift, fault_dip)
    begin_dz = (hanging_wall_x_theoretical, bottom_left_point[1])
    #find highest blue point within 200 px to the left of the theoretical begin
    x_slice_begin = hanging_wall_x_theoretical - 200
    x_slice_end = hanging_wall_x_theoretical + 200
    extracted_blues = operations.extract_blues(image)[:, x_slice_begin:x_slice_end]
    extracted_blues = operations.remove_sparse_pixels(extracted_blues)
    highest_blue = operations.find_highest_point(extracted_blues)
    begin_dz = (highest_blue[0] + x_slice_begin, highest_blue[1])

    if begin_dz == None:
        begin_dz = (-1,-1)
    return begin_dz