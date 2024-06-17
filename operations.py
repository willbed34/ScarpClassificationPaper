import cv2
import numpy as np
import os
import re
import csv
import constants
import matplotlib.pyplot as plt

#
#
#
#CONVERSION
def get_conversion_factor(pixel_amount, meter_amount):
    #using measurement of pixels in the wall height along with known height in meters
    conversion_factor = np.round(meter_amount / pixel_amount, 5)
    return conversion_factor

#
#
#
#PARSING
def find_directory_with_pattern(filepath):
    target_underscore_count = constants.TRIAL_PATTERN.count('_')
    root = os.path.dirname(filepath)

    for dirpath, dirnames, filenames in os.walk(root):
        for dir_name in dirnames:
            if dir_name.count('_') == target_underscore_count:
                return dir_name

    return None


#
#
#
#MASKS/PREPROCESSING
def remove_sparse_pixels(image):
    kernel = np.ones((3, 3), np.uint8)
    # blurred = cv2.GaussianBlur(image, kernel, 0)
    # dilated = cv2.dilate(blurred, kernel, iterations=1)
    eroded = cv2.erode(image, kernel, iterations=1)

    return eroded
def remove_ruler(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper threshold for red color in BGR
    lower_red = np.array([0, 100, 100])  # Lower threshold for red hue
    upper_red = np.array([150, 255, 255])  # Upper threshold for red hue

    # Create a mask for red pixels
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Apply the mask to the original image
    image_without_red = cv2.bitwise_and(image, image, mask=red_mask)
    return image_without_red

def keep_reds(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper threshold for red color in HSV
    lower_red = np.array([0, 100, 100])  # Lower threshold for red hue
    upper_red = np.array([10, 255, 255])  # Upper threshold for red hue

    # Create a mask for red pixels
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Apply the mask to the original image
    image_without_red = cv2.bitwise_and(image, image, mask=red_mask)
    return image_without_red

def extract_blues(image):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    extracted_image = cv2.bitwise_and(image, image, mask=blue_mask)

    return extracted_image


def find_leftmost_rightmost_pixel(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the leftmost and rightmost non-zero pixels
    leftmost_pixel = None
    rightmost_pixel = None

    for y in range(image.shape[0]):
        # Find the first non-zero pixel from the left
        leftmost_nonzero = np.argmax(gray[y, :] > 0)
        if leftmost_nonzero > 0 and (leftmost_pixel is None or leftmost_nonzero < leftmost_pixel[0]):
            leftmost_pixel = (leftmost_nonzero, y)

        # Find the first non-zero pixel from the right
        rightmost_nonzero = np.argmax(np.flip(gray[y, :]) > 0)
        rightmost_nonzero = image.shape[1] - rightmost_nonzero - 1
        if rightmost_nonzero < image.shape[1] - 1 and (rightmost_pixel is None or rightmost_nonzero > rightmost_pixel[0]):
            rightmost_pixel = (rightmost_nonzero, y)

    if leftmost_pixel is not None and rightmost_pixel is not None:
        return leftmost_pixel[0], rightmost_pixel[0]

def reduce_brightness(image, factor = 0.5):
    """
    Reduces the brightness of an image by a given factor.

    Args:
        image (numpy.ndarray): Input image.
        factor (float): Brightness reduction factor. Values between 0 and 1 reduce brightness, while values greater than 1 increase brightness.

    Returns:
        numpy.ndarray: Image with reduced brightness.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV color space
    hsv_image[..., 2] = hsv_image[..., 2] * factor  # Scale the V (value) channel by the brightness reduction factor
    processed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)  # Convert the modified image back to BGR color space
    return processed_image

#TODO
def extract_color(image, lower_bound, upper_bound):
    
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result
#
#
#
#EXTRACTING POINTS
def find_highest_point(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find non-black pixels
    non_black_pixels = np.where(gray > 0)
    # Get the maximum y-coordinate from the non-black pixels
    high_y = np.min(non_black_pixels[0])

    # Find the corresponding x-coordinate for the maximum y-coordinate
    high_x = non_black_pixels[1][np.argmin(non_black_pixels[0])]
    return (high_x, high_y)

def find_lowest_point(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find non-black pixels
    non_black_pixels = np.where(gray > 0)
    # Get the maximum y-coordinate from the non-black pixels
    low_y = np.max(non_black_pixels[0])
    # Find the corresponding x-coordinate for the maximum y-coordinate
    low_x = non_black_pixels[1][np.argmax(non_black_pixels[0])]
    return (low_x, low_y)

def find_top_left_point(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the indices of non-zero (non-black) pixels
    non_zero_indices = np.nonzero(gray)

    # Get the leftmost non-zero pixel coordinates
    top_left_x = np.min(non_zero_indices[1])
    #adding this padding
    padding = 5
    while True:
        try:
            top_left_y = non_zero_indices[0][np.where(non_zero_indices[1] == top_left_x + padding)][0]
        except Exception as e:
            padding += 1
        else:
            break 


    return (top_left_x, top_left_y)

def find_bottom_left_point(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the indices of non-zero (non-black) pixels
    non_zero_indices = np.nonzero(gray)

    # Get the leftmost non-zero pixel coordinates
    bottom_left_x = np.min(non_zero_indices[1])
    #adding this padding
    padding = 5
    while True:
        try:
            bottom_left_y = non_zero_indices[0][np.where(non_zero_indices[1] == bottom_left_x + padding)][-1]
        except Exception as e:
            padding += 1
        else:
            break

    return (bottom_left_x, bottom_left_y)

def find_top_right_point(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the indices of non-zero (non-black) pixels
    non_zero_indices = np.nonzero(gray)

    # Get the rightmost non-zero pixel coordinates
    top_right_x = np.max(non_zero_indices[1])
    #adding this padding
    padding = 5
    while True:
        try:
            top_right_y = non_zero_indices[0][np.where(non_zero_indices[1] == top_right_x - padding)][0]
        except Exception as e:
            padding += 1
        else:
            break


    return (top_right_x, top_right_y)

def find_bottom_right_point(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the indices of non-zero (non-black) pixels
    non_zero_indices = np.nonzero(gray)

    # Get the rightmost non-zero pixel coordinates
    bottom_right_x = np.max(non_zero_indices[1])
    #adding this padding
    padding = 5
    while True:
        try:
            bottom_right_y = non_zero_indices[0][np.where(non_zero_indices[1] == bottom_right_x - padding)][-1]
        except Exception as e:
            padding += 1
        else:
            break
    return (bottom_right_x, bottom_right_y)


def detect_nonzero_below_threshold(image, threshold_y, direction = 'left'):
    """
    Detect the first non-zero pixel below a specified threshold row in an image.

    Parameters:
        image (numpy.ndarray): The input image.
        threshold_y (int): The threshold row index.
        direction (str): The direction of pixel scanning. Supported values: 'left', 'right'.
                         Default is 'left'.

    Returns:
        tuple: A tuple containing the column index and row index of the first non-zero pixel below the threshold.
               If no non-zero pixel is found, it returns None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the height and width of the image
    height, width = gray.shape

    # Define the x-range based on the direction
    if direction == 'left':
        x_range = range(width)
    elif direction == 'right':
        x_range = range(width - 1, -1, -1)
    else:
        raise ValueError("Invalid direction. Supported values: 'left', 'right'")

    # Iterate through the image row-wise based on the specified direction
    for y in range(threshold_y, height):
        for x in x_range:
            # Check if the pixel is non-zero
            if gray[y, x] != 0:
                return (x, y)

    # If no non-zero pixel is found, return None
    return (-1,-1)

def find_first_nonzero_pixel(image, row_index):
    """
    Find the first non-zero pixel in a specified row of an image.

    Parameters:
        image (numpy.ndarray): The input image.
        row_index (int): The index of the row to search.

    Returns:
        tuple: A tuple containing the column index and row index of the first non-zero pixel.
               If no non-zero pixel is found, it returns None.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the specified row
    row = gray[row_index, :]
    # Find the index of the first non-zero pixel
    nonzero_indices = np.nonzero(row)[0]

    if len(nonzero_indices) > 0:
        return (nonzero_indices[0], row_index)
    else:
        return (-1,-1)
    
def find_last_nonzero_pixel(image, row_index):
    """
    Find the last non-zero pixel in a specified row of an image.

    Parameters:
        image (numpy.ndarray): The input image.
        row_index (int): The index of the row to search.

    Returns:
        tuple: A tuple containing the column index and row index of the last non-zero pixel.
               If no non-zero pixel is found, it returns None.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the specified row
    row = gray[row_index, :]

    # Find the indices of non-zero pixels
    nonzero_indices = np.nonzero(row)[0]

    if len(nonzero_indices) > 0:
        return (max(nonzero_indices), row_index)
    else:
        return (-1,-1)
    
def detect_furthest_blue(image, cutoff, direction):
    blues = extract_blues(image)
    cv2.imshow('img', blues)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    # Remove the bottom 100 pixels
    if cutoff > 0:
        blues = blues[:cutoff, :]
    try:
        eroded = remove_sparse_pixels(blues)
    except:
        print("an error occurred")
        return (-1,-1)
    cv2.imshow('img', eroded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    try:
        if direction == "right":
            pixel = find_top_right_point(eroded)  
        elif direction == "left":
            pixel = find_top_left_point(eroded)
    
    except:
        print("an error occurred")
        return (-1,-1)
    return pixel

def find_hanging_wall(x_init, vertical_uplift, fault_dip):
    angle_rad = np.deg2rad(fault_dip)

    # Calculate the x-value using trigonometry
    x_final = x_init + vertical_uplift / np.tan(angle_rad)

    return round(x_final)

def calculate_foot_wall_x(bottom_right_point, pixel_to_m):
    new_x = bottom_right_point[0] - constants.FAULT_START_FROM_RIGHT_M / pixel_to_m
    return (round(new_x), bottom_right_point[1])

def check_column(image, start_point, number_of_rows, threshold = 0.5):
    # Get the specified column from the image
    column_pixels = image[start_point[1]:start_point[1]+number_of_rows, start_point[0]]

    # Count the number of non-black pixels in the column
    non_black_pixels = cv2.countNonZero(column_pixels)

    # Calculate the percentage of non-black pixels
    percentage = (non_black_pixels / (number_of_rows))

    # Check if the percentage of non-black pixels meets the threshold
    if percentage >= threshold:
        return True
    else:
        return False

def check_if_overhang(image, starting_point, number_of_rows, required_black_pixels):
    #TODO fix this, right now it isn't great when there is a thick hanging wall
    #   - POTENTIALLY use 1D transform to see if there is a certain dropoff? ask kristen
    #   - OR use only blue pixels? IDK
    # Get the column of pixels below the starting row
    column = image[starting_point[1]:starting_point[1] + number_of_rows, starting_point[0]]

    # Initialize counters
    consecutive_black_pixels = 0
    max_consecutive_black_pixels = 0

    # Iterate through the column
    for pixel in column:
        if np.array_equal(pixel, [0, 0, 0]):  # Black pixel
            consecutive_black_pixels += 1
            max_consecutive_black_pixels = max(max_consecutive_black_pixels, consecutive_black_pixels)
        else:
            consecutive_black_pixels = 0

    # Check if the maximum number of consecutive black pixels meets the requirement
    if max_consecutive_black_pixels >= required_black_pixels:
        return True
    else:
        return False
    
def check_if_simple_scarp(image, scarp_point):
    #checks by seeing if there are a certain amount of blues in the top few rows of hanging wall
    sub_image = image[scarp_point[1]:scarp_point[1] + 4, :scarp_point[0]]
    blues = extract_blues(sub_image)
    non_zero_count = np.count_nonzero(blues)
    # cv2.imshow("blues",blues)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    if non_zero_count > 100:
        return False
    return True
    # height, width, _ = blues.shape

    # # Calculate the total number of pixels
    # total_pixels = height * width
    # if non_zero_count / total_pixels > .1:
    #     return False
    # return True
def get_highest_light_blue(image):
    # Need to fix this so that the first colluvium past the hanging wall, not necessarily the highest
    #TODO Trim bottom edges of foot and hanging wall
    light_blues = extract_color(image, constants.LOWER_LIGHT_BLUE, constants.UPPER_LIGHT_BLUE)
    cv2.imshow("blues",light_blues)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    #TODO find rgb value of the key
    return find_highest_point(light_blues)


#
#
#
#VISUALIZATION
def box_and_caption_pixel(image, x_coord, y_coord, caption_direction = 'right', caption = None, box_color = (0, 255, 0)):
    box_width = constants.BOX_WIDTH
    start_x = x_coord - (box_width // 2)
    start_y = y_coord - (box_width // 2)
    end_x = start_x + box_width
    end_y = start_y + box_width
    
    if box_color is not None:
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), box_color, 2)
    if caption is None or caption == "":
        return image
    image = put_text_with_background(image, caption, (x_coord, y_coord), caption_direction)
    return image

def put_text_with_background(image, text, position, caption_direction):
    #NOTE:  TEXT is the coordinates of the bottom-left corner of the text string in the image. 
    # The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).

    thickness = 2
    background_color = (255, 255, 255) 
    padding = constants.BOX_WIDTH
    box_padding = 5
    # Determine the size of the text
    (text_width, text_height), _ = cv2.getTextSize(text, constants.FONT, constants.FONT_SCALE, thickness)

    text_padding = (0, 0)
    if caption_direction == 'right':
        text_padding = (padding, text_height // 2)
    elif caption_direction == 'left':
        text_padding = (-1 * text_width - padding, text_height // 2)
    elif caption_direction == 'above':
        text_padding = (-1 * text_width // 2, -1 * padding)
    elif caption_direction == 'below':
        text_padding = (-1 * text_width // 2, text_height + padding)

    position = (position[0] + text_padding[0], position[1] + text_padding[1])

    # Set the coordinates for the background rectangle
    background_top_left = position[0] - box_padding, position[1] - box_padding - text_height
    background_bottom_right = position[0] + text_width + box_padding, position[1] + box_padding

    

    # Draw the background rectangle
    cv2.rectangle(image, background_top_left, background_bottom_right, background_color, -1)

    # Put the text on the image
    cv2.putText(image, text, position, constants.FONT, constants.FONT_SCALE, constants.TEXT_COLOR, thickness=1, lineType=cv2.LINE_AA)

    return image

def place_line_segment(image, start_point, end_point, color):
    thickness = 2
    cv2.line(image, start_point, end_point, color, thickness)
    return image

def show_image(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1200, 700)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def make_line_plot(x_values, y_values, points = None):
    plt.figure(figsize=(12, 2))
    plt.plot(x_values, y_values, color='red')
    if points is not None:
        if isinstance(points, tuple):
            point_x, point_y = points
            if point_x >= 0:  # Check if the point's x value is greater than or equal to 0
                plt.plot(point_x, point_y, 'go')
        elif isinstance(points, list):
            valid_points = [(p[0], p[1]) for p in points if p[0] >= 0 and p[1] >= 0]
            point_x = [p[0] for p in valid_points]
            point_y = [p[1] for p in valid_points]
            plt.plot(point_x, point_y, 'go')
    plt.xlabel('Index')
    plt.ylabel('Height(px)')
    plt.title('Surface')
    plt.show()

def plot_dashed_line(image, start_point, end_point, color=(255, 0, 0)):
    thickness = 2 
    dash_length = 5
    gap_length = 10

    # Convert the start and end points to integers
    start_point = (int(start_point[0]), int(start_point[1]))
    end_point = (int(end_point[0]), int(end_point[1]))

    # Calculate the line length and direction
    line_length = int(np.linalg.norm(np.array(end_point) - np.array(start_point)))
    line_direction = (end_point[0] - start_point[0], end_point[1] - start_point[1])

    # Normalize the direction vector
    line_direction_normalized = np.array(line_direction) / np.linalg.norm(np.array(line_direction))

    # Calculate the number of dashes to draw
    num_dashes = line_length // (dash_length + gap_length)

    # Calculate the dash position increment
    remaining_length = line_length
    remaining_dashes = num_dashes
    dash_increment = line_direction_normalized * (dash_length + gap_length)

    # Draw the dashed line
    current_point = np.array(start_point)
    for _ in range(num_dashes):
        # Calculate the end point of the dash
        dash_length_actual = min(dash_length, remaining_length)
        dash_end = current_point + dash_length_actual * line_direction_normalized

        # Draw the dash using the cv2.line function
        cv2.line(image, tuple(current_point.astype(int)), tuple(dash_end.astype(int)), color, thickness)

        # Update the current point and remaining length for the next dash
        current_point = current_point + dash_increment
        remaining_length -= dash_length_actual
        remaining_dashes -= 1

    return image


#
#
#
#WRITING TO FILES

def parse_dir(dir_name):
    # Split the string on underscore
    split_string = dir_name.split('_')

    return split_string
def create_measurement_dict(fullpath, measurement_dict, photo_num, dir_size):
    basedir = os.path.dirname(fullpath)
    trial_name = os.path.basename(basedir) 
    file_to_write_to = "results/text_files/" + trial_name + str(photo_num) + "_measurements.txt"
    slip_amount = round(int(photo_num) / dir_size * 5, 4)
    with open(file_to_write_to, 'w') as opened:
        opened.write('Trial: ' + trial_name + '\n')
        opened.write('Photo ' + str(photo_num) + ' out of ' + str(dir_size) + ' corresponding to ' + str(slip_amount) + ' of slip\n')
        opened.write('\n')
        for key in measurement_dict.keys():
            measurement = measurement_dict[key]
            output_string = ""
            if type(measurement) == str:
                output_string = key + ": " + measurement + '\n'
            elif key != "Scarp_Dip":
                measurement = measurement
                output_string = key + ": " + str(measurement) + " meters \n"
            else:
                output_string = key + ": " + str(measurement) + " degrees \n"
            opened.write(output_string)

def write_row_to_csv(file, row, flag = 'a'):
    with open(file, flag, encoding='UTF8', newline='') as opened:
        writer = csv.writer(opened)
        # write the header
        writer.writerow(row)

def put_dict_in_csv(file, dict):
    row = []
    for key in constants.HEADER:
        val = dict[key]
        row.append(str(val))
    write_row_to_csv(file, row, 'a')

def get_sediment_strength(cohesion):
    output = ""
    weak = ["1.0E+05", "5.0E+05"]
    moderate = ["1.0E+06"]
    strong = ["15E+05", "1.5E+06", "2.0E+06", "20E+05"]
    if cohesion in weak:
        output = "Weak"
    elif cohesion in moderate:
        output = "Moderate"
    elif cohesion in strong:
        output = "Strong"
    else:
        output = "Not listed"
    return output

            



###TEST FUNCTIONS ###
def detect_lines(binary_image):
    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(binary_image, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Create a blank image to draw the lines
    line_image = np.zeros_like(binary_image)

    # Draw the lines on the blank image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)

    return line_image

from sklearn.linear_model import LinearRegression

def fit_line_to_pixels(binary_image):
    # Find the coordinates of non-zero pixels in the binary image
    nonzero_pixels = np.transpose(np.nonzero(binary_image))

    # Extract the X and Y coordinates
    X = nonzero_pixels[:, 1].reshape(-1, 1)
    Y = nonzero_pixels[:, 0]

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, Y)

    # Predict the Y coordinates based on the fitted model
    predicted_Y = model.predict(X)

    # Draw the lines on the original image
    image_with_lines = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)  # Convert binary image to BGR color space

    for i in range(len(X)):
        x = X[i][0]
        y = int(predicted_Y[i])

        # Draw a circle at each pixel position
        cv2.circle(image_with_lines, (x, y), 2, (0, 255, 0), -1)  # Green circle

    # Draw the regression line
    x1 = 0
    y1 = int(model.predict([[x1]]))
    x2 = binary_image.shape[1] - 1
    y2 = int(model.predict([[x2]]))
    cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red line

    return image_with_lines

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# def visualize_pca_components(image, num_components=2):
#     # Convert the binary image to grayscale
#     gray = image * 255

#     # Flatten the grayscale image
#     flattened = gray.reshape(-1)

#     # Perform PCA on the flattened image
#     pca = PCA(n_components=num_components)
#     pca.fit(flattened.reshape(-1, 1))

#     # Get the principal components from PCA
#     components = pca.components_

#     # Reshape the components to match the image shape
#     reshaped_components = components.reshape(num_components, *image.shape)

#     # Create a blank image for visualization
#     vis_image = np.zeros_like(image)

#     # Assign different pixel values to each component
#     for i, component in enumerate(reshaped_components, start=1):
#         vis_image += (component > 0) * i

#     # Display the resulting image
#     plt.imshow(vis_image, cmap='rainbow')
#     plt.axis('off')
#     plt.show()


def fit_lines_to_binary_image(binary_image):
    # Extract the coordinates of white pixels (non-zero values) from the binary image
    coordinates = np.argwhere(binary_image)

    # Separate x and y coordinates
    x_coords = coordinates[:, 1]
    y_coords = coordinates[:, 0]

    # Fit lines using linear regression
    line1 = LinearRegression().fit(x_coords.reshape(-1, 1), y_coords)
    line2 = LinearRegression().fit(x_coords.reshape(-1, 1), y_coords)

    # Calculate the line equations
    line1_slope = line1.coef_[0]
    line1_intercept = line1.intercept_
    line2_slope = line2.coef_[0]
    line2_intercept = line2.intercept_

    return line1_slope, line1_intercept, line2_slope, line2_intercept