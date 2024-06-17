import matplotlib.pyplot as plt
import preprocess
import operations
import linear_operations
import cv2
import numpy as np
import os
import re
import constants
from constants import ScarpClassification
from PIL import Image
def interpolate_image(input_image_path, input_image, trial_size_dict_param, visualize_points = False, make_line_plots = False, hetero = False):
    #Create dictionary to store data, see constants for the keys
    pixel_measurements = {key: "NA" for key in constants.HEADER}

    #get trial name and how many images it has
    if os.name == "nt":
        trial_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(input_image_path))))
    elif os.name == "posix":
        trial_name = os.path.basename(os.path.dirname(input_image_path))
    trial_size = trial_size_dict_param[trial_name]

    #homo: D3_coh1.0e+05_ten1.0e+05_dip20_FS0.5
    #hete: D3_A_dip20_FS0.5
    #get number of the photo from the path name
    basefile = os.path.basename(input_image_path)
    photo_num = re.findall(r'\d+', basefile)[0]
    pixel_measurements["Trial"] = trial_name

    #parsing trial name to get the input parameters
    trial_characteristics = operations.parse_dir(trial_name)
    density = trial_characteristics[0][0]
    if density == 'D':
        density = "Dense"
    elif density == 'L':
        density = "Loose"
    else:
        density = "Medium"
    pixel_measurements["Density"] = density
    pixel_measurements["Depth"] = trial_characteristics[0][1:]

    if hetero:
        cohesion = trial_characteristics[1]
        pixel_measurements["Cohesion"] = cohesion
        fault_dip = int(trial_characteristics[2][3:])
        pixel_measurements["Fault_Dip"] = fault_dip
        pixel_measurements["Fault_Seed"] = trial_characteristics[3][2:]
    else:
        cohesion = trial_characteristics[1][3:].upper()
        pixel_measurements["Cohesion"] = cohesion
        pixel_measurements["Sediment_Strength"] = operations.get_sediment_strength(cohesion)
        fault_dip = int(trial_characteristics[3][3:])
        pixel_measurements["Fault_Dip"] = fault_dip
        pixel_measurements["Fault_Seed"] = trial_characteristics[4][2:]

    if int(photo_num) <= 3:
        pixel_measurements["Slip"] = 0
        return pixel_measurements
    else:
        slip_amount = round(int(photo_num) / (trial_size-3) * 5, 4)
    pixel_measurements["Slip"] = slip_amount
    annotated_image = np.copy(input_image)

    #get image dimensions
    height, width = annotated_image.shape[0], annotated_image.shape[1]

    #obtain boundaries of the walls, and other points from the image
    top_left_point = operations.find_top_left_point(annotated_image)
    bottom_left_point = operations.find_bottom_left_point(annotated_image)
    top_right_point = operations.find_top_right_point(annotated_image)
    bottom_right_point = operations.find_bottom_right_point(annotated_image)
    foot_wall_point = operations.find_first_nonzero_pixel(annotated_image, bottom_right_point[1])

    #early on where barely any change is detected
    if foot_wall_point[0] == -1:
        foot_wall_point = hanging_wall_point
    vertical_uplift = bottom_right_point[1] - bottom_left_point[1]
    hanging_wall_x_theoretical = operations.find_hanging_wall(foot_wall_point[0], vertical_uplift, fault_dip)
    hanging_wall_point = (hanging_wall_x_theoretical, bottom_left_point[1])
    highest_point = operations.find_highest_point(annotated_image)

    #obtain conversion factor
    height_of_wall_m = int(trial_characteristics[0][1:])
    pixel_to_m = operations.get_conversion_factor(bottom_right_point[1] - top_right_point[1], height_of_wall_m)
    pixel_measurements["Conversion_Factor"] = pixel_to_m


    classification = ""
    
    right_begin_index = highest_point[1]
    right_end_index = top_right_point[1] - 5
    rightmost_pixels_list, right_begin_index, right_end_index = linear_operations.find_rightmost_pixel(annotated_image, right_begin_index, right_end_index)

    #initialize these variables that will store object detection, and points of dropoff that determine collapse
    contour = None
    pr_decrease_loc = (-1,-1)
    mc_decrease_loc = (-1,-1)

    y_values = linear_operations.get_highest_pixel_rows(annotated_image)
    top_begin_index = top_left_point[0] + 10
    top_end_index = top_right_point[0] - 10
    num_leading_zeros = top_begin_index
    y_values = y_values[top_begin_index:top_end_index]
    y_values = np.array(y_values)
    try:
        y_values_sg = linear_operations.savitzky_golay_smooth(y_values, 40, 1)
    except Exception as e:
        print("Error with this one: ", input_image_path)
        y_values_sg = y_values

    if make_line_plots:
        operations.make_line_plot([x + top_begin_index for x in np.arange(len(y_values_sg))], y_values_sg)

    #when we have an uplift of this 15 pixels, we have a pressure ridge, should we scale with how far slip has gone?
    if top_left_point[1] - highest_point[1] >= constants.PR_THRESHOLD:
        classification = ScarpClassification.PRESSURE_RIDGE

        #find scarp point by checking where our right pixel list increases only slightly, indicating a downhill slope
        scarp_point = linear_operations.find_index_difference_less_than_amt(rightmost_pixels_list, 5)
        #find DZ end by using rightmost pixel values, checking where we reach the a large jump (meaning the right edge of footwall)
        end_dz = linear_operations.find_deviation(y_values_sg, y_values_sg[-1], 5, traverse_left=False)
        dz_end = (int(end_dz[0]) + num_leading_zeros, height - int(end_dz[1]))
        #instead, find average height difference
        #ERROR CHECK
        if end_dz[0] == -1:
            if top_right_point[1] - highest_point[1] > 15:
                #we need to default to anohter method to find dz end
                end_dz = linear_operations.find_deviation(y_values_sg, y_values_sg[-1], 3, traverse_left=False)
                dz_end = (int(end_dz[0]) + num_leading_zeros, height - int(end_dz[1]))
            else:
                end_dz = scarp_point
                end_dz = (end_dz[0], len(rightmost_pixels_list) - end_dz[1])
                dz_end = (end_dz[0], right_end_index - end_dz[1])
        #find DZ begin by using smoothed surface, where a deviation indicates deformation
        begin_dz = linear_operations.find_deviation(y_values_sg, y_values_sg[0], 4, traverse_left=True)

        #gotta take compliment of x value
        scarp_point = (scarp_point[0], len(rightmost_pixels_list) - scarp_point[1])

        #translates from 1D signal into 2D
        dz_begin = (int(begin_dz[0]) + num_leading_zeros, height - int(begin_dz[1]))
        
        scarp_point = (scarp_point[0], right_end_index - scarp_point[1])
        #ERROR CHECK
        if scarp_point[0] == -1:
            scarp_point = highest_point
        #MAKE SURE BEGIN IS BEFORE SCARP POINT
        if scarp_point[0] < dz_begin[0]:
                #resetting scarp point to dz begin
                scarp_point = dz_begin
        
        #only look at image after scarp point and check for either:
        index_for_object = scarp_point
        image_for_collapse = annotated_image[:,index_for_object[0]:]
        y_values_c = linear_operations.get_highest_pixel_rows(image_for_collapse)
        top_begin_index_c = 0
        top_end_index_c = top_right_point[0] - 10 - index_for_object[0]
        num_leading_zeros = top_begin_index_c
        y_values_c = y_values_c[top_begin_index_c:top_end_index_c]
        y_values_c = np.array(y_values_c)
        try:
            y_values_c_sg = linear_operations.savitzky_golay_smooth(y_values_c, 40, 1)
        except Exception as e:
            print("Error with this one: ", input_image_path)
            y_values_c_sg = y_values_c

        #1) an large change of pixel heights, indicating collapse
        huge_change_point = linear_operations.find_point_difference(y_values_c, 5, 60)
        #2) or a sudden increase in slope, which wouldn't occur after the scarp point unless it was a collapse
        positive_slope_point = linear_operations.find_increase(y_values_c_sg, 5, 1)

        if make_line_plots:
            operations.make_line_plot([x + index_for_object[0] for x in np.arange(len(y_values_c))], y_values_c, [(huge_change_point[0] + index_for_object[0], huge_change_point[1]), (positive_slope_point[0] + index_for_object[0], positive_slope_point[1])])
        #extract objects from image, which is indicative of a collapse
        no_blues = operations.keep_reds(annotated_image)[:,index_for_object[0]:]
        no_blues = operations.remove_sparse_pixels(no_blues)
        contour = linear_operations.find_and_draw_rectangles(no_blues, objects_to_remove = 0, min_area = 3000, max_area = 5000, max_height = top_right_point[1])
        if huge_change_point[0] >= 0:
            pr_decrease_loc = (huge_change_point[0] + index_for_object[0] + num_leading_zeros, height - huge_change_point[1])
            pr_decrease_loc = (int(pr_decrease_loc[0]), int(pr_decrease_loc[1]))
            classification = ScarpClassification.PRESSURE_RIDGE_COLLAPSE
            contour = None
        elif positive_slope_point[0] >= 0:
            pr_decrease_loc = (positive_slope_point[0] + index_for_object[0] + num_leading_zeros, height - positive_slope_point[1])
            pr_decrease_loc = (int(pr_decrease_loc[0]), int(pr_decrease_loc[1]))
            classification = ScarpClassification.PRESSURE_RIDGE_COLLAPSE
            contour = None
        
        elif contour is not None:
            contour = contour + [index_for_object[0], 0] 
            pr_decrease_locations =  linear_operations.find_decrease(y_values_c_sg, 5, 10, True)
            #we need to make sure the particle has a decrease in it (maybe increase)(do we even need this code?)
            distance_threshold = 5
            for location in pr_decrease_locations:
                temp_location = (location[0] + index_for_object[0] + num_leading_zeros, height - location[1])
                
                x, y, w, h = cv2.boundingRect(contour)

                if(x - distance_threshold) <= temp_location[0] <= (x + w + distance_threshold) and \
                    (y - distance_threshold) <= temp_location[1] <= (y + h + distance_threshold):
                    classification = ScarpClassification.PRESSURE_RIDGE_COLLAPSE
                    pr_decrease_loc = temp_location
                    break;

            if pr_decrease_loc[0] >= 0:
                pr_decrease_loc = (int(pr_decrease_loc[0]), int(pr_decrease_loc[1]))

            else:
                contour = None
                pr_decrease_loc = (-1,-1)
        try:
            r_squared_lobf_1 = linear_operations.get_squared_value(dz_begin, scarp_point, y_values, num_leading_zeros, top_begin_index, top_end_index)
            
            r_squared_lobf_2 = linear_operations.get_squared_value(scarp_point, dz_end, y_values, num_leading_zeros, top_begin_index, top_end_index)
            
            avg_r_squared = r_squared_lobf_1 * (scarp_point[0]-dz_begin[0]) + r_squared_lobf_2 * (dz_end[0]-scarp_point[0])
            avg_r_squared = avg_r_squared / (dz_end[0] - dz_begin[0])
            # print("R-squared value:", avg_r_squared)
            pixel_measurements["R^2 Value"] = avg_r_squared
        except:
            pixel_measurements["R^2 Value"] = "N/A"
        
    else:
        #find where we have a abberation from a generally flat surface, indicating deformation
        scarp_point = linear_operations.find_deviation(y_values_sg, y_values_sg[0], 5, traverse_left=True)
        if scarp_point[0] == -1:
            scarp_point = highest_point
        #find either a large decrease
        huge_change_point = linear_operations.find_point_difference(y_values, 13, 50, decrease_or_increase = False)
        #find a positive slope in the surface
        positive_slope_point = linear_operations.find_increase(y_values_sg, 5, 3)
        if huge_change_point[0] >= 0:
            huge_change_point = (huge_change_point[0] + num_leading_zeros, height - huge_change_point[1])
            huge_change_point = (int(huge_change_point[0]), int(huge_change_point[1]))

            #look at the right pixels, and determine if there is a negative slope(a feature that is unique to Simple Scarps)
            chopping_index = huge_change_point
            chopping_image = annotated_image[:,:chopping_index[0]]
            chopped_begin_index = highest_point[1]
            chopped_end_index = highest_point[1] + int((top_right_point[1] - highest_point[1]) * 0.7)
            chopped_pixels_list, chopped_begin_index, chopped_end_index = linear_operations.find_rightmost_pixel(chopping_image, chopped_begin_index, chopped_end_index)
            simple_chopped_begin = linear_operations.find_decrease(chopped_pixels_list, 10, 3, return_list = False)
            if make_line_plots:
                operations.make_line_plot(chopped_pixels_list, [chopped_begin_index - y for y in np.arange(len(chopped_pixels_list))])
            

            #these two checks are to decrease chance that the dropoff is due to a piece of collapse material, instead of the hanging wall
            #checks if the scarp point x value is near the dropoff [we index at 1 instead of 0 since rightmost pixels has swapped coords]
            ss_begin_valid = huge_change_point[0] - 20 <= simple_chopped_begin[1] <= huge_change_point[0] + 10
            #checks if scarp point is near the top of hanging wall
            ss_begin_near_top = huge_change_point[1] - top_left_point[1] <= 40
            #if we have a large dropoff and positive slope, this indicates a monoclinal collapse. simple scarps don't tend to have positive slopes at any point
            if positive_slope_point[0] >= 0:
                mc_decrease_loc = (positive_slope_point[0] + num_leading_zeros, height - positive_slope_point[1])
                mc_decrease_loc = (int(mc_decrease_loc[0]), int(mc_decrease_loc[1]))
                classification = ScarpClassification.MONOCLINAL_COLLAPSE
                contour = None
                #find scarp point and DZ end in similar fashion as before
                scarp_point = linear_operations.find_decrease(y_values_sg, 15, 2)
                scarp_point = (int(scarp_point[0]), int(scarp_point[1]))
                end_dz = linear_operations.find_deviation(y_values_sg, y_values_sg[-1], 5, traverse_left=False)
                dz_end = (int(end_dz[0]) + num_leading_zeros, height - int(end_dz[1]))
                #error checking, set the scarp point, dz begin, and dz end to center if we are early
                #if not enough deformation, set everything to the middle
                if scarp_point[0] == -1:
                    middle_x_val = (top_right_point[0] + top_left_point[0]) // 2
                    middle_y_val = (top_right_point[1] + top_left_point[1]) // 2
                    scarp_point = (middle_x_val, middle_y_val)
                    dz_begin = scarp_point
                    dz_end = dz_begin
                elif end_dz[0] == -1:
                    if top_right_point[1] - highest_point[1] > 15:
                        #we need to default to anohter method to find dz end
                        end_dz = linear_operations.find_deviation(y_values_sg, y_values_sg[-1], 3, traverse_left=False)
                        dz_end = (int(end_dz[0]) + num_leading_zeros, height - int(end_dz[1]))
                    else:
                        scarp_point = (scarp_point[0] + num_leading_zeros, height - scarp_point[1])
                        dz_begin = scarp_point
                        dz_end = dz_begin
                else:
                    scarp_point = (scarp_point[0] + num_leading_zeros, height - scarp_point[1])
                    dz_begin = scarp_point
                try:
                    r_squared_lobf = linear_operations.get_squared_value(dz_begin, dz_end, y_values, num_leading_zeros, top_begin_index, top_end_index)
                    pixel_measurements["R^2 Value"] = r_squared_lobf
                except:
                    pixel_measurements["R^2 Value"] = "N/A"

            #if we have a large dropoff in the highest pixels in each column, no positive slope, and a negative hanging wall: likely a simple scarp
            elif simple_chopped_begin[0] >= 0 and ss_begin_valid and ss_begin_near_top:
                ss_begin_for_image = (simple_chopped_begin[1], simple_chopped_begin[0] + chopped_begin_index)
                scarp_point = ss_begin_for_image
                #Kristen said to do this vs set scarp point to the location where the negative slope occurs
                scarp_point = huge_change_point
                scarp_point = operations.find_last_nonzero_pixel(annotated_image, top_left_point[1]+3)

                #USING OBJECT DETECTION FOR DZ BEGIN
                image_for_finding_dz = annotated_image[:,:scarp_point[0]]
                image_for_finding_dz = operations.keep_reds(image_for_finding_dz)
                image_for_finding_dz = operations.remove_sparse_pixels(image_for_finding_dz)
                contour = linear_operations.find_and_draw_rectangles(image_for_finding_dz)
                object_mask = np.zeros(image_for_finding_dz.shape, dtype=np.uint8)
                cv2.drawContours(object_mask, [contour], -1, 255, thickness=cv2.FILLED)
                dz_begin = operations.find_last_nonzero_pixel(object_mask, top_right_point[1] - 10)
                if dz_begin[0] == -1:
                    dz_begin = operations.find_last_nonzero_pixel(object_mask, bottom_left_point[1] - 5)
                    if dz_begin[0] == -1:
                        dz_begin = (scarp_point[0], top_right_point[1])


                no_blues = operations.keep_reds(annotated_image)
                
                contour = linear_operations.find_and_draw_rectangles(no_blues, 0, min_area = 3000, max_area = 100000, max_height = top_right_point[1] + 50)
                
                #detect objects in the image, which would indicate a collapse
                no_blues = operations.keep_reds(annotated_image)
                contour = linear_operations.find_and_draw_rectangles(no_blues, 0, min_area = 3000, max_area = 100000, max_height = top_right_point[1] + 50)
                end_dz = linear_operations.find_index_difference_more_than_amt(rightmost_pixels_list, 100)

                end_dz = linear_operations.find_deviation(y_values_sg, y_values_sg[-1], 5, traverse_left=False)
                dz_end = (int(end_dz[0]) + num_leading_zeros, height - int(end_dz[1]))
                
                if end_dz[0] == -1:
                    if top_right_point[1] - highest_point[1] > 15:
                        end_dz = linear_operations.find_deviation(y_values_sg, y_values_sg[-1], 3, traverse_left=False)
                        dz_end = (int(end_dz[0]) + num_leading_zeros, height - int(end_dz[1]))
                    else:
                        dz_end = dz_begin
                if contour is not None:
                    classification = ScarpClassification.SIMPLE_COLLAPSE
                #if the point where the Z shape begins is far below hanging wall, this mean tip was collapsed
                elif ss_begin_for_image[1] - top_left_point[1] >= 35:
                    temp = np.copy(annotated_image)
                    operations.box_and_caption_pixel(temp, *ss_begin_for_image, "right", "ss begin")
                    classification = ScarpClassification.SIMPLE_COLLAPSE
                else:
                    classification = ScarpClassification.SIMPLE

                #if the hanging wall sticks out past the end of DZ, then set the DZ x val to the hanging wall tip
                if scarp_point[0] > dz_end[0]:
                    dz_end = (scarp_point[0], dz_end[1])

            #if a big dropoff, but no increase slope or Z shape, then MC collapse
            else:
                classification = ScarpClassification.MONOCLINAL_COLLAPSE
                mc_decrease_loc = huge_change_point
                scarp_point = linear_operations.find_decrease(y_values_sg, 15, 2)
                scarp_point = (int(scarp_point[0]), int(scarp_point[1]))
                end_dz = linear_operations.find_deviation(y_values_sg, y_values_sg[-1], 5, traverse_left=False)
                dz_end = (int(end_dz[0]) + num_leading_zeros, height - int(end_dz[1]))

                #if not enough deformation, set everything to the middle
                if scarp_point[0] == -1:
                    middle_x_val = (top_right_point[0] + top_left_point[0]) // 2
                    middle_y_val = (top_right_point[1] + top_left_point[1]) // 2
                    scarp_point = (middle_x_val, middle_y_val)
                    dz_begin = scarp_point
                    dz_end = dz_begin
                elif end_dz[0] == -1:
                    if top_right_point[1] - highest_point[1] > 15:
                        end_dz = linear_operations.find_deviation(y_values_sg, y_values_sg[-1], 3, traverse_left=False)
                        dz_end = (int(end_dz[0]) + num_leading_zeros, height - int(end_dz[1]))
                    else:
                        scarp_point = (scarp_point[0] + num_leading_zeros, height - scarp_point[1])
                        dz_begin = scarp_point
                        dz_end = dz_begin
                else:
                    scarp_point = (scarp_point[0] + num_leading_zeros, height - scarp_point[1])
                    dz_begin = scarp_point
            #if no significant surface deformation, then this is just a MC
            try:
                r_squared_lobf = linear_operations.get_squared_value(dz_begin, dz_end, y_values, num_leading_zeros, top_begin_index, top_end_index)
                pixel_measurements["R^2 Value"] = r_squared_lobf
            except:
                pixel_measurements["R^2 Value"] = "N/A"
        else:
            classification = ScarpClassification.MONOCLINAL
            scarp_point = linear_operations.find_index_difference_less_than_amt(rightmost_pixels_list, 5)
            #find DZ end by using rightmost pixel values, checking where we reach the a large jump (meaning the right edge of footwall)
            end_dz = linear_operations.find_deviation(y_values_sg, y_values_sg[-1], 5, traverse_left=False)
            dz_end = (int(end_dz[0]) + num_leading_zeros, height - int(end_dz[1]))
            #instead, find average height difference
            #ERROR CHECK
            if end_dz[0] == -1:
                if top_right_point[1] - highest_point[1] > 15:
                    end_dz = linear_operations.find_deviation(y_values_sg, y_values_sg[-1], 3, traverse_left=False)
                    dz_end = (int(end_dz[0]) + num_leading_zeros, height - int(end_dz[1]))
                else:
                    end_dz = scarp_point
                    end_dz = (end_dz[0], len(rightmost_pixels_list) - end_dz[1])
                    dz_end = (end_dz[0], right_end_index - end_dz[1])
            #find DZ begin by using smoothed surface, where a deviation indicates deformation
            begin_dz = linear_operations.find_deviation(y_values_sg, y_values_sg[0], 4, traverse_left=True)

            #gotta take compliment of x value
            scarp_point = (scarp_point[0], len(rightmost_pixels_list) - scarp_point[1])
            
            #translates from 1D signal into 2D
            dz_begin = (int(begin_dz[0]) + num_leading_zeros, height - int(begin_dz[1]))
            
            scarp_point = (scarp_point[0], right_end_index - scarp_point[1])
            #ERROR CHECK
            if scarp_point[0] == -1:
                middle_x_val = (top_right_point[0] + top_left_point[0]) // 2
                middle_y_val = (top_right_point[1] + top_left_point[1]) // 2
                scarp_point = (middle_x_val, middle_y_val)
                dz_begin = scarp_point
                dz_end = dz_begin
            #MAKE SURE BEGIN IS BEFORE SCARP POINT
            if scarp_point[0] < dz_begin[0]:
                scarp_point = dz_begin

            #FINDING r-squared value
            #To do this we will:
            #1: Find the DZ begin and DZ end
            #2: Plot the DZ begin and DZ end, as well as a straight line between the two
            #3: Plot the surface of the scarp, ensuring that dz begin and end are on the surface
            #4: Find R^2 values between surface and straight line
            #RATIONALE: Don't do smoothing, use real y values since smoothing will take away from roughness measure
            #TWO Options for "true line":
            # 1) Find line of best fit of surface between dz_begin and dz_end <---- doing this one
            # 2) Find straight line between dz_begin and dz_end 

            #code to find y values sg:
            ###############
            y_values_1 = linear_operations.get_highest_pixel_rows(annotated_image)
            top_begin_index_1 = top_left_point[0] + 10
            top_end_index_1 = top_right_point[0] - 10
            num_leading_zeros_1 = top_begin_index_1
            y_values_cropped_1 = y_values_1[top_begin_index_1:top_end_index_1]
            y_values_np_1 = np.array(y_values_cropped_1)
            try:
                y_values_sg_1 = linear_operations.savitzky_golay_smooth(y_values_cropped_1, 40, 1)
            except:
                y_values_sg_1 = y_values_np_1
            begin_dz_1 = linear_operations.find_deviation(y_values_sg_1, y_values_sg_1[0], 4, traverse_left=True)
            dz_begin_1 = (int(begin_dz_1[0]) + num_leading_zeros_1, int(begin_dz_1[1]))

            end_dz_1 = linear_operations.find_deviation(y_values_sg_1, y_values_sg_1[-1], 5, traverse_left=False)
            dz_end_1 = (int(end_dz_1[0]) + num_leading_zeros_1, int(end_dz_1[1]))

            try:
                r_squared_lobf = linear_operations.get_squared_value(dz_begin_1, dz_end_1, y_values_cropped_1, num_leading_zeros_1, top_begin_index_1, top_end_index_1)
                pixel_measurements["R^2 Value"] = r_squared_lobf
            except:
                pixel_measurements["R^2 Value"] = "N/A"

            ###############
            



    pixel_measurements["Scarp_Class"] = classification.value
    #Y value is in descending order
    us_minus_ud = top_left_point[1] - scarp_point[1]
    pixel_measurements["Us - Ud"] = round(us_minus_ud * pixel_to_m, 3)

    vd_hw = bottom_right_point[1] - bottom_left_point[1]
    pixel_measurements["VD_HW"] = round(vd_hw * pixel_to_m, 3)

    hd_hw = hanging_wall_point[0] - foot_wall_point[0]
    pixel_measurements["HD_HW"] = round(hd_hw * pixel_to_m, 3)

    dz_width = dz_end[0] - dz_begin[0]
    pixel_measurements["DZW"] = round(dz_width * pixel_to_m, 3)

    scarp_height = top_right_point[1] - scarp_point[1]
    pixel_measurements["Scarp_Height"] = round(scarp_height * pixel_to_m, 3)

    if classification == ScarpClassification.SIMPLE or classification == ScarpClassification.SIMPLE_COLLAPSE:
        scarp_angle = np.degrees(np.arctan2(dz_begin[0] - scarp_point[0], dz_begin[1] - scarp_point[1]))
        scarp_angle = round(scarp_angle, 3)
        pixel_measurements["Scarp_Dip"] = scarp_angle
    else:
        scarp_angle = np.degrees(np.arctan2(scarp_height, dz_end[0] - scarp_point[0]))
        scarp_angle = round(scarp_angle, 3)
        pixel_measurements["Scarp_Dip"] = scarp_angle

    #CONVERT TO M from PX
    # to do this, we multiply X values by pixel_to_m factor
    # then we multiply height - Y values by pixel_to_m_factor
    #USX, USY, DZW XMIN, DZW XMIN_Y, DZW X_MAX, DZW XMAX_Y, DZW XMAX_Y0
    # Usx - DZWxmin
    # Usy - DZWxminy 
    pixel_measurements["Us_x"] = round(scarp_point[0] * pixel_to_m, 3)
    pixel_measurements["Us_y"] = round((height - scarp_point[1]) * pixel_to_m, 3)
    pixel_measurements["DZW xmin"] = round(dz_begin[0] * pixel_to_m, 3)
    pixel_measurements["DZW xmin_y"] = round((height - dz_begin[1]) * pixel_to_m, 3)
    pixel_measurements["DZW xmax"] = round(dz_end[0] * pixel_to_m, 3)
    pixel_measurements["DZW xmax_y"] = round((height - dz_end[1]) * pixel_to_m, 3)

    
    if visualize_points:
        temp = np.copy(input_image)
        edited_image = operations.reduce_brightness(temp)

        box_color = (255, 255, 255)
        line_color = (0, 255, 255)
        if classification == ScarpClassification.SIMPLE or classification == ScarpClassification.SIMPLE_COLLAPSE:
            operations.place_line_segment(edited_image, (scarp_point[0], dz_begin[1]), scarp_point, line_color)
            operations.place_line_segment(edited_image, dz_begin, (scarp_point[0], dz_begin[1]), line_color)
            operations.plot_dashed_line(edited_image, dz_begin, scarp_point, line_color)
            if contour is not None:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(edited_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            operations.place_line_segment(edited_image, (scarp_point[0], dz_end[1]), scarp_point, line_color)
            operations.place_line_segment(edited_image, dz_end, (scarp_point[0], dz_end[1]), line_color)
            operations.plot_dashed_line(edited_image, dz_end, scarp_point, line_color)
        
        # operations.box_and_caption_pixel(edited_image, *scarp_point, "above", "Top of Scarp", box_color)
        # operations.box_and_caption_pixel(edited_image, *dz_begin, "below", "DZ Begin", box_color)
        # operations.box_and_caption_pixel(edited_image, *dz_end, "below", "DZ End ", box_color)
        operations.box_and_caption_pixel(edited_image, *scarp_point, "above", "", box_color)
        operations.box_and_caption_pixel(edited_image, *dz_begin, "below", "", box_color)
        operations.box_and_caption_pixel(edited_image, *dz_end, "below", "", box_color)

        # operations.box_and_caption_pixel(edited_image, *top_left_point, "right", "TL", box_color)
        # operations.box_and_caption_pixel(edited_image, *top_right_point, "left", "TR", box_color)
        # operations.box_and_caption_pixel(edited_image, *bottom_left_point, "right", "BL", box_color)
        # operations.box_and_caption_pixel(edited_image, *bottom_right_point, "left", "BR", box_color)

        # operations.box_and_caption_pixel(edited_image, 100, 100, "right", "Trial: " + trial_name, None)
        operations.box_and_caption_pixel(edited_image, 100, 100, "right", classification.value, None)
        operations.box_and_caption_pixel(edited_image, 100, 150, "right", "Slip Amount: " + str(slip_amount) + "m", None)
        # operations.box_and_caption_pixel(edited_image, 100, 200, "right", "Image: " + str(photo_num) + " Slip Amount: " + str(slip_amount) + "m", [0,0,0])
 
        output_dir = os.path.join("results", "visualizations", trial_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"{photo_num}.png")
        cv2.imwrite(output_path, edited_image)
    return pixel_measurements