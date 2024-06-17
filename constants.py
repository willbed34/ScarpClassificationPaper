import numpy as np
from enum import Enum
import cv2

class ScarpClassification(Enum):
    SIMPLE = "Simple"
    SIMPLE_COLLAPSE = "Simple Collapse"
    MONOCLINAL = "Monoclinal"
    MONOCLINAL_COLLAPSE = "Monoclinal Collapse"
    PRESSURE_RIDGE = "Pressure Ridge"
    PRESSURE_RIDGE_COLLAPSE = "Pressure Ridge Collapse"

HEADER = [
"Trial",
"Depth",
"Density",
"Cohesion",
"Sediment_Strength",
"Fault_Dip",
"Fault_Seed",
"Conversion_Factor",
"Slip",
"VD_HW",
"HD_HW",
"Scarp_Height",
"Us - Ud",
"Us_x",
"Us_y",
"DZW",
"DZW xmin",
"DZW xmin_y",
"DZW xmax",
"DZW xmax_y",
"DZW xmax_y0",
"Scarp_Dip",
"Fit_Scarp_Dip",
"Usx - DZWxmin",
"Usy - DZWxminy",
"Scarp_Class",
"R^2 Value"
]

LOWER_Y_CROP = 0
UPPER_Y_CROP = 1800
LOWER_X_CROP = 1000
UPPER_X_CROP = 3800
MAX_CUSHION = 40

PR_THRESHOLD = 12


TRIAL_PATTERN = "D5_coh1.0e+05_ten1.0e+05_dip40_FS0.25"
BOX_WIDTH = 20
FONT  = cv2.FONT_HERSHEY_TRIPLEX
FONT_SCALE = 1.2
TEXT_COLOR = (0, 0, 0)



FAULT_START_FROM_RIGHT_M = 30.0

# colors of pixels in BGR
LIGHT_BLUE = np.array([255, 255, 6], dtype=np.uint8)
LOWER_LIGHT_BLUE = np.array([240, 240, 0], dtype=np.uint8)
UPPER_LIGHT_BLUE = np.array([255, 255, 12], dtype=np.uint8)

GREEN = np.array([4, 254, 64], dtype=np.uint8)
LOWER_GREEN = np.array([0, 244, 54], dtype=np.uint8)
UPPER_GREEN = np.array([14, 255, 74], dtype=np.uint8)

DARK_BLUE = np.array([255, 1, 3], dtype=np.uint8)
DARK_BLUE_LOWER = np.array([245, 0, 0], dtype=np.uint8)
DARK_BLUE_UPPER = np.array([255, 15, 15], dtype=np.uint8)

PURPLE = np.array([255, 101, 154], dtype=np.uint8)
PURPLE_LOWER = np.array([245, 91, 144], dtype=np.uint8)
PURPLE_UPPER = np.array([255, 111, 164], dtype=np.uint8)

RED = np.array([0, 2, 238], dtype=np.uint8)
LOWER_RED = np.array([0, 0, 228], dtype=np.uint8)
UPPER_RED = np.array([10, 10, 248], dtype=np.uint8)

DARK_RED = np.array([30, 1, 183], dtype=np.uint8)
LOWER_DARK_RED = np.array([20, 0, 173], dtype=np.uint8)
UPPER_DARK_RED = np.array([40, 10, 193], dtype=np.uint8)

