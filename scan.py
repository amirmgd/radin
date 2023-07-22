import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------------


m = 6000
n = 7000
sheet = np.zeros((m, n), dtype=int)


def import_images_list(folder_path):
    images = []
    threshold = 128
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = np.flip(
                np.array(Image.open(image_path).convert("L")), axis=0)
            images.append(np.where(image >= threshold, 0, 1))
    return images


def images_area(arr):
    return np.sum(arr)


address = "./test"


sorted_shape_list = sorted(import_images_list(
    address), key=len, reverse=True)


def placing_items(sheet, col_index, row_index, item):
    small_rows, small_cols = item.shape
    # Clear the area in the sheet before adding the new shape
    sheet[row_index:row_index + small_rows, col_index:col_index + small_cols] = 0
    sheet[row_index:row_index + small_rows, col_index:col_index + small_cols] += item
    return sheet

def can_push_left(sheet, row_index, col_index, item_rows, item_cols):
    return not np.any(sheet[row_index:row_index + item_rows, col_index - 1:col_index] == 1)

def can_push_down(sheet, row_index, col_index, item_cols):
    return not np.any(sheet[row_index - 1:row_index, col_index:col_index + item_cols] == 1)

row_index = 0
col_index = 0
max_row = 0
sheet_col = sheet.shape[1]
shapeSP = np.zeros((len(sorted_shape_list), 3), dtype=int)
box_num = 0

for i in range(len(sorted_shape_list)):
    small_rows, small_cols = sorted_shape_list[i].shape
    if col_index + small_cols > sheet_col:
        col_index = 0
        row_index += max_row
        box_num += 1

    # Check if we can push the shape further left
    while col_index > 0 and can_push_left(sheet, row_index, col_index, small_rows, small_cols):
        col_index -= 1

    # Check if we can push the shape further down
    while row_index > 0 and can_push_down(sheet, row_index, col_index, small_rows, small_cols):
        row_index -= 1

    sheet = placing_items(sheet, col_index, row_index, sorted_shape_list[i])
    shapeSP[i] = np.array([row_index, col_index, box_num])
    col_index += small_cols
    max_row = max(max_row, small_rows)


plt.contourf(sheet, cmap='gray')
plt.colorbar()
plt.show()

