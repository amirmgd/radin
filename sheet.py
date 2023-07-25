import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------------

m = 1500
n = 1000
sheet = np.zeros((m, n), dtype=int)


def rotate_matrix(matrix):
    return np.rot90(matrix, k=1)


def import_images_list(folder_path):
    images = []
    threshold = 128
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = np.flip(np.array(Image.open(image_path).convert("L")), axis=0)

            # Check the height and width of the image
            height, width = image.shape

            # If the height-to-width ratio is greater than 1, rotate the image
            if (height / width > 1.5):
                image = rotate_matrix(image)

            images.append(np.where(image >= threshold, 0, 1))
    return images


def images_area(arr):
    return np.sum(arr)


address = "./test"

sorted_shape_list = sorted(import_images_list(
    address), key=images_area, reverse=True)


def placing_items(sheet, col_index, row_index, item):
    small_rows, small_cols = item.shape
    # Check if we can push the shape further down
    while row_index > 0 and can_push_down(sheet, row_index, col_index, small_cols):
        row_index -= 1

    # Check if we can push the shape further left
    while col_index > 0 and can_push_left(sheet, row_index, col_index, small_rows):
        col_index -= 1

    # Place the shape in the calculated position
    sheet[row_index:row_index + small_rows, col_index:col_index + small_cols] = 0
    sheet[row_index:row_index + small_rows, col_index:col_index + small_cols] += item
    shapeSP.append((row_index, col_index,box_num))
    return sheet


def can_push_left(sheet, row_index, col_index, item_rows):
    return not np.any(sheet[row_index:row_index + item_rows, col_index - 1:col_index] == 1)


def can_push_down(sheet, row_index, col_index, item_cols):
    return not np.any(sheet[row_index-1:row_index, col_index:col_index + item_cols] == 1)


row_index = 0
col_index = 0
max_row = 0
sheet_col = sheet.shape[1]
shapeSP = []
box_num = 0

for i in range(len(sorted_shape_list)):
    small_rows, small_cols = sorted_shape_list[i].shape
    if col_index + small_cols > sheet_col:
        col_index = 0
        row_index += max_row
        box_num += 1

    sheet = placing_items(sheet, col_index, row_index, sorted_shape_list[i])
    col_index += small_cols
    max_row = max(max_row, small_rows)


def filledSpaceScanner(page):
    filled_rows = 0
    for i in range(page.shape[0]):
        for j in range(page.shape[1]):
            if (sheet[i][j] == 1):
                filled_rows += 1
                break

    return (filled_rows,((page.shape[0]-filled_rows))/page.shape[0])

maximumFilledRows = filledSpaceScanner(sheet)[0]

def move_smaller_items_to_right(sheet, item_list, maxFilledRows, shapePosition):
    sheet_rows, sheet_cols = sheet.shape
    
    for m in range(len(item_list)-1,0,-1):
        item_rows, item_cols = item_list[m].shape
        active_line = 0
        while active_line < maxFilledRows:
            found_1 = False
            insert_col = sheet_cols - item_cols
            # Search the right side of the sheet for the size of the item
            for j in range(sheet_cols - 1, sheet_cols - 1 - item_cols, -1):
                if 1 in sheet[active_line:active_line + item_rows, j]:
                    found_1 = True
                    insert_col = j + 1
                    break
            
            if found_1:
                # If 1 is found, add the line value to the active line and resume the search
                active_line += 1
            else:
                # If 1 is not found, place the item at that point and continue the search
                sheet[shapePosition[m][0]:shapePosition[m][0]+item_rows, shapePosition[m][1]:shapePosition[m][1]+ item_cols] = 0
                sheet[active_line:active_line + item_rows, insert_col:insert_col + item_cols] = item_list[m]
                active_line += item_rows
                break

    return sheet

move_smaller_items_to_right = move_smaller_items_to_right(sheet,sorted_shape_list,filledSpaceScanner(sheet)[0],shapeSP)

plt.contourf(sheet, cmap='gray')
plt.colorbar()
plt.show()