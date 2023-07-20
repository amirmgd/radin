import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------------


m = 7000
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
    address), key=images_area, reverse=True)


row_index = 0
col_index = 0
max_row = 0
sheet_col = sheet.shape[1]
shapeSP = []
box_num=0


def big_row(row):
    global max_row
    if (row > max_row):
        max_row = row
    return max_row

def placing_items(sheet,col_index,row_index,item):
    small_rows, small_cols = item.shape
    sheet[row_index:row_index+small_rows,
          col_index:col_index+small_cols] += item
    return sheet





for i in range (len(sorted_shape_list)):
    small_rows, small_cols = sorted_shape_list[i].shape
    if (col_index + small_cols > sheet_col):
        col_index = 0
        row_index += max_row
        box_num +=1

    placing_items (sheet,col_index,row_index,sorted_shape_list[i])
    shapeSP.append((row_index, col_index,box_num))        
    col_index += small_cols
    max_row = big_row(small_rows)


def scanner(page):
    filled_rows = 0
    for i in range(page.shape[0]):
        for j in range(page.shape[1]):
            if (sheet[i][j] == 1):
                filled_rows += 1
                break

    return ((page.shape[0]-filled_rows))/page.shape[0]


def push_column(shapePL,sorted_shape_list,sheet):   
    for i in range(len(shapePL)):
        if(shapeSP[i][2] >0):
            item_rows, item_cols = sorted_shape_list[i].shape
            sheet[shapePL[i][0]:shapePL[i][0]+item_rows +1,
            shapePL[i][1]:shapePL[i][1]+item_cols +1] *= 0

            j = shapePL[i][0] - 1  # Start searching from the row above the current position
            while j >= 0:
                if np.any(sheet[j, shapePL[i][1]:shapePL[i][1] + item_cols] == 1):
                    break
                j -= 1

            # Push the shape down in the sheet matrix
            if j >= 0:  # If a valid row is found, push the shape
                sheet[j + 1:j + 1 + item_rows, shapePL[i][1]:shapePL[i][1] + item_cols] += sorted_shape_list[i]
    return sheet


push_column (shapeSP,sorted_shape_list,sheet)
plt.contourf(sheet, cmap='gray')
plt.colorbar()
plt.show()





#def extract_zero_areas_positions(sheet):
#     zero_areas = []
#     m, n = sheet.shape
#     visited = np.zeros((m, n), dtype=bool)

#     def dfs(i, j):
#         if i < 0 or i >= m or j < 0 or j >= n or visited[i, j] or sheet[i, j] == 1:
#             return 0, 0, 0, 0
#         visited[i, j] = True
#         left, right, up, down = dfs(i, j - 1)
#         new_left, new_right, new_up, new_down = dfs(i - 1, j)
#         return min(left, j - 1), max(right, j + 1), min(up, i - 1), max(down, i + 1)

#     for i in range(m):
#         for j in range(n):
#             if not visited[i, j] and sheet[i, j] == 0:
#                 left, right, up, down = dfs(i, j)
#                 zero_area_matrix = sheet[i + up:i + down + 1, j + left:j + right + 1]
#                 zero_area = {
#                     'matrix': zero_area_matrix,
#                     'start_row': i + up,
#                     'start_col': j + left,
#                     'end_row': i + down,
#                     'end_col': j + right
#                 }
#                 zero_areas.append(zero_area)

#     return zero_areas

# zero_areas = extract_zero_areas_positions(sheet)
# print("Zero-areas Positions and Matrices in the Sheet:")
# for i, zero_area in enumerate(zero_areas, 1):
#     print(f"Zero-area {i}:")
#     print("Start Position: ({}, {})".format(zero_area['start_row'], zero_area['start_col']))
#     print("End Position: ({}, {})".format(zero_area['end_row'], zero_area['end_col']))
#     print("Matrix:")
#     print(zero_area['matrix'])
#     print()
