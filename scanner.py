import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import random
image1 = np.flip(np.array(Image.open("../Radin-Co/18.jpg").convert("L")),axis=0)
threshold = 128

binary_array1 = np.where(image1 >= threshold, 0, 1)

image2 = np.flip(np.array(Image.open("../Radin-Co/22(3).jpg").convert("L")),axis=0)
threshold = 128

binary_array2 = np.where(image2 >= threshold, 0, 1)

def import_images_list(folder_path):
    images = []
    threshold = 128
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = np.flip(np.array(Image.open(image_path).convert("L")), axis=0)

            # Check the height and width of the image
            # height, width = image.shape

            # If the height-to-width ratio is greater than 1, rotate the image
            # if (height / width > 1.5):
            #     image = rotate_matrix(image)

            images.append(np.where(image >= threshold, 0, 1))
    return images

def aspectRatio(shape):
    return (shape.shape[0]/shape.shape[1])
def perimeter(shape):
    return (shape.shape[0]*shape.shape[1])
def area_v_perimeter(shape):
    return area(shape)/perimeter(shape)
def area(arr):
    return np.sum(arr)
address = "./test"

shapeList = sorted(import_images_list(
    address), key=area,reverse=True)

def filledSpaceScanner(page):
    filled_rows = 0
    for i in range(page.shape[0]):
        for j in range(page.shape[1]):
            if (page[i][j] == 1):
                filled_rows += 1
                break

    return (((page.shape[0]-filled_rows))/page.shape[0]) *100

def allSame(myList):
    return all(x == myList[0] for x in myList)

def divide_to_four(matrix,origin):
    rows,cols = matrix.shape
    if rows == 1 and cols == 1:
        return matrix
    center_row = rows // 2
    center_col = cols // 2
    # Divided matrices
    topLeft = matrix[center_row:, :center_col]
    topRight = matrix[center_row:, center_col:]
    bottomLeft= matrix[:center_row, :center_col]
    bottomRight = matrix[:center_row, center_col:]
    # Matrices start
    topLeft_start = (origin[0] + center_row, origin[1])
    topRight_start = (origin[0] + center_row, center_col + origin[1])
    bottomLeft_start = (origin[0], origin[1])
    bottomRight_start = (origin[0], center_col+ origin[1])

    # Return the values

    return (topLeft,topLeft_start,topRight,topRight_start,bottomLeft,bottomLeft_start,bottomRight,bottomRight_start)


def zero_finder(matrix, origin):
    # Define a nested function to handle recursion and results
    rows, cols = matrix.shape
    sub_areas = []
    # If the matrix is 1x1, return the matrix
    if rows == 1 and cols == 1:
        if matrix[0, 0] == 0:
            return [origin]
        else:
            return []
    divided_matrix = divide_to_four(matrix, origin)
    for i in range(0, 8, 2):
        sub_areas.append(area(divided_matrix[i]))
    # print(sub_areas)
    if sub_areas[sub_areas.index(min(sub_areas))] == 0:
        return  [divided_matrix[sub_areas.index(min(sub_areas)) * 2],
                            divided_matrix[sub_areas.index(min(sub_areas)) * 2 + 1]]
    elif(allSame(sub_areas)==True):
        index = random.randint(0,3)
        return zero_finder(divided_matrix[index* 2],
                            divided_matrix[index * 2 + 1])
    else:
        return zero_finder(divided_matrix[sub_areas.index(min(sub_areas)) * 2],
                            divided_matrix[sub_areas.index(min(sub_areas)) * 2 + 1])


def zero_exapnder(matrix,origin,zero_block):
    counterList=[]
    #scan_left
    col_indexL = origin[1]
    counterL = 0
    while col_indexL > 0 and not (np.any(matrix[origin[0]:origin[0] + zero_block.shape[0],col_indexL -1: col_indexL]) == 1):
        counterL +=1
        col_indexL -=1
    counterList.append(counterL)
    #scan_bottom
    row_indexB = origin[0]
    counterB = 0
    while row_indexB-1 > 0 and not(np.any(matrix[row_indexB - 1 : row_indexB,origin[1]: origin[1] + zero_block.shape[1]] == 1)):
        counterB += 1
        row_indexB -= 1
    counterList.append(counterB)
    maxIndex = counterList.index(max(counterList))
    if(max(counterList) ==0):
        return (origin)
    elif(maxIndex == 0):
        additional_cols = np.zeros((zero_block.shape[0], counterList[0]), dtype=int)
        new_zeroBlock = np.concatenate((zero_block, additional_cols),axis=1)
        return zero_exapnder(matrix,(origin[0],origin[1]-counterList[maxIndex]),new_zeroBlock)
    elif(maxIndex == 1):
        additional_rows = np.zeros((counterList[1], zero_block.shape[1]),dtype=int)
        new_zeroBlock = np.vstack((zero_block, additional_rows))
        return zero_exapnder(matrix,(origin[0]-counterList[maxIndex],origin[1]),new_zeroBlock)


# print(test)
for index,shape in enumerate(shapeList):
    zero = zero_finder(binary_array1, (0,0))
    new_origin=zero_exapnder(binary_array1, zero[1],zero[0])
    if(new_origin[0] + shape.shape[0] < binary_array1.shape[0] and new_origin[1] + shape.shape[1] < binary_array1.shape[1]
       and not(np.any(binary_array1[new_origin[0]: new_origin[0] + shape.shape[0], new_origin[1]: new_origin[1] + shape.shape[1]] + shape >1))):
            binary_array1[new_origin[0]: new_origin[0] + shape.shape[0], new_origin[1]: new_origin[1] + shape.shape[1]] += shape

print(filledSpaceScanner(binary_array1))
plt.contourf(binary_array1,cmap="gray")
plt.show()