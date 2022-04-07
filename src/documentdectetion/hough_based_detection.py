import os
import PIL
import cv2
import time
import glob
import heapq
import itertools
import PIL.Image
import numpy as np

from skimage import draw
from IPython import display
from skimage import transform
from skimage.feature import peak_local_max


def resize_to_240(image_1):
    if image_1.shape[0] > image_1.shape[1]:
        width = 240
        k = width / image_1.shape[1]
        height = round(image_1.shape[0] * k)
    else:
        height = 240
        k = height / image_1.shape[0]
        width = round(image_1.shape[1] * k)
    
    dim = (width, height)
    image_2 = cv2.resize(image_1, dim)
    return image_2, k


# EDGE DETECTION
def morp_per_col_on_mat(image):
    image_1 = np.expand_dims(image, axis=0)
    
    kernelSize = (1, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    
    image_2 = cv2.morphologyEx(image_1, cv2.MORPH_OPEN, None)
    image_3 = cv2.morphologyEx(image_2, cv2.MORPH_CLOSE, None)
    
    image_4 = np.squeeze(image_3)
    return image_4


def morp_per_col_on_tensor(image):
    list_channel = cv2.split(image)
    
    morp_list_channel = [morp_per_col_on_mat(channel)
                         for channel in list_channel]
    
    squeezed_list_channel = [np.squeeze(channel)
                             for channel in morp_list_channel]
    
    morp_image = cv2.merge(squeezed_list_channel)
    return morp_image


def compute_edge_per_direction(image):
    #============#
    # 1: apply an opening operation followed by a closing operation
    image_1 = morp_per_col_on_tensor(image)
    
    #============#
    # 2: calculate derivative along the Y-axis
    image_2 = np.zeros_like(image_1, dtype=np.int16)
    temp = image_1.astype(np.int16)
    image_2[1:] = np.abs(temp[:-1] - temp[1:]).astype(np.uint8)
    del temp
    
    #============#
    # 3: average over each pixel of the RGB channels
    image_3 = np.mean(image_2, axis=2)
    
    #============#
    # 4: apply non-maxima suppression operation inside each column
    up_bool = (image_3[1:-1] - image_3[2:]) > 0
    down_bool = (image_3[1:-1] - image_3[:-2]) > 0
    total_bool = up_bool & down_bool

    # values in border is unchanged
    # image_4 = image_3.copy()
    # image_4[1:-1][total_bool == False] = 0

    # values in border is zeros
    image_4 = np.zeros_like(image_3)
    image_4[1:-1][total_bool == True] = image_3[1:-1][total_bool == True]

    # remove pixels with absolute derivative values which are not greater than 1
    image_4[image_4 <= 1.0] = 0.0
    
    #============#
    # 5: collect the horizontal connectivity components
    binary_map = (image_4 > 0.1).astype(np.uint8) * 255
    connectivity = 8

    # compute connected Components Algorithm
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)

    # Find the connected Components have less than 10% of the minimum
    # between the size of the maximum component and half the image width.
    largest_width = np.max(stats[1:,2])
    keep_width = round(0.1 * min(240 / 2, largest_width))

    indices = np.arange(num_labels)

    selected_components = indices[stats[:,2] > keep_width][1:]

    mask = np.zeros_like(image_4, dtype=np.uint8)

    for idx in selected_components:
        componentMask = (labels == idx).astype("uint8") * 255
        mask = cv2.bitwise_or(mask, componentMask)

    image_5 = mask
    
    #============#
    # 6: Gaussian filtering are performed for each column independently
    image_6 = cv2.GaussianBlur(image_5,(1, 3),0)
    
    return image_6


def Edges_detection(horizontal_image):
    # prepare input
    vertical_image = cv2.rotate(horizontal_image, cv2.ROTATE_90_CLOCKWISE)
    
    # calculate edges
    horizontal_edges = compute_edge_per_direction(horizontal_image)
    vertical_edges = compute_edge_per_direction(vertical_image)
    vertical_edges = cv2.rotate(vertical_edges, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return horizontal_edges, vertical_edges


# LINE DETECTION
def locate_maxima_and_inverse_FHT(edge_image, hough, angleRange, threshold_abs=None, threshold_rel=None):
    # 15 local maxima are sequentially selected in each part, provided that 
    # (i) the current maximum exceeds the 20% threshold of the global maximum and 
    # (ii) the current maximum lies more than 10 pixels by l2 norm away from the one 
    # which has been already selected.
    coordinates = peak_local_max(hough, min_distance=10, threshold_abs=threshold_abs, 
                                 threshold_rel=threshold_rel, num_peaks=15, p_norm=2)
    
    # using the inverse FHT to convert selected maxima into straight lines
    lines = np.zeros((coordinates.shape[0], 4), dtype=np.int16)
    for i, coord in enumerate(coordinates):
        lines[i] = cv2.ximgproc.HoughPoint2Line(coord[::-1], edge_image, angleRange=angleRange)
    
    return coordinates, lines
    
        
def horizontal_line_detection(edge_image, angleRange, threshold_rel=None):
    # perform the Fast Hough Transform (FHT)
    hough = cv2.ximgproc.FastHoughTransform(edge_image, cv2.CV_32S, angleRange=angleRange)
    
    # locate local maxima and perform the inverse FHT
    coordinates, lines = locate_maxima_and_inverse_FHT(edge_image, hough, angleRange, 
                                                       None, threshold_rel)
    
    return hough, coordinates, lines


def line_detection(image, horizontal_edges, vertical_edges):
    # parameter
    horizontal_angleRange = cv2.ximgproc.ARO_45_135
    vertical_angleRange = cv2.ximgproc.ARO_315_45
    horizontal_threshold = 0.2
    
    # perform Hough transform
    hor_hough, hor_coordinates, hor_lines = horizontal_line_detection(horizontal_edges, 
                                                                      horizontal_angleRange, 
                                                                      threshold_rel=horizontal_threshold)
    
    ver_hough, ver_coordinates, ver_lines = horizontal_line_detection(vertical_edges, 
                                                                      vertical_angleRange,
                                                                      threshold_rel=horizontal_threshold)
    
    return hor_lines, ver_lines


# TWO-STEP RANKING
## CONTOUR SCORE
def find_intersection_points(quad):
    line_1 = quad[:,[0, 2, 1, 3]]
    line_2 = quad[:, [2, 1, 3, 0]]

    u = ((line_2[:, :, 2]-line_2[:, :, 0])*(line_1[:, :, 1]-line_2[:, :, 1]) 
         - (line_2[:, :, 3]-line_2[:, :, 1])*(line_1[:, :, 0]-line_2[:, :, 0])) / \
        ((line_2[:, :, 3]-line_2[:, :, 1])*(line_1[:, :, 2]-line_1[:, :, 0]) 
         - (line_2[:, :, 2]-line_2[:, :, 0])*(line_1[:, :, 3]-line_1[:, :, 1]) )
    
    num_quad = quad.shape[0]
    four_points = np.zeros((num_quad, 4, 2))
    four_points[:,:,0] = line_1[:, :, 0] + u * (line_1[:, :, 2]-line_1[:, :, 0])
    four_points[:,:,1] = line_1[:, :, 1] + u * (line_1[:, :, 3]-line_1[:, :, 1])
    
    return four_points


def find_outside_points(four_points):
    # vector u of pair 3-0, 0-1, 1-2, 2-3:
    directive_u = four_points[:, [3, 0, 1, 2], :] - four_points
    
    # calculate scale factor k
    k = np.linalg.norm(directive_u, axis=2) / 10
    
    # scale vector u to 10 pixel length
    normalized_u = directive_u / k[:, :, None]
    inverse_u = -normalized_u
    
    # find 8 outside points
    eight_points = np.zeros((four_points.shape[0], 4, 2, 2))
    eight_points[:, :, 0, :] = four_points[:, [3, 0, 1, 2], :] + normalized_u
    eight_points[:, :, 1, :] = four_points + inverse_u
    
    return eight_points


def bounding_indices(shape, indices):
    indices = indices.copy()
    mask_x = (indices[0]>= shape[1]) | (indices[0]< 0)
    mask_y = (indices[1]>= shape[0]) | (indices[1]< 0)
    mask = mask_x | mask_y
    
    indices[:, mask] = 0
    
    return indices[1], indices[0]


def compute_score(gradient_map, four_points, eight_points):
    # modify input
    four_points = np.transpose(four_points, axes=(0, 2, 1)).round().astype(np.int64)
    eight_points = np.transpose(eight_points, axes=(0, 1, 3, 2)).round().astype(np.int64)
    
    shape = gradient_map.shape
    num_quad = four_points.shape[0]
    
    indices = np.arange(num_quad)
    scores = np.zeros((num_quad,))
    
    for i in indices:
        # init score variable
        wb_score = 0
        cb_score = 1
        w_b_score = 0
        
        inside_quad = four_points[i]
        outside_quad = eight_points[i]
        
        # init variable to store intensity of all pixel four side
        wb_score += gradient_map[bounding_indices(shape, inside_quad)].sum()
        
        for j in range(4):
            ## inside
            # using bresenham algorithm to draw the line between two points
            inside_edge = np.array(
                draw.line(*inside_quad[:, [j-1, j]].T.flatten())
            )
            
            # get intensity of pixel on side b
            side_value = gradient_map[bounding_indices(shape, inside_edge)]
            
            # compute cb score
            cb_score += 1 - (np.sum(side_value > 0) / side_value.shape[0])
            
            # compute wb score
            wb_score += side_value[1:-1].sum()
            
            ## outside
            # find interval
            left_interval = np.array(
                draw.line(inside_quad[0, j-1], inside_quad[1, j-1],
                                      outside_quad[j, 0, 0], outside_quad[j, 1, 0])
            )
            right_interval = np.array(
                draw.line(inside_quad[0, j], inside_quad[1, j],
                                       outside_quad[j, 0, 1], outside_quad[j, 1, 1])
            )
            total_interval = np.append(left_interval[:, 1:], right_interval[:, 1:], axis=1)
            
            # compute w_b score
            w_b_score += gradient_map[bounding_indices(shape, total_interval)].sum()
            
        # compute contour score
        scores[i] = wb_score / (1 + cb_score) - w_b_score
    
    return scores


def step_3_1(gradient_map, hor_lines, ver_lines):
    if len(ver_lines) < 2 or len(hor_lines) < 2:
        return np.zeros((1, 4, 2)), np.zeros((1,))

    # concatenate 3 part of vertical lines
#     ver_lines = np.concatenate((ver_line_list[0], 
#                                 ver_line_list[1], 
#                                 ver_line_list[2]), axis=0)

    # combination of 2 line into a pair
    hor_idx = np.transpose(np.triu_indices(len(hor_lines), 1))
    ver_idx = np.transpose(np.triu_indices(len(ver_lines), 1))

    hor_pair = hor_lines[hor_idx]
    ver_pair = ver_lines[ver_idx]

    # combination of 2 pair into a quadrilateral
    pair_idx = np.array(np.meshgrid(np.arange(len(hor_pair)), 
                                    np.arange(len(ver_pair)))).T.reshape(-1, 2)
    
    quad = np.concatenate((hor_pair[pair_idx[:,0]], ver_pair[pair_idx[:,1]]), axis=1).astype(np.int64)
    
    # find intersection points
    four_points = find_intersection_points(quad)
    
    # find outside points
    eight_points = find_outside_points(four_points)
    
    # compute score
    score = compute_score(gradient_map, four_points, eight_points)
    
    # select top 4 score
    idx = heapq.nlargest(4, np.arange(score.shape[0]), key=lambda i:score[i])
    
    return four_points[idx], score[idx]


## CONTRAST SCORE
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    top_left_idx = np.argmin(s)
    bottom_right_idx = np.argmax(s)
    rect[0] = pts[top_left_idx]
    rect[2] = pts[bottom_right_idx]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    bool_idx = [i not in [top_left_idx, bottom_right_idx] for i in range(4)]
    left_pts = pts[bool_idx]
    diff = np.diff(left_pts, axis = 1)
    rect[1] = left_pts[np.argmin(diff)]
    rect[3] = left_pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def merge_string(value):
    return int(''.join(value), 8)


def histogram(region):
    value, count = np.unique(region, axis=0, return_counts=True)
    idx = np.apply_along_axis(merge_string, 1, value.astype(np.str_))
    
    hist = np.zeros((512, ), dtype=np.int64)
    hist[idx] = count
    return hist


def step_3_2(resized_image, quads, contour_score):
    if len(contour_score) == 1:
        return order_points(quads[0])
    
    on_side = np.array([[77, 186],
                        [163, 186],
                        [163, 240],
                        [77, 240]])
    
    outer_mask = np.zeros(resized_image.shape[:2], bool)
    outer_mask[177:249, 68:172] = True
    outer_mask[186:240, 77:163] = False

    inner_mask = np.zeros(resized_image.shape[:2], bool)
    inner_mask[186:240, 77:163] = True
    inner_mask[195:231, 86:154] = False
    
    contrast_score = np.zeros(contour_score.shape)
    
    for i in range(quads.shape[0]):
        quad = order_points(quads[i]).astype(np.int32)
        
        # compute homography and quantize image
        tform = transform.estimate_transform('projective', 
                                             quad, 
                                             np.array(on_side))
        transformed = transform.warp(resized_image[:,:,::-1], tform.inverse) * 7
        transformed = transformed.round().astype(np.uint8)
        
        # compute histogram
        inner_region = transformed[inner_mask]
        outer_region = transformed[outer_mask]
        inner_hist = histogram(inner_region)
        outer_hist = histogram(outer_region)
        
        # compute chi-quare distance
        contrast_score[i] = np.sum((inner_hist - outer_hist)**2 / (inner_hist + outer_hist + 1e-5))
    
    # merge score
    select_idx = np.argmax(0.1836 * contour_score + contrast_score)
    
    return order_points(quads[select_idx])


## MAIN FUNCTION
def detector(input_dir):
    start = time.time()
    
    image_list = None
    if os.path.isdir(input_dir) is True:
        image_list = glob.glob(os.path.join(input_dir, '*.jpg'))
    elif os.path.isfile(input_dir) is True:
        image_list = [input_dir]
    else:
        print('Path is not exist')
    
    result_bbox = np.zeros((len(image_list), 8))

    print('Document detection stage:')

    for iter_, image_dir in enumerate(image_list):
        print("Image {} is being processed".format(iter_))
        image = cv2.imread(image_dir)
        resized_image, k = resize_to_240(image) # hight > width
        height, width, _ = resized_image.shape

        # 1. EDGE DETECTION
        horizontal_edges, vertical_edges = Edges_detection(resized_image)
        gradient_map = (horizontal_edges + vertical_edges) / 2

        # 2. LINE DETECTION
        # width, height order
        hor_lines, ver_lines = line_detection(resized_image, horizontal_edges, vertical_edges)
        
        # 3.1. Contour score
        quads, contour_score = step_3_1(gradient_map, hor_lines, ver_lines)

        # 3.2. Contrast score
        quad_result = step_3_2(resized_image, quads, contour_score) / k
        
            
        #     f.write('{},{},{},{},{},{},{},{}'.format(
        #         quad_result[0, 1], quad_result[0, 0],
        #         quad_result[1, 1], quad_result[1, 0],
        #         quad_result[2, 1], quad_result[2, 0],
        #         quad_result[3, 1], quad_result[3, 0],))

        result_bbox[iter_] = quad_result[:, ::].flatten()
        
    total_time = time.time() - start
    print("FPS = {}".format(len(image_list) / total_time))
    print("average time = {}".format(total_time / len(image_list)))

    return result_bbox

def single_detector(image):
    start = time.time()
    print('Document detection stage: Image is being processed')
    
    resized_image, k = resize_to_240(image) # hight > width

    # 1. EDGE DETECTION
    horizontal_edges, vertical_edges = Edges_detection(resized_image)
    gradient_map = (horizontal_edges + vertical_edges) / 2

    # 2. LINE DETECTION
    # width, height order
    hor_lines, ver_lines = line_detection(resized_image, horizontal_edges, vertical_edges)
    
    # 3.1. Contour score
    quads, contour_score = step_3_1(gradient_map, hor_lines, ver_lines)

    # 3.2. Contrast score
    quad_result = step_3_2(resized_image, quads, contour_score) / k

    result_bbox = quad_result[:, ::].flatten()
        
    total_time = time.time() - start
    print("FPS = {}".format(1 / total_time))
    print("average time = {}".format(total_time / 1))

    return result_bbox