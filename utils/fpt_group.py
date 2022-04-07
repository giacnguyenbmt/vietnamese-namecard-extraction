import cv2
import numpy as np
import Polygon as plg

def draw_quads(image, pts, color =(255, 0, 0), thickness=1, isClosed=True, idx_puttext=-1):
    raw_image = image.copy()
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)

    draw_image = cv2.polylines(raw_image, [pts], 
                          isClosed, color, thickness)
    if idx_puttext != -1:
        x, y = pts[0, 0, :]
        draw_image = cv2.putText(draw_image, '{}'.format(idx_puttext), (x, y-10), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), thickness)
    return draw_image

def show_in_order(img, group_bb, m_pD, group_label, output='output/group_block.jpg'):
    drawed_img = img.copy()
    colors = np.random.random(size=(group_bb.shape[0] + 1, 3)) * 256

    for i in range(group_bb.shape[0]):
        pG = group_bb[i]
        pG = pG.reshape((-1, 1, 2))
        drawed_img = draw_quads(drawed_img, pG, colors[i+1], 3)

    for i in range(m_pD.shape[0]):
        pD = m_pD[i]
        pD = pD.reshape((-1, 1, 2))
        color = colors[group_label[i] + 1]
        drawed_img = draw_quads(drawed_img, pD, color, 3, idx_puttext=i)
        
    cv2.imwrite(output, drawed_img)
    # return(drawed_img)

def resize_base_width(image, width=240):
    if image.shape[0] > image.shape[1]:
        k = width / image.shape[1]
        height = round(image.shape[0] * k)
    else:
        height = width
        k = height / image.shape[0]
        width = round(image.shape[1] * k)
    
    dim = (width, height)
    image_rs = cv2.resize(image, dim)
    return image_rs, k

def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """        
    resBoxes=np.empty([1,8],dtype='int32')
    resBoxes[0,0]=int(points[0])
    resBoxes[0,4]=int(points[1])
    resBoxes[0,1]=int(points[2])
    resBoxes[0,5]=int(points[3])
    resBoxes[0,2]=int(points[4])
    resBoxes[0,6]=int(points[5])
    resBoxes[0,3]=int(points[6])
    resBoxes[0,7]=int(points[7])
    pointMat = resBoxes[0].reshape([2,4]).T
    return plg.Polygon(pointMat)    

def get_intersection(pD,pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

def get_union(pD,pG):
    areaA = pD.area();
    areaB = pG.area();
    return areaA + areaB - get_intersection(pD, pG);

def get_detect_intersection_over_union(pD,pG):
    try:
        return get_intersection(pD, pG) / pD.area();
    except:
        return 0
    
def swap_idx(arr, i, j):
    store = arr[j]
    for idx in range(j, i + 1, -1):
        arr[idx] = arr[idx - 1]
    arr[i + 1] = store

def selective_search(arr):
    indices = np.arange(len(arr))
    
    for i in range(len(arr) - 1):
        if arr[indices[i]] == -1:
            continue
        if arr[indices[i]] == arr[indices[i + 1]]:
            continue
        else:
            for j in range(i + 1, len(arr), 1):
                if arr[indices[j]] == arr[indices[i]]:
                    swap_idx(indices, i, j)
                    break
    return indices

def find_block(img):
    """
    Returns a list of text block, each bock is a rectangle in format: x1,y1,x2,y2,x3,y3,x4,y4
    """
    # declare
    blur_ksize = 5
    kernel = np.ones((30,30),np.uint8)
    MIN_CONTOUR_AREA = 100
    
    # =====Step 1=====
    resized_img, k = resize_base_width(img, 640)

    step_1 = resized_img
    
    # =====Step 2=====
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray,(blur_ksize,blur_ksize),0)
    img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=3)
    (thresh, im_bw) = cv2.threshold(img_sobelx, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    step_2 = im_bw
    
    # =====Step 3=====
    closing = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)

    step_3 = closing
    
    # =====Step 4=====
    outputImage = step_1.copy()
    npaContours, npaHierarchy = cv2.findContours(step_3,        
                                                 cv2.RETR_EXTERNAL,                 
                                                 cv2.CHAIN_APPROX_SIMPLE) 
    group_bb = []
    for npaContour in npaContours:                         
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)
            group_bb.append([intX, intY, 
                             intX + intW, intY,
                             intX + intW, intY + intH,
                             intX, intY + intH])
            cv2.rectangle(outputImage,           
                  (intX, intY),                 # upper left corner
                  (intX + intW, intY + intH),   # lower right corner
                  (0, 0, 255),                  # red
                  2)
    step_4 = (np.array(group_bb) / k).astype(int)
    
    return step_4

def group_text_into_block(m_pD, group_bb):
    """
    Returns a tuple (a, b), a is a list of group label, b is a list of sorted-selectively index
    """
    group_label = np.full((m_pD.shape[0], ), -1)
    pG_list = [polygon_from_points(i) for i in group_bb]

    for i in range(m_pD.shape[0]):
        pD = polygon_from_points(m_pD[i])

        label_list = [get_detect_intersection_over_union(pD,pG) for pG in pG_list]
        idx_with_max_value = np.argmax(label_list)
        if label_list[idx_with_max_value] > 0.20:
            group_label[i] = idx_with_max_value

    order = selective_search(group_label)
    
    return group_label, order

def group_by_block(image, m_pD):
    group_bb = find_block(image)
    group_label, order = group_text_into_block(m_pD, group_bb)

    show_in_order(image, group_bb, m_pD, group_label)
    return group_label, order
