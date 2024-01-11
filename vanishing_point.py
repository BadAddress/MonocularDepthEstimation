
from lu_vp_detect import VPDetection
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import math
import os
from itertools import combinations
import numpy as np


def get_best_one(point_list, cnt=1):
    for _ in range(cnt):
        if len(point_list)<2:
            break
        xs = [p[0] for p in point_list]
        ys = [p[1] for p in point_list]
        x_mean = np.mean(xs)
        y_mean = np.mean(ys)
        max_dist = 0
        max_idx = 0
        for i, p in enumerate(point_list):
            dx = p[0] - x_mean
            dy = p[1] - y_mean
            dist = dx**2 + dy**2
            # 如果距离更远,更新最远距离和索引
            if dist > max_dist: 
                max_dist = dist
                max_idx = i
        point_list.pop(max_idx)
    xs = [p[0] for p in point_list]
    ys = [p[1] for p in point_list]
    # 计算均值点,作为参照点
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    return (x_mean, y_mean)


def detect_and_mark_vanishing_point(image_path, focal_length, save_path, cnt,need_to_save):
    # Parameters
    length_thresh = 12
    principal_point = None
    img = Image.open(image_path)
    w, h = img.size
    #focal_length = ((w+h)/2) *  (  (float(focal_length)-26)/200 +26  )  / 28.48
    focal_length = 518.85789
    # 存储每次检测的消失点和对应距离
    vps_list = []
    for _ in range(cnt):
        # Create VPDetection object without seed
        vpd = VPDetection(length_thresh, principal_point, focal_length, None)
        # Run detection
        vps = vpd.find_vps(image_path)
        # Find the closest vanishing point to the center for this iteration
        D = float('inf')
        resP = []
        for p in vpd.vps_2D:
            cur_d = (p[0] - w / 2) ** 2 + (p[1] - h / 2) ** 2
            if cur_d < D:
                D = cur_d
                resP = p
        # Add the closest vanishing point and its distance
        vps_list.append(resP)
    # 选择离群值最小的消失点
    final_vp = get_best_one(vps_list)
    if need_to_save==True:
        # Plot and mark the vanishing point
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.plot(final_vp[0], final_vp[1], 'ro', markersize=18, fillstyle='none', markeredgewidth=7)
        ax.axis('off')
        # Save the image
        filename = os.path.basename(image_path)
        dirname = os.path.basename(os.path.split(image_path)[0])
        save_path = os.path.join(save_path,dirname+"_"+filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    return final_vp


