import math
import os
import shutil
from typing import Tuple

import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from scipy import interpolate

from typing import Tuple


def sitk_format(base_itk: sitk.Image, dst_itk: sitk.Image):
    """
    :param base_itk:
    :param dst_itk:
    :return:
    """
    dst_itk.SetOrigin(base_itk.GetOrigin())
    dst_itk.SetSpacing(base_itk.GetSpacing())
    dst_itk.SetDirection(base_itk.GetDirection())


def max_connected_domain(itk_mask):
    """
    :param itk_mask: SimpleITK.Image
    :return:
    """

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_mask = cc_filter.Execute(itk_mask)

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)

    num_connected_label = cc_filter.GetObjectCount()

    area_max_label = 0
    area_max = 0


    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)
        if area > area_max:
            area_max_label = i
            area_max = area

    np_output_mask = sitk.GetArrayFromImage(output_mask)

    res_mask = np.zeros_like(np_output_mask)
    res_mask[np_output_mask == area_max_label] = 1

    res_itk = sitk.GetImageFromArray(res_mask)
    res_itk.SetOrigin(itk_mask.GetOrigin())
    res_itk.SetSpacing(itk_mask.GetSpacing())
    res_itk.SetDirection(itk_mask.GetDirection())

    return res_itk



class TreeNode:
    def __init__(self, idx, box):
        """
        :param box:top-left-bottom-right
        :param idx:idx of slice
        """
        self.idx = idx
        self.box = box
        self.mid_point = [(box[1] + box[3]) / 2, (box[0] + box[2]) / 2]
        self.father = set()
        self.children = set()

    def __str__(self):
        # print('idx', self.idx, 'box', self.box)
        return 'idx' + str(self.idx) + 'box' + str(self.box)

    def is_connect(self, other, threshold=0.1):
        """
        analyse connectivity from iou & distance
        :param threshold:
        :param other:
        :return:
        """

        return self.iou(other) > threshold

    def get_mid_point(self):
        return self.mid_point

    def is_left(self):
        """
        analyse node position of left head or right head
        :return:
        """
        mid = []
        if len(self.father) > 0:
            root = self
            while True:
                root = list(root.father)[0]
                mid.append((root.box[1] + root.box[3]) / 2)
                if len(root.father) == 0:
                    break

        if len(self.children) > 0:
            bottom = self
            while True:
                bottom = list(bottom.children)[0]
                mid.append((bottom.box[1] + bottom.box[3]) / 2)
                if len(bottom.children) == 0:
                    break
        return sum(mid) / len(mid) < 256

    def get_box_size(self):
        return self.box[3] - self.box[1], self.box[2] - self.box[0]  # (width,height)

    def iou(self, other):
        w1, h1 = self.get_box_size()
        w2, h2 = other.get_box_size()
        union = w1 * h1 + w2 * h2
        h, w = 0, 0
        if self.box[0] < other.box[0]:
            if other.box[0] < self.box[2] < other.box[2]:
                h = self.box[2] - other.box[0]
            elif self.box[2] >= other.box[2]:
                h = h2
        elif self.box[0] < other.box[2]:
            h = h1 if self.box[2] < other.box[2] else other.box[2] - self.box[0]
        if self.box[1] < other.box[1]:
            if other.box[1] < self.box[3] < other.box[3]:
                w = self.box[3] - other.box[1]
            elif self.box[3] >= other.box[3]:
                w = w2
        elif self.box[1] < other.box[3]:
            w = w1 if self.box[3] < other.box[3] else other.box[3] - self.box[1]
        intersection = h * w
        return intersection / (union - intersection)


    def add_next(self, child):
        """
        add new node to cur node
        :param child:
        :return:
        """
        fathers = self.father.intersection(child.father)

        for f in fathers:
            assert isinstance(f, TreeNode)
            child.father.remove(f)
            f.children.remove(child)
        child.father.add(self)
        self.children.add(child)

    def get_bottom(self):
        if len(self.children) == 0:
            return self
        return list(self.children)[0].get_bottom()

    def get_root(self):
        if len(self.father) == 0:
            return self
        return list(self.father)[0].get_root()

    def crop(self, bottoms: set, reverse=False):
        """
        :param bottoms:set of vessls
        :param reverse:root->bottom or bottom->root
        :return:
        """
        nexts = self.father if reverse else self.children
        max_depth = 0
        bottom = self
        max_iou = 0
        next_node = None
        for node in nexts:
            bottom_node = node.crop(bottoms, reverse=reverse)
            depth = abs(self.idx - bottom_node.idx)
            if depth > max_depth:
                max_depth = depth
                bottom = bottom_node
                max_iou = self.iou(node)
                next_node = node
            elif depth == max_depth:
                iou_ = self.iou(node)
                if abs(self.idx - node.idx) < abs(self.idx - next_node.idx) or (
                        abs(self.idx - node.idx) <= abs(self.idx - next_node.idx) and iou_ > max_iou):
                    if bottom_node != bottom and bottom in bottoms:
                        bottoms.remove(bottom)
                    max_iou = iou_
                    bottom = bottom_node
                    next_node = node

        nexts.clear()
        if next_node is not None:
            nexts.add(next_node)

        return bottom


# change json to node list
def list2node(vol: list):
    nodes = []
    for i, box_info in enumerate(vol):
        nodes.append([])
        for box in box_info['box']:
            nodes[i].append(TreeNode(i, box))
    return nodes


def interp1d(xs: list, ys: list, kind='linear') -> Tuple[np.ndarray, np.ndarray]:
    """
    :param xs:
    :param ys:
    :param kind:
    :return:
    """
    xs = np.array(xs)
    ys = np.array(ys)
    f = interpolate.interp1d(xs, ys, kind=kind)
    x_new = np.linspace(xs[0], xs[-1], xs[-1] - xs[0] + 1)
    y_new = f(x_new)
    return x_new, y_new


def smooth(begin: TreeNode):
    mid_xs = [begin.get_mid_point()[0]]
    mid_ys = [begin.get_mid_point()[1]]
    width, height = begin.get_box_size()
    box_wid = [width]
    box_height = [height]
    idxs = [begin.idx]
    node = begin
    while len(node.father) > 0:
        assert len(node.father) == 1
        next_ = list(node.father)[0]
        assert isinstance(next_, TreeNode)
        idxs.append(next_.idx)
        mid_point = next_.get_mid_point()
        mid_xs.append(mid_point[0])
        mid_ys.append(mid_point[1])
        width, height = next_.get_box_size()
        box_wid.append(width)
        box_height.append(height)
        node = next_
    idxs.reverse(), mid_xs.reverse(), mid_ys.reverse(), box_wid.reverse(), box_height.reverse()
    _, mid_xs = interp1d(idxs, mid_xs)
    _, mid_ys = interp1d(idxs, mid_ys)

    _, box_wid = interp1d(idxs, box_wid)
    _, box_height = interp1d(idxs, box_height)

    idxs = np.linspace(idxs[0], idxs[-1], idxs[-1] - idxs[0] + 1).astype('int')

    boxes = {}  # {index:[top，left，bottom，right]}
    for i, index in enumerate(idxs):
        boxes[int(index)] = [int(mid_ys[i] - box_height[i] / 2), int(mid_xs[i] - box_wid[i] / 2),
                             int(mid_ys[i] + box_height[i] / 2), int(mid_xs[i] + box_wid[i] / 2)]
    return boxes


def neck2brain(idx, shape):
    """
    :param idx:
    :param shape:
    :return:
    """
    return shape * 0.63 <= idx <= shape * 0.7


def complete(vol: list, shape, name=''):
    """
    :param vol:a list
    :param name:
    :return:
    """
    print('vessel：', name)
    vol = list2node(vol)
    padding = 30
    roots = set()
    bottoms = set()
    for idx, slice_nodes in enumerate(vol):
        for node in slice_nodes:
            assert isinstance(node, TreeNode)
            if len(node.father) == 0:
                roots.add(node)
            flag = False
            for i in range(idx + 1, min(len(vol), idx + padding)):
                for next_node in vol[i]:
                    assert isinstance(next_node, TreeNode)
                    if node.is_connect(next_node, threshold=0.05 if neck2brain(node.idx, shape[0]) else 0.1):
                        node.add_next(next_node)
                        flag = True
                if flag:
                    for j in range(i + 1, min(len(vol), idx + padding, i + 5)):
                        for next_node in vol[j]:
                            assert isinstance(next_node, TreeNode)
                            if node.is_connect(next_node, threshold=0.05 if neck2brain(node.idx, shape[0]) else 0.1):
                                node.add_next(next_node)
                    break
            if len(node.children) == 0:
                bottoms.add(node)

    for root in roots:
        root.crop(bottoms)
    for bottom in bottoms:
        bottom.crop(roots, reverse=True)

    bottoms = list(bottoms)
    bottoms.sort(key=lambda x: x.idx - x.get_root().idx, reverse=True)

    mask = np.zeros(shape)
    all_boxes = []
    exist_boxes = []
    for i in range(len(bottoms)):
        bottom = bottoms[i]
        length = abs(bottom.idx - bottom.get_root().idx)
        in_brain_flag = bottom.idx >= shape[0] * 0.65
        if (length < 100 and not in_brain_flag) or length < 30:
            continue
        if len(exist_boxes) >= 2:
            flag = True
            for j in exist_boxes:
                if bottom.is_left() == bottoms[j].is_left():
                    assert bottom.idx > bottom.get_root().idx
                    assert bottoms[j].idx > bottoms[j].get_root().idx
                    if in_brain_flag and neck2brain(bottom.get_root().idx, shape[0]):
                        if bottom.idx > bottoms[j].get_root().idx and bottom.get_root().idx < bottoms[j].idx - 20:
                            flag = False
                            break
                        r = bottom.get_root()
                        while r != bottom and r.idx < bottoms[j].idx:
                            r = list(r.children)[0]
                            r.father.clear()
                    else:
                        if not (bottom.idx <= bottoms[j].get_root().idx or bottom.get_root().idx >= bottoms[j].idx):
                            flag = False
                            break
            if not flag:
                continue
        elif len(exist_boxes) == 1:
            if bottom.is_left() == bottoms[0].is_left():
                continue
        exist_boxes.append(i)
        boxes = smooth(bottoms[i])
        all_boxes.append(boxes)
        for idx in boxes:
            mask[idx][boxes[idx][0]:boxes[idx][2], boxes[idx][1]:boxes[idx][3]] = 1

    left = [512, 512, -1, -1, 1000, -1]
    right = [512, 512, -1, -1, 1000, -1]
    max_flag = [False, False, True, True]
    for boxes in all_boxes:
        xs = []
        tmp = [512, 512, -1, -1]
        for box in boxes:
            xs.append((boxes[box][1] + boxes[box][3]) / 2)
            for i in range(4):
                tmp[i] = max(tmp[i], boxes[box][i]) if max_flag[i] else min(tmp[i], boxes[box][i])
        new_box = left if sum(xs) / len(xs) < 256 else right
        new_box[4] = min(new_box[4], min(boxes.keys()))
        new_box[5] = max(new_box[5], max(boxes.keys()))
        for i in range(4):
            new_box[i] = max(new_box[i], tmp[i]) if max_flag[i] else min(new_box[i], tmp[i])
    padding = 5
    for b in [left, right]:
        for i in [0, 1, 4]:
            b[i] = int(max(b[i] - padding, 0))
        for i in [2, 3]:
            b[i] = int(min(b[i] + padding, 511))
        b[5] = int(min(b[5] + padding, shape[0] - 1))
    boxes_3d = [left, right]
    boxes_2d = all_boxes

    return mask, boxes_3d, boxes_2d


def generate_smooth_res_from_2dyolo():
    d1 = r''
    d2 = r''
    d3 = r''
    d4 = r''
    d5 = r''
    if not os.path.exists(d5):
        os.makedirs(d5)
    cases = os.listdir(d1)
    for c in tqdm(cases):
        s = c.split('-')
        cc = '{:02d}-D-{}'.format(int(s[0]), s[1].replace('.json', '.nii.gz'))
        js = json.load(open(os.path.join(d1, c)))
        ori_mask_vol = sitk.ReadImage(os.path.join(d2, cc))
        mask = sitk.GetArrayFromImage(ori_mask_vol)

        outpath = d3

        assert len(js) == 1
        for k in js:
            box_list = js[k]

            mask, boxes3d, boxes2d = complete(box_list, shape=mask.shape, name=cc)
            boxes3d = {'info': '[y1,x1,y2,x2,z1,z2]', 'box': boxes3d}
            f = open(os.path.join(d5, cc.replace('.nii.gz', '.json')), mode='w')
            padding_2d = 10
            for box2d in boxes2d:
                for z in box2d:
                    for i, add in enumerate([-1, -1, 1, 1]):
                        box2d[z][i] = int(box2d[z][i] + add * padding_2d)
            boxes2d = {'info': '[y1,x1,y2,x2], padding={}'.format(padding_2d), 'box': boxes2d}
            res = {'3d': boxes3d, '2d': boxes2d}
            json.dump(res, f)

        dst_mask_vol = sitk.GetImageFromArray(mask)
        sitk_format(ori_mask_vol,dst_mask_vol)
        sitk.WriteImage(dst_mask_vol, os.path.join(outpath, cc))

    print('finish')


def get_crop_from_3dbox():
    box3d_dir = r''
    vol_dir = r''
    mask_dir = r''
    save_vol_dir = r''
    save_mask_dir = r''
    for c in tqdm(os.listdir(box3d_dir)):
        nii_name = c.replace('.json', '.nii.gz')
        boxes3d = json.load(open(os.path.join(box3d_dir, c)))
        boxes3d = boxes3d['3d']['box']
        vol_itk = sitk.ReadImage(os.path.join(vol_dir, nii_name))
        mask_itk = sitk.ReadImage(os.path.join(mask_dir, nii_name))

        vol = sitk.GetArrayFromImage(vol_itk)
        mask = sitk.GetArrayFromImage(mask_itk)

        for i, box in enumerate(boxes3d):
            crop_vol = vol[box[4]:box[5], box[0]:box[2], box[1]:box[3]]
            crop_mask = mask[box[4]:box[5], box[0]:box[2], box[1]:box[3]]
            crop_vol = sitk.GetImageFromArray(crop_vol)
            crop_mask = sitk.GetImageFromArray(crop_mask)

            sitk_format(vol_itk, crop_vol)
            sitk_format(mask_itk, crop_mask)

            sitk.WriteImage(crop_vol, os.path.join(save_vol_dir, c.replace('.json', '_{}.nii.gz'.format(i))))
            sitk.WriteImage(crop_mask, os.path.join(save_mask_dir, c.replace('.json', '_{}.nii.gz'.format(i))))

    print('finish')


def bilinear_interpolation(image, x, y, min_idx=0, max_idx=511):
    x_floor, y_floor = max(min_idx, int(np.floor(x))), max(min_idx, int(np.floor(y)))
    x_ceil, y_ceil = min(max_idx, int(np.ceil(x))), min(max_idx, int(np.ceil(y)))

    q11 = image[y_floor, x_floor]
    q21 = image[y_floor, x_ceil]
    q12 = image[y_ceil, x_floor]
    q22 = image[y_ceil, x_ceil]

    value = (q11 * (x_ceil - x) * (y_ceil - y) +
             q21 * (x - x_floor) * (y_ceil - y) +
             q12 * (x_ceil - x) * (y - y_floor) +
             q22 * (x - x_floor) * (y - y_floor))

    return value


if __name__ == "__main__":
    get_crop_from_3dbox()

