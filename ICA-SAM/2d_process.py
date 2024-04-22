import numpy as np
import os
import nibabel as nib
import imageio
from tqdm import tqdm
from glob import glob
import json
from skimage.transform import resize
import random


def mid_exp_crop(img_fdata, k, y1, x1, y2, x2, target_x, target_y):
    y1n = int(round(y2 + y1 - target_y) / 2.)
    x1n = int(round(x2 + x1 - target_x) / 2.)
    return img_fdata[x1n:x1n+target_x, y1n:y1n+target_y, int(k)].astype(np.uint8)


def resize_padding(img_fdata, k, y1, x1, y2, x2, target_x, target_y):
    width = x2 - x1
    height = y2 - y1
    ratio = min(target_x / width, target_y / height)
    new_width = round(width * ratio)
    new_height = round(height * ratio)
    tmp_slice = img_fdata[x1:x2, y1:y2, int(k)].astype(np.uint8)
    tmp_slice = resize(tmp_slice, (new_width, new_height))
    padding_slice = np.zeros((target_x, target_y), dtype=np.uint8)
    padding_slice[round(target_x - new_width)//2:round(target_x + new_width)//2, round(target_y - new_height)//2:round(target_y + new_height)//2] = tmp_slice
    return padding_slice


def crop_cond(y1, x1, y2, x2, target_x, target_y):
    if abs(x1 - x2) <= target_x and abs(y1 - y2) <= target_y:
        return True
    else:
        return False


def nii_crop2png(root_dir, save_dir,crop_mask=True, crop_img=True, target_x = 64, target_y = 64):
    print("Start crop nii2png...")
    img_nii_dir = os.path.join(root_dir, 'dicom2nii')
    img_save_dir = os.path.join(save_dir, 'images')
    mask_nii_dir = os.path.join(root_dir, 'mask/wall_cal_lumen_mask_nii')
    mask_save_dir = os.path.join(save_dir, 'labels')
    json_dir = os.path.join(root_dir, 'final_json_res')
    jsons = os.listdir(json_dir)
    label_list = [1,2,3]
    img2label = {}
    cnt = 0
    for j in tqdm(jsons):
        cnt = cnt + 1
        json_info_path = os.path.join(json_dir, j)
        f_idx = int(j.split('-')[0])  # CT样例编号
        mask_nii_path = glob(os.path.join(mask_nii_dir, r'{:02d}-*.nii.gz'.format(f_idx)))[0]
        img_nii_path = glob(os.path.join(img_nii_dir, r'{:02d}-*.nii.gz'.format(f_idx)))[0]
        mask = nib.load(mask_nii_path)  # 读取nii
        mask_fdata = mask.get_fdata()
        img = nib.load(img_nii_path)
        img_fdata = img.get_fdata()
        with open(json_info_path) as f:
            info = json.load(f)
            z_min = 1024
            z_max = 0
            for item in info["3d"]["box"]:
                if item[4] < z_min:
                    z_min = item[4]
                if item[5] > z_max:
                    z_max = item[5]
            for z in range(z_min, z_max + 1):
                mask_slice = mask_fdata[:,:,int(z)].astype(np.uint8)
                if len(np.unique(mask_slice)) == 1:   # no label, all 0
                    continue
                else:
                    if not os.path.exists(img_save_dir):
                        os.makedirs(img_save_dir)
                    img_slice = img_fdata[:,:,int(z)].astype(np.uint8)
                    imageio.imwrite(os.path.join(img_save_dir,r'{:02d}-z{}.png'.format(f_idx, z)), img_slice)
                    if 'data_vessel/images/' + r'{:02d}-z{}.png'.format(f_idx, z) not in img2label.keys():
                        img2label['data_vessel/images/' + r'{:02d}-z{}.png'.format(f_idx, z)] = []
                    if not os.path.exists(mask_save_dir):
                        os.makedirs(mask_save_dir)
                    for l in np.unique(mask_slice)[1:]:
                        tmp_mask_slice = mask_slice.copy()
                        tmp_mask_slice[tmp_mask_slice == l] = 255
                        tmp_mask_slice[tmp_mask_slice != 255] = 0
                        imageio.imwrite(os.path.join(mask_save_dir,r'{:02d}-z{}_{}.png'.format(f_idx, z, l)), tmp_mask_slice)
                        img2label['data_vessel/images/' + r'{:02d}-z{}.png'.format(f_idx, z)].append('data_vessel/labels/' +r'{:02d}-z{}_{}.png'.format(f_idx, z, l))
        if cnt == 10:
            break
    keys = list(img2label.keys())
    random.shuffle(keys)
    split_point = int(len(keys) * 0.8)
    train_keys = keys[:split_point]
    test_keys = keys[split_point:]
    train_dict = {key:img2label[key] for key in train_keys}
    test_dict = {key:img2label[key] for key in test_keys}
    reversed_test_dict = {}
    for key, values in test_dict.items():
        for value in values:
            if value in reversed_test_dict:
                reversed_test_dict[value].append(key)
            else :
                reversed_test_dict[value] = key
    with open(os.path.join(save_dir, 'image2label_train.json'), "w") as f:
        json_str = json.dumps(train_dict)
        f.write(json_str)
    with open(os.path.join(save_dir, 'label2image_test_save.json'), "w") as f:
        json_str = json.dumps(reversed_test_dict)
        f.write(json_str)
    print("Finish crop nii2png!")


def concat_img(root_dir, target_x, target_y):
    save_json_path = os.path.join(root_dir, 'concat_img_nc_info.json')
    mask_png_dir = os.path.join(root_dir, 'mask2dresize_nc')
    img_png_dir = os.path.join(root_dir, 'img2dresize_nc')
    save_mask_dir = os.path.join(root_dir, 'mask2dconcat_nc')
    save_img_dir = os.path.join(root_dir, 'img2dconcat_nc')
    mask_pngs = os.listdir(mask_png_dir)
    x_num = target_x // 64   # img: 64x64
    y_num = target_y // 64
    cnt = 0
    team_num = 0
    img_concat_info = {}
    save_img = np.zeros((target_x, target_y), dtype=np.uint8)
    save_mask = np.zeros((target_x, target_y), dtype=np.uint8)
    for n in tqdm(mask_pngs):
        if n[0] != '-':
            mask_png_path = os.path.join(mask_png_dir, n)
            img_png_path = os.path.join(img_png_dir, n)
            mask_png = imageio.imread_v2(mask_png_path)
            img_png = imageio.imread_v2(img_png_path)
            x1 = (cnt % x_num)*64
            x2 = (cnt % x_num + 1)*64
            y1 = (cnt // x_num) % y_num * 64
            y2 = ((cnt // x_num) % y_num + 1) * 64
            # print("%d %d %d %d %d", cnt, x1, x2, y1, y2)
            save_mask[x1:x2, y1:y2] = mask_png
            save_img[x1:x2, y1:y2] = img_png
            if cnt % (x_num * y_num) == (x_num * y_num - 1):
                if len(np.unique(save_mask)) != 1:    # mask is not null
                    if team_num not in img_concat_info.keys():
                        img_concat_info[team_num] = []
                    img_concat_info[team_num].append(n)
                    if not os.path.exists(save_mask_dir):
                        os.makedirs(save_mask_dir)
                    imageio.imwrite(os.path.join(save_mask_dir, r'{:02d}.png'.format(team_num)), save_mask)
                    if not os.path.exists(save_img_dir):
                        os.makedirs(save_img_dir)
                    imageio.imwrite(os.path.join(save_img_dir, r'{:02d}.png'.format(team_num)), save_img)
                    team_num = team_num + 1
            cnt = cnt + 1
    imageio.imwrite(os.path.join(save_mask_dir, r'{:02d}.png'.format(team_num)), save_mask)
    imageio.imwrite(os.path.join(save_img_dir, r'{:02d}.png'.format(team_num)), save_img)
    with open(save_json_path, "w") as f:
        json_str = json.dumps(img_concat_info)
        f.write(json_str)


def crop2size(root_dir):
    img_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')
    for f in tqdm(os.listdir(label_dir)):
        img_path = os.path.join(label_dir, f)
        img = imageio.imread_v2(img_path)
        if len(np.unique(img[128:384,128:384])) == 1:
            print(f)
        imageio.imwrite(img_path,img[128:384,128:384])


if __name__ == '__main__':
    root_dir = r"/home/lenovo/pzy/MedSAM/data/Vessel_CT/processed"
    save_dir = r"/home/lenovo/pzy/SAM-Med2D/data_vessel"
    # nii_crop2png(root_dir, save_dir, True, True)
    # concat_img(root_dir, 1024, 1024)
    crop2size(save_dir)
