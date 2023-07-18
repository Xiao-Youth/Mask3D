import os
from turtle import color
import argparse
import base64
from email.mime import image
from os import path as osp

import sys
from unittest import result
import random
import mmcv
import numpy as np
from nuimages import NuImages
from nuimages.utils.utils import mask_decode, name_to_index_mapping
from nuscenes.nuscenes import NuScenes
import shutil
import asyncio
from argparse import ArgumentParser

import time



from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot, single_gpu_test)
PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208), (255, 255, 128), (147, 211, 203),
               (150, 100, 100), (168, 171, 172), (146, 112, 198),
               (210, 170, 100), (92, 136, 89), (218, 88, 184), (241, 129, 0),
               (217, 17, 255), (124, 74, 181), (70, 70, 70), (255, 228, 255),
               (154, 208, 0), (193, 0, 92), (76, 91, 113), (255, 180, 195),
               (106, 154, 176),
               (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
               (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
               (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
               (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
               (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
               (146, 139, 141),
               (70, 130, 180), (134, 199, 156), (209, 226, 140), (96, 36, 108),
               (96, 96, 96), (64, 170, 64), (152, 251, 152), (208, 229, 228),
               (206, 186, 171), (152, 161, 64), (116, 112, 0), (0, 114, 143),
               (102, 102, 156), (250, 141, 255)]

categories=['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'barrier','flat','background']

def parse_args():
    parser = argparse.ArgumentParser(description='Output Image Segmentation Annotation')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/nuscenes',
        help='specify the root path of dataset')
    parser.add_argument(
        '--infos-name',
        type=str,
        default='nuscenes_infos_train.pkl',
        help='specify the name of pkl file which contains the infos of nuscenes'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='nuscenes_infos_train_seg.pkl',
        help='specify the name of output pkl file'
    )
    parser.add_argument(
        '--version',
        type=str,
        nargs='+',
        default=['v1.0-trainval'],
        required=False,
        help='specify the dataset version')
    parser.add_argument('--config', type=str, default='K-Net/configs/det/knet/knet_s3_r101_fpn_ms-3x_coco-panoptic.py', help='Config file')

    parser.add_argument('--checkpoint', type=str, default='ckpts/K-Net_fine-tuned-on-nuimage.pth',help='Checkpoint file')

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--extra-tag', type=str, default='nuimages')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    data_root = args.data_root
    infos_name = args.infos_name
    output_file = args.output_file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    if os.path.exists('segment_tmp.pkl'):
        tmp=mmcv.load('segment_tmp.pkl')
        data=tmp['data']
        segment_anno=tmp['segment_anno']
        iter=tmp['iter']
    else:
        data = mmcv.load(os.path.join(data_root,infos_name), file_format='pkl')
        segment_anno = []
        iter=0

    data_infos = data['infos']
    # data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    # data_infos = data_infos[::1]

    total = len(data_infos)

    version = data['metadata']['version']

    start = time.time()
    
    batch_size = 8

    image_root_dirs = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    
    start_iter=iter
    for batchstart in range(start_iter, total, batch_size):
        if batch_size + batchstart > total:
            batchend = total
        else:
            batchend = batch_size + batchstart

        image_abs_dirs = []
        image_tokens = []

        print(f'## from {batchstart} to {batchend - 1} ## total: {total} ## time consumed: {time.time()-start}')
        for batchidx in range(batchstart, batchend):

            for image_root_dir in image_root_dirs:
                image_abs_dirs.append(os.path.join(data_root,data_infos[batchidx]['cams'][image_root_dir]['data_path'].split('nuscenes/')[-1]))
                image_tokens.append(data_infos[batchidx]['cams'][image_root_dir]['sample_data_token'])
        result = inference_detector(model, image_abs_dirs)

        saveidx = batchstart
        cnt = 0
        savesegment = {}
        for residx, sresult in enumerate(result):

            single_anno = {}

            filename = image_abs_dirs[residx].split('nuscenes/')[-1]
            pngname = filename.replace('jpg', 'png')
            type = filename.split('/')[-2]

            single_anno['filename'] = pngname
            single_anno['token'] = image_tokens[residx]

            panres = sresult['pan_results']
            flatten_panres = np.array(panres).flatten()
            list_panres = set(flatten_panres.tolist())

            color_box = []
            H,W = panres.shape
            segmentimage = np.zeros((H,W,3))
            bkmask = np.ones((H,W))

            image_segment_anno = []
            for num in list_panres:
                cat_id = num % 1000
                # if cat_id == 0 or cat_id >= 12:
                if (cat_id >=0 and cat_id <=9) or (cat_id == 80): 
                    color = random.choice(PALETTE)
                    while color in color_box:
                        color = random.choice(PALETTE)
                    color_box.append(color)
                
                    mask = (panres == num)
                    bkmask[mask==1] = 0
                    segmentimage[mask] = color

                    B = color[0]
                    G = color[1]
                    R = color[2]
                    if cat_id == 80:
                        cat_id = 10
                    image_segment_anno.append(
                        dict(
                            category_id = categories[cat_id],
                            id = R+G*256+B*256*256
                        )
                    )
            color = random.choice(PALETTE)
            while color in color_box:
                color = random.choice(PALETTE)
            color_box.append(color)
            segmentimage[bkmask==1] = color
            B = color[0]
            G = color[1]
            R = color[2]                     
            image_segment_anno.append(
                dict(
                    category_id = categories[11],
                    id = R+G*256+B*256*256
                )
            )
            single_anno['segmentation'] = image_segment_anno
            segment_anno.append(single_anno)
            
            savesegment[type] = single_anno

            cnt += 1
            if cnt == 6:
                data['infos'][saveidx]['segmentation'] = savesegment
                savesegment = {}
                saveidx += 1
                cnt = 0

            mmcv.imwrite(segmentimage,f'{data_root}/segmentations/{pngname}')
        iter=iter+batch_size
        if iter%(100*batch_size)==0:
            tmp={'data':data,'segment_anno':segment_anno,'iter':iter}
            mmcv.dump(tmp,'segment_tmp.pkl')

    jsonfile = f'{data_root}/segmentations/{version}/segmentation.json'
    mmcv.dump(segment_anno, jsonfile)
    pklfile =os.path.join(data_root,output_file)
    mmcv.dump(data,pklfile)

if __name__ == '__main__':
    main()