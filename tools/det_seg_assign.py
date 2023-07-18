import imp
import mmcv
import numpy as np
import torch
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
from mmdet.core.mask.utils import mask2bbox
from tqdm import trange
from nuscenes.nuscenes import NuScenes
from typing import List, Tuple, Union
from scipy.optimize import linear_sum_assignment
from shapely.geometry import MultiPoint, box
from pyquaternion.quaternion import Quaternion
from collections import OrderedDict
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
import argparse


tmp_mask_bbox_path='gt_mask_bbox.pkl'
tmp_bbox2d_path='gt_bbox2d.pkl'

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'barrier', 'flat', 'background'
]

cams=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']

def parse_args():
    parser = argparse.ArgumentParser(description='Output Image Segmentation Annotation')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/nuscenes',
        help='specify the root path of dataset')
    parser.add_argument(
        '--ann-path',
        type=str,
        default='data/nuscenes/nuscenes_infos_val_seg.pkl',
        help='specify the name of pkl file which contains the infos of nuscenes'
    )
    parser.add_argument(
        '--version',
        type=str,
        nargs='+',
        default=['v1.0-trainval'],
        required=False,
        help='specify the dataset version')
    args = parser.parse_args()
    return args



def Cls_Cost(labels1,labels2):
    m=len(labels1)
    n=len(labels2)
    cost=labels1.unsqueeze(1).repeat(1,n)
    cost=(cost==labels2)+0
    return cost

def mask_to_bbox(ann_file,args):
    gt_mask_bbox=[]
    for i in trange(len(ann_file['infos'])):
        gt_seg_label=[]
        gt_seg_bbox=[]
        info=ann_file['infos'][i]
        for cam_type, cam_info in info['cams'].items():
            filename = cam_info['data_path'].split('nuscenes/')[-1]
            pngname = filename.replace('jpg', 'png')
            pngpath = args.data_root+f'segmentations/{pngname}'
            single_gt_seg_mask = []
            segimage = mmcv.imread(pngpath)
            single_gt_seg_label = []
            seginfolist = info['segmentation'][cam_type]['segmentation']

            segimagecopy = segimage.astype(np.uint32)
            encodeimage = segimagecopy[:, :, 0] * 256 * 256 + segimagecopy[:, :, 1] * 256 + segimagecopy[:, :, 2]
            for seginfo in seginfolist:
                label = seginfo['category_id']
                single_gt_seg_label.append(label)

                segid = seginfo['id']

                mask = (encodeimage == segid)+0
                single_gt_seg_mask.append(mask)
                # a = True+0

            gt_seg_label.append(torch.tensor(np.array(single_gt_seg_label)))
            gt_seg_bbox.append(mask2bbox(torch.tensor(np.array(single_gt_seg_mask))))
        gt_mask_bbox.append({'bbox':gt_seg_bbox,'label':gt_seg_label})

    mmcv.dump(gt_mask_bbox,tmp_mask_bbox_path)

def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

def get_2d_boxes(nusc, sample_token: str, visibilities: List[str]) -> List[OrderedDict]:
    # sd_rec = nusc.get('sample_data', sample_data_token)

    s_rec = nusc.get('sample', sample_token)

    cams = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
    cam_bbox2d=list()
    cam_vel = list()
    cam_ind = list()
    lid = nusc.get('sample_data', s_rec['data']['LIDAR_TOP'])
    lidar_path, boxes, _ = nusc.get_sample_data(lid['token'])
    # boxes = [box for box in boxes if (nusc.get('sample_annotation', box.token)['visibility_token'] in ['3','4'])]
    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = np.array([b.orientation.yaw_pitch_roll[0]
                        for b in boxes]).reshape(-1, 1)
    gt_boxes_lidar = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
    for cam in cams:
        sd_rec = nusc.get('sample_data', s_rec['data'][cam])
        # Get the calibrated sensor and ego pose record to get the transformation matrices.
        cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        # pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

        # Get all the annotation with the specified visibilties.
        ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
        # ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in ['3','4'])]
        repro_recs = []
        vel = []
        ind = []

        index = -1
        for ann_rec in ann_recs:
            index = index + 1
            # Augment sample_annotation with token information.
            ann_rec['sample_annotation_token'] = ann_rec['token']
            # ann_rec['sample_data_token'] = sample_data_token

            # Get the box in global coordinates.
            # box = nusc.get_box(ann_rec['token'])
            record = nusc.get('sample_annotation', ann_rec['token'])
            box = Box(
                record['translation'],
                record['size'],
                Quaternion(record['rotation']),
                velocity=nusc.box_velocity(ann_rec['token']),
            )

            # Move them to the ego-pose frame.
            box.translate(-np.array(pose_rec['translation']))
            box.rotate(Quaternion(pose_rec['rotation']).inverse)

            # Move them to the calibrated sensor frame.
            box.translate(-np.array(cs_rec['translation']))
            box.rotate(Quaternion(cs_rec['rotation']).inverse)

            # Filter out the corners that are not in front of the calibrated sensor.
            corners_3d = box.corners()
            in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
            corners_3d = corners_3d[:, in_front]

            # Project 3d box to 2d.
            corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

            # Keep only corners that fall within the image.
            final_coords = post_process_coords(corner_coords)

            # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
            if final_coords is None:
                continue
            else:
                min_x, min_y, max_x, max_y = final_coords
                box_velocity = box.velocity
                box_index = index

            # Generate dictionary record to be included in the .json file.
            # repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec['filename'])
            bbox2d = [min_x, min_y, max_x, max_y]
            repro_recs.append(bbox2d)
            vel.append(box_velocity)
            ind.append(box_index)
        cam_bbox2d.append(repro_recs)
        cam_vel.append(vel)
        cam_ind.append(ind)

    return cam_bbox2d, cam_vel, gt_boxes_lidar, cam_ind

def bbox3d_to_2d(ann_file,args):
    nusc = NuScenes(version=args.version, dataroot = args.data_root, verbose=True)
    bbox2_and_vel = list()
    for i in trange(len(ann_file['infos'])):
        gen_bbox2d, vel_bbox2d, gt_boxes_lidar,index = get_2d_boxes(nusc,ann_file['infos'][i]['token'],[1,2,3,4])

        bbox2_and_vel.append({'bbox2d':gen_bbox2d,'index_bbox2d':index,'velocity':vel_bbox2d,'gt_boxes_lidar':gt_boxes_lidar})
    mmcv.dump(bbox2_and_vel,tmp_bbox2d_path)

def match(ann_file,mask_bbox,bbox3d_bbox2d,args):
    for i in trange(len(bbox3d_bbox2d)):
        single_gt_assigned_masks=np.full([len(bbox3d_bbox2d[i]['gt_boxes_lidar']),6],-1)
        for vi in range(len(cams)):
            bbox2d=torch.tensor(np.array(bbox3d_bbox2d[i]['bbox2d'][vi]))
            inds=bbox3d_bbox2d[i]['index_bbox2d'][vi]
            labels=ann_file['infos'][i]['gt_names'][inds]
            labels=torch.tensor([class_names.index(label) if label in class_names else -1 for label in labels])
            if len(bbox2d)==0:
                continue
            bbox_mask=mask_bbox[i]['bbox'][vi]
            mask_labels=torch.tensor(mask_bbox[i]['label'][vi])
            IOU=BboxOverlaps2D()
            iou_cost=IOU(bbox2d,bbox_mask)
            cls_cost=Cls_Cost(labels,mask_labels)
            cost=-iou_cost-cls_cost
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)

            for j in range(len(matched_row_inds)):

                if cost[matched_row_inds[j],matched_col_inds[j]]>-1.2:
                    continue
                single_gt_assigned_masks[inds[matched_row_inds[j]]][vi]=matched_col_inds[j]

        ann_file['infos'][i]['gt_assigned_masks']=single_gt_assigned_masks

    mmcv.dump(ann_file,args.ann_path)

if __name__ == '__main__':
    args = parse_args()
    ann_file=mmcv.load(args.ann_path)
    print('Start generating 2d bbox for mask.')
    mask_to_bbox(ann_file,args)
    print('Start generating 2d bbox for bbox3d.')
    bbox3d_to_2d(ann_file,args)
    print('Start matching.')
    mask_bbox=mmcv.load(tmp_mask_bbox_path)
    bbox2d=mmcv.load(tmp_bbox2d_path)
    match(ann_file,mask_bbox,bbox2d,args)
