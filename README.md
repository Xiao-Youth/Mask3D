# Mask3D

This repo contains the implementations of Mask3D. The code is heavily based on [Object DGCNN & DETR3D](https://github.com/WangYueFt/detr3d).

### Prerequisite

1. mmcv (https://github.com/open-mmlab/mmcv)

2. mmdet (https://github.com/open-mmlab/mmdetection)

3. mmseg (https://github.com/open-mmlab/mmsegmentation)

4. mmdet3d (https://github.com/open-mmlab/mmdetection3d)

### Data
1. Follow the mmdet3d to process the data.
2. Fine-tune a panoptic segmentation model on nuImage. We provided a [fine-tuned K-Net model](https://drive.google.com/file/d/1UpcWkxpCoUiQUnzfzPRND4Aio4L9N7lA/view?usp=drive_link) here.
3. Run the script below to obtain the pseudo segmentation annotations and the pkl file that contains the information of them:

   `python tools/segment_nuscenes.py --data-root data/nuscenes --infos-name nuscenes_infos_${train or val}.pkl --output-file nuscenes_infos_${train or val}_seg.pkl`

4. Run the script below to match the segmentation and detection annotations:

   `python tools/det_seg_assign.py --data-root data/nuscenes --ann-path data/nuscenes/nuscenes_infos_${train or val}_seg.pkl `

5. The final dataset directory should be like this:

   ```
   data
     └── nuscenes
         ├── maps
         ├── samples
         ├── segmentations
         ├── sweeps
         ├── v1.0-mini
         ├── v1.0-trainval
         ├── nuscenes_gt_database
         ├── nuscenes_dbinfos_train.pkl
         ├── nuscenes_infos_train.pkl
         ├── nuscenes_infos_train_seg.pkl
         ├── nuscenes_infos_val.pkl
         └── nuscenes_infos_val_seg.pkl
   ```


### Train
1. Downloads the [pretrained backbone weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN?usp=sharing) to ckpts/ 

2. Run the script below:

   `bash tools/dist_train.sh projects/configs/${config_file_name} ${num_GPU} --auto-resume`

### Evaluation
1. Download the weights accordingly.  

   |  Model   | mAP | NDS | Download |
   | :---------: | :----: |:----: | :------: |
   |[Mask3D, ResNet101](./projects/configs/mask3d/mask3d_r101.py)|37.9|43.9|[model](https://drive.google.com/file/d/1q__ZC1BCjcPAzwOyYOIQAVFViLOaFwMU/view?usp=drive_link)
   |[Mask3D, V2-99](./projects/configs/mask3d/mask3d_v2-99.py)|47.7|50.7|[model](https://drive.google.com/file/d/14gavEJ6jMa922SoGE15P88W4DyuEOV_D/view?usp=drive_link)


2. Run the scipt below:

   `bash tools/dist_test.sh projects/configs/${config_file_name} ${num_GPU} --checkpoint ${checkpoint_path} --eval=bbox`

 ### Visualize
 1. To visualize with an opencv window:

      `python tools/test.py projects/configs/${config_file_name} --checkpoint ${checkpoint_path} --show`
 2. To output images for visulization:
      
      `python tools/test.py projects/configs/${config_file_name} --checkpoint ${checkpoint_path} --show --show-dir ${output directory}`
   