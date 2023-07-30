# CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh \
#     configs/top_down/srpose/coco/hrt_small_srpose_coco_256x192.py \
#     /home/whn@mcg/ViTPose/ckp/hrt_small_lr_higher4_kp_multi32_sigma8_coco_256x192/best.pth \
#     4

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh \
#     configs/top_down/srpose/coco/hrnet_w32_srpose_coco_256x192.py \
#     /home/whn@mcg/ViTPose/ckp/hrnet_w32_lr_higher4_kp_multi32_2d_sigma8_coco_256x192/best_AP_epoch_100.pth \
#     4

CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh \
    configs/top_down/srpose/coco/res50_srpose_coco_256x192.py \
    SRPose_res50.pth \
    4

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh \
#     configs/bottom_up/sr/coco/res50_sr_coco_512x512.py \
#     /home/whn@mcg/ViTPose/ckp/res50_higher_kp_coco_512x512/best_AP_epoch_300.pth \
#     4

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh \
#     configs/bottom_up/sr/coco/hrnet_w32_sr_coco_512x512.py \
#     /home/whn@mcg/ViTPose/ckp/hrnet_w32_higher_coco_512x512/best_AP_epoch_250.pth \
#     4
