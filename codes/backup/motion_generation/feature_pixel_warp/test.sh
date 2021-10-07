python3 motion_imitate.py --gpu_ids 4 \
   --image_size 512 \
   --num_source 1   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "google_1" \
   --src_path   "path?=/home/chenkanghao/mywork/iPER/iPERCore_pixel_warp/assets/samples/sources/google_1,name?=google_1" \
   --ref_path   "path?=./assets/samples/references/akun_2.mp4,name?=akun_2,pose_fc?=300" \
   --gen_name pixel_warp