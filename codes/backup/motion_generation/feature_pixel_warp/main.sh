python3 dist_train.py --gpu_ids 4,5,6,7 --master_port 10091\
        --dataset_mode    ProcessedVideo+Place2 \
        --dataset_dirs    ./datasets/iPER ./datasets/MotionSynthetic ./datasets/fashionvideo \
        --background_dir  ./datasets/places  \
        --output_dir      ./experiments \
        --model_id   AttLWB_iPER+MS+Fashion+Place2_2 \
        --image_size 512 \
        --num_source 4   \
        --time_step  2   \
        --batch_size 1   --Train.niters_or_epochs_no_decay 400000 \
        --gen_name pixel_warp

