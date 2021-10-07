# python train.py --gpu 0-3 --stage lixel 
# python train.py --gpu 0-3 --stage param --continue

sleep 7200 &
wait 
echo 'start train'
python train.py --gpu 0-3 --stage lixel 