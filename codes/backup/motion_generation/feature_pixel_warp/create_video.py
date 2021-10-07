# import numpy as np
# import cv2
# import os
# size = (1920,1080)#这个是图片的尺寸，一定要和要用的图片size一致
# video_name = 'outdoors_fencing_01'
# file_dir = os.path.join('../output/vis', video_name)

# #完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
# videowrite = cv2.VideoWriter(os.path.join('./', video_name)+'.mp4',-1,10,size)#20是帧数，size是图片尺寸

# list=[]
# for root,dirs,files in os.walk(file_dir):
#     for file in files:
#        list.append(file)  #获取目录下文件名列表
# print(len(list))

# for i in range(len(list)):
#     img=cv2.imread(os.path.join(file_dir, list[i]))  #读取图片
#     img=cv2.resize(img,size) #将图片转换为1280*720
#     videowrite.write(img)   #写入视频

# videowrite.release()
# print('end!')


import os
import cv2
# video_name = 's_11_act_16_subact_02_ca_04' # outdoors_fencing_01
# file_dir = os.path.join('../output/vis', video_name)
file_dir = '/home/chenkanghao/mywork/iPER/iPERCore_uvmap/results/primitives/iper1/synthesis/imitations/iper1-akun_2'

size = (1000,1000)#这个是图片的尺寸，一定要和要用的图片size一致

img_list=[]
for root,dirs,files in os.walk(file_dir):
    for file in files:
       img_list.append(file)  #获取目录下文件名列表
img_list = list(sorted(img_list))
print(img_list)

# video_name = 's_11_act_16_subact_02_ca_04' # outdoors_fencing_01
# file_dir_bl = os.path.join('/home/chenkanghao/mywork/pose_estimate/I2L-MeshNet_bl/output/vis', video_name)
file_dir_bl = '/home/chenkanghao/mywork/iPER/iPERCore_uvmap/results/primitives/iper1_baseline/synthesis/imitations/iper1-akun_2'

size = (1000,1000)#这个是图片的尺寸，一定要和要用的图片size一致

img_list_bl=[]
for root,dirs,files in os.walk(file_dir_bl):
    for file in files:
       img_list_bl.append(file)  #获取目录下文件名列表
img_list_bl = list(sorted(img_list_bl))
print(img_list_bl)

assert len(img_list) == len(img_list_bl)

video=cv2.VideoWriter('./iper1.avi',cv2.VideoWriter_fourcc(*'MJPG'),30,(2000,1000))  #定义保存视频目录名称及压缩格式，fps=10,像素为1280*720
for i in range(1,len(img_list)):
    img=cv2.imread(os.path.join(file_dir, img_list[i-1]))  #读取图片
    img_bl=cv2.imread(os.path.join(file_dir_bl, img_list_bl[i-1]))
    img=cv2.resize(img,size) #将图片转换为1280*720
    img_bl=cv2.resize(img_bl,size)
    img = cv2.hconcat([img_bl,img])

    video.write(img)   #写入视频

video.release()

