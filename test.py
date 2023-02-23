import argparse
import torch
from torch.autograd import Variable
import torchvision
from torchvision.ops import nms
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage.draw import rectangle_perimeter
import cv2

from models.HRCenterNet import HRCenterNet

input_size = 512
output_size = 128

test_tx = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('divece: ', device)

# 二值化 line 49
def easy_binarization(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR to GRAY
    img_gray[img_gray>127] = 255
    img_gray[img_gray<=127] = 0
    return img_gray

def main(args):
    
    if not (args.log_dir == None):
        print("Load checkpoint from " + args.log_dir)
        checkpoint = torch.load(args.log_dir, map_location="cpu")    
    
    model = HRCenterNet()
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    for file in os.listdir(args.data_dir): # 拿到每一张图片file
        # 先二值化 存下来（这样总错不了了吧）
        img_cv2 = cv2.imread(args.data_dir + file)  # cv打开
        img_bin = easy_binarization(img_cv2)
        img_bin = 255 - img_bin        
        cv2.imwrite(args.output_dir + file, img_bin)
        
        # 原操作
        img = Image.open(args.data_dir + file).convert("RGB")  # pil打开
        img_bin = Image.open(args.output_dir + file).convert("RGB")
        
        # img.show()
        # img_bin.show()
    
        image_tensor = test_tx(img_bin)
        image_tensor = image_tensor.unsqueeze_(0)
        inp = Variable(image_tensor)
        inp = inp.to(device, dtype=torch.float)
        predict = model(inp)
        
        out_img = _nms(args, img_bin, predict, nms_score=0.3, iou_threshold=0.1, original_img=img) # 输出的是bin版本
        print('saving image to ', args.output_dir + file )
        Image.fromarray(out_img).save(args.output_dir + file)
        img_bin.save(args.output_dir + 'number.jpg')
        print('saving number image to ', args.output_dir)
        
    
    
def _nms(args, img, predict, nms_score, iou_threshold, original_img):
    
    draw = ImageDraw.Draw(img)
    bbox = list()
    score_list = list()
    im_draw = np.asarray(torchvision.transforms.functional.resize(img, (img.size[1], img.size[0]))).copy()
    
    heatmap=predict.data.cpu().numpy()[0, 0, ...]
    offset_y = predict.data.cpu().numpy()[0, 1, ...]
    offset_x = predict.data.cpu().numpy()[0, 2, ...]
    width_map = predict.data.cpu().numpy()[0, 3, ...]
    height_map = predict.data.cpu().numpy()[0, 4, ...]
    
    
    for j in np.where(heatmap.reshape(-1, 1) >= nms_score)[0]:

        row = j // output_size 
        col = j - row*output_size
        
        bias_x = offset_x[row, col] * (img.size[1] / output_size)
        bias_y = offset_y[row, col] * (img.size[0] / output_size)

        width = width_map[row, col] * output_size * (img.size[1] / output_size)
        height = height_map[row, col] * output_size * (img.size[0] / output_size)

        score_list.append(heatmap[row, col])

        row = row * (img.size[1] / output_size) + bias_y
        col = col * (img.size[0] / output_size) + bias_x

        top = row - width // 2
        left = col - height // 2
        bottom = row + width // 2
        right = col + height // 2

        start = (top, left)
        end = (bottom, right)

        bbox.append([top, left, bottom, right])
        
    _nms_index = torchvision.ops.nms(torch.FloatTensor(bbox), scores=torch.flatten(torch.FloatTensor(score_list)), iou_threshold=iou_threshold)
    
    # 建立一个列表存下所有grid的坐标
    ls = []
    for k in range(len(_nms_index)):
    
        top, left, bottom, right = bbox[_nms_index[k]]
        
        start = (top, left)
        end = (bottom, right)
        
        rr, cc = rectangle_perimeter(start, end=end,shape=(img.size[1], img.size[0]))
        
        im_draw[rr, cc] = (255, 0, 0)
        
        # Modified 2
        # start 加几圈红色更加显眼
        circle = 6
        for m in range(circle):
            start = (top-m, left-m)
            end = (bottom+m, right+m)
            rr_, cc_ = rectangle_perimeter(start, end=end, shape=(img.size[1], img.size[0]))
            im_draw[rr_, cc_] = (255, 0, 0)
        # end
        
        # Modified 4
        # start 再加一下grid的数字（推理分割的时候不要写字）
        textsize = 50
        ft = ImageFont.truetype(r"E:\Users\momentum\Downloads\font\苹方-简.ttf", size = textsize)
        draw.text(xy=(max(cc), max(rr)), text=str(k), font = ft, fill=(255, 0, 0))
        # end
        
        # Modified 3
        # start 保留二值化后的单图 TODO  img对应的是bin的图，original_img对应的是原图
        print(f'-------------------{k}th---------------------')
        # print(f'图片尺寸{img.size}')
        # print(f'左上角({min(cc)},{min(rr)}), 右下角({max(cc)},{max(rr)})')
        ls.append((min(cc), min(rr), max(cc), max(rr)))  # (x, y, x+w, y+h)
        region = original_img.crop((min(cc), min(rr), max(cc), max(rr)))  # (x, y, x+w, y+h)
        region.save(rf"E:\desktop\new\experiment\out\single\{k}.jpg")        
        # end
        
    return im_draw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRCenterNet.")
    
    parser.add_argument("--data_dir", default = r"E:\desktop\new\experiment\in\\",
                      help="待处理图片文件夹")
    
    parser.add_argument("--log_dir", default=r'E:\desktop\HRCenterNet\weights\HRCenterNet.pth.tar',
                      help="Where to load for the pretrained model.")
    
    parser.add_argument("--output_dir", default = r"E:\desktop\new\experiment\out\\",
                      help="处理后存储图片文件夹Where to save for the outputs.")

    main(parser.parse_args())