import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.autograd import Variable
from utils import utils
import numpy as np
import time
from models.net import net

def load_model(path):
    fuse_net = net()
    fuse_net.load_state_dict(torch.load(path))
    para = sum([np.prod(list(p.size())) for p in fuse_net.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(fuse_net._get_name(), para * type_size / 1024 / 1024))
    fuse_net.eval()
    fuse_net.cuda()
    return fuse_net

def _generate_fusion_image(model, vi, ir):
    out = model(vi, ir)
    return out

def run_demo(model, vi_path, ir_path, output_path_root, index):
    vi_img = utils.get_test_images(vi_path, height=None, width=None)
    ir_img = utils.get_test_images(ir_path, height=None, width=None)
    out = utils.get_image(vi_path, height=None, width=None)
    vi_img = vi_img.cuda()
    ir_img = ir_img.cuda()
    vi_img = Variable(vi_img, requires_grad=False)
    ir_img = Variable(ir_img, requires_grad=False)
    img_fusion = _generate_fusion_image(model, vi_img, ir_img)
    file_name = f'{index:03d}.png'
    output_path = output_path_root + file_name
    if torch.cuda.is_available():
        img = img_fusion.cpu().clamp(0, 255).numpy()
    else:
        img = img_fusion.clamp(0, 255).numpy()
    img = img.astype('uint8')
    utils.save_images(output_path, img, out)
    print(output_path)

def main():
    vi_path = "..."
    ir_path = "..."
    output_path = '...'

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    in_c = 1
    out_c = in_c
    model_path = "./pretrained/Final_epoch_50.model"

    with torch.no_grad():
        model = load_model(model_path)
        for i in range(640):
            index = i + 1
            visible_path = vi_path + '{:03d}.png'.format(index)
            infrared_path = ir_path + '{:03d}.png'.format(index)
            start = time.time()
            run_demo(model, visible_path, infrared_path, output_path, index)
            end = time.time()
            print('time:', end - start, 'S')
    print('Done......')

if __name__ == "__main__":
    main()