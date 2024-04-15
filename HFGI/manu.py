import argparse
import torch
import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前文件夹路径添加到Python路径中
sys.path.append(current_dir)
sys.path.append(".")
sys.path.append("..")
from .utils.model_utils import setup_model
from .editings import latent_editor
#from .datasets.inference_dataset import InferenceDataset


device = "cuda"
parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--images_dir", type=str, default=None, help="The directory to the images")
parser.add_argument("--save_dir", type=str, default=None, help="The directory to save.")
parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
parser.add_argument("--edit_attribute", type=str, default='smile', help="The desired attribute")
parser.add_argument("--edit_degree", type=float, default=0, help="edit degreee")
parser.add_argument("--ckpt", type=str, default="HFGI/checkpoint/ckpt.pt", help="path to generator checkpoint")
parser.add_argument("--gpus", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--feat", type=bool, default=False)
parser.add_argument("--id", type=bool, default=False)
parser.add_argument("--model", type=str, default="advG")
parser.add_argument("--mask", type=str, default="None")
parser.add_argument("--KPI", type=str, default=False)
args = parser.parse_args()


def get_latents(net, x):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    return codes



def HFGI_Model():
    net,_ = setup_model(args.ckpt, device)
    return net

def HFGI_Fake(img,att,net):
    generator = net.decoder
    generator.eval()
    is_cars = False
    editor = latent_editor.LatentEditor(net.decoder, is_cars)
    latent_codes = get_latents(net, img)
    # set the editing operation


    x = img.to(device).float()
    # calculate the distortion map
    imgs, _ = generator([latent_codes.to(device)],None, input_is_latent=True, randomize_noise=False, return_latents=True)
    #return torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
    res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')

    # produce initial editing image
    # edit_latents = editor.apply_interfacegan(latent_codes[i].to(device), interfacegan_direction, factor_range=np.linspace(-3, 3, num=40))  
    if att[0] == 'age' or att[0] == 'smile':
        img_edit, edit_latents = editor.apply_interfacegan(latent_codes.to(device), att[2], factor=3)
    else:
        img_edit, edit_latents = editor.apply_ganspace(latent_codes.to(device), att[1], [att[2]])

    # align the distortion map
    img_edit = torch.nn.functional.interpolate(torch.clamp(img_edit, -1., 1.), size=(256,256) , mode='bilinear')
    res_align  = net.grid_align(torch.cat((res, img_edit), 1))

    # consultation fusion
    conditions = net.residue(res_align)
    imgs, _ = generator([edit_latents],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
    return torch.nn.functional.interpolate(imgs, size=(256,256) , mode='bilinear')



def HFGI_Features(img,att,net):
    return get_latents(net, img)