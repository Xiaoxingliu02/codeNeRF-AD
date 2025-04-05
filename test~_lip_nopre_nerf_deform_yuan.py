from test_load_audface_multiid import load_audface_data
import os
import sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from natsort import natsorted
from lip_nopre_helpers_deform_yuan import *
import cv2
from utils import cv_utils
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import lip_nopre_model_D
import lip_nopre_model_G as model_G
import loss
import torch.optim as optim
from tensorboardX import SummaryWriter
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def batchify_cache(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs, training = False):

        ret_list = [fn(inputs[i:i+chunk], training=training) for i in range(0, int(inputs.shape[0]), chunk)]

        return torch.cat((ret for ret in ret_list), 0)
    return ret

def batchify(fn, chunk, aud_para, world_fn = lambda x:x, gather_func = None, lip_rect=None):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs, training = False, world_fn=world_fn):
        #add
        embedded = inputs[0]
        attention_poses = inputs[1]
        intrinsic = inputs[2]
        images_features1 = inputs[3]
        G_features = inputs[4]

        pts = inputs[5]
        #print("images_features:",images_features.shape)4 450 450 128
        input_paras1, loss_translation1 = gather_func(world_fn(pts), attention_poses, intrinsic, images_features1,aud_para, lip_rect)
        #input_paras2, loss_translation2 = gather_func(world_fn(pts), attention_poses, intrinsic, G_features, aud_para, lip_rect)
        #print("input_paras:",input_paras.shape)4 57600 130
        loss_translation=(loss_translation1)#+loss_translation2 )/2.0
        ret_list = fn([embedded, input_paras1,G_features, pts], training=training)
        if fn.coarse:
            return ret_list[0], ret_list[1], loss_translation
        else:
            return ret_list[0], None, loss_translation
    return ret


def run_network(inputs, viewdirs, aud_para, fn, embed_fn, embeddirs_fn, netchunk=1024*64, attention_poses=None, intrinsic=None, training=False,
                images_features1=None,G_feature=None,world_fn=None, gather_func=None, lip_rect = None):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    #print(inputs.shape)# 1400 64 3
    embedded = embed_fn(inputs_flat)
    aud = aud_para.unsqueeze(0).expand(inputs_flat.shape[0], -1)
    #print(aud.shape)# 89600 64
    #embedded = torch.cat((embedded, aud), -1)
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat, attention_cache, loss_translation = batchify(fn, netchunk, aud_para, world_fn = world_fn, gather_func = gather_func, lip_rect=lip_rect)([embedded, attention_poses,
                                                                                           intrinsic, images_features1,G_feature, inputs_flat], training)

    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs, attention_cache, loss_translation


def batchify_rays(lip_rect, rays_flat, bc_rgb, aud_para, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    all_loss_translation = []
    #print("rays_flat",rays_flat.shape) 202500 11

    for i in range(0, rays_flat.shape[0], chunk):
        ret, loss_translation = render_rays(rays_flat[i:i+chunk], bc_rgb[i:i+chunk],
                          aud_para, lip_rect=lip_rect, **kwargs)
        all_loss_translation.append(loss_translation)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    loss_translation = torch.mean(torch.stack(all_loss_translation))

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret, loss_translation

import utils.util as util
import torchvision

def _do_if_necessary_saturate_mask(m, saturate=False):
    return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m


def _compute_loss_smooth(mat):
    return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
            torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))
def tensor2im1(img, imtype=np.uint8, unnormalize=True, idx=0, nrows=None):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    if unnormalize:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)

    image_numpy = img.numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t*254.0

    return image_numpy_t.astype(imtype)
def tensor2maskim1(mask, imtype=np.uint8, idx=0, nrows=1):
    im = tensor2im1(mask, imtype=imtype, idx=idx, unnormalize=False, nrows=nrows)
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=-1)
    return im    


def align_face(image):
    #print(image.shape)
    #imageio.imwrite('/fs1/home/tjuvis_2022/lxx/NeRF-pre/test/1.jpg', image)
    eye_detector = cv2.CascadeClassifier("./haarcascade_eye.xml")
    aligned_face=[]
    if image.shape[0]<100:
        for i in range(image.shape[0]):
            gray_face = cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY)
            eyes = eye_detector.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            right_eye_center = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
            left_eye_center = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)
            angle = np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0]) * 180 / np.pi
            M = cv2.getRotationMatrix2D((112 / 2, 112 / 2), angle, 1)
            aligned_face = cv2.warpAffine(image[i], M, (112, 112))
        aligned_face=torch.stack(aligned_face)
        aligned_face=aligned_face.cuda()
    else:
        gray_face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes = eye_detector.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        #print(eyes.shape)
        if type(eyes) == tuple or eyes.shape[0]!=2:
            aligned_face = image
            #imageio.imwrite('/fs1/home/tjuvis_2022/lxx/NeRF-pre/test/'+str(eyes.shape[0])+'.jpg', image)
        else:
            eyes = sorted(eyes, key=lambda x: x[0])
            #print(eyes)
            right_eye_center = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)
            left_eye_center = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
            angle = np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0]) * 180 / np.pi
            M = cv2.getRotationMatrix2D((112 / 2, 112 / 2), angle, 1)
            aligned_face = cv2.warpAffine(image, M, (112, 112))        
    return aligned_face
                


def render_dynamic_face_new(iden_gt_aus,iden,iden_,target,attention_pkl,i_pkl,G_gt_aus,face_real,face_real_,gt_aus,i,H, W, focal, cx, cy, chunk=1024*32, rays=None, bc_rgb=None, aud_para=None,
                        c2w=None, ndc=True, near=0., far=1.,
                        use_viewdirs=False, c2w_staticcam=None, attention_images=None,attention_images_head=None,attention_images_face = None,attention_images_face_=None, attention_poses=None, intrinsic=None, render_pose=None,
                        attention_embed_fn=None, attention_embed_ln=None, feature_extractor=None,feature_extractor2=None,generator=None,lip_encoder=None,lip_encoder_new=None,img_encoder=None, rotation_embed_fn = None, rotation_embed_ln = None, use_render_pose = True, lip_rect=None,head_rect=None, 
                        **kwargs):
    #print("jin")
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w, cx, cy)
        bc_rgb = bc_rgb.reshape(-1, 3)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:#T
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:#F
            print("hhhhh")
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam, cx, cy)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:#F
        # for forward facing scenes
        print("hhh")
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    # print("rays_o",rays_o.shape)202500 3

    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * \
        torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    #module for image feature
    viewpoints = attention_poses[...,3]
    embedded_viewpoints = attention_embed_fn(viewpoints)
    
    

    bc_viewpoints = embedded_viewpoints[:,None,None].expand(attention_images.shape[:-1] + (attention_embed_ln,))
    #bc_viewpoints1 = np.broadcast_to(embedded_viewpoints[:,None,None].cpu().numpy(), attention_images.shape[:-1] + (attention_embed_ln,))    
    #print(bc_viewpoints==torch.Tensor(bc_viewpoints1))
    #print(bc_viewpoints.shape,bc_viewpoints1.shape)
    #bc_viewpoints = torch.broadcast_to(embedded_viewpoints[:,None,None], attention_images.shape[:-1] + (attention_embed_ln,))
    if use_render_pose:# F
        bc_render_transl = np.broadcast_to(attention_embed_fn(render_pose[...,3])[None,None,None].cpu().numpy(), attention_images.shape[:-1] + (attention_embed_ln,))
        
        bc_viewpoints = torch.cat((bc_viewpoints, bc_render_transl), -1)

    #print(attention_images_face.shape,gt_aus.shape)
    fake_imgs, fake_img_mask =generator(Variable(attention_images_face),Variable(gt_aus))
    #print("fake_img_mask",fake_img_mask.shape) 4 450 450 1
    fake_img_mask = _do_if_necessary_saturate_mask(fake_img_mask)
    image_numpy = fake_img_mask.cpu().detach().numpy()
    image_numpy_t = np.transpose(image_numpy, (0,2, 3, 1))
    image_numpy_t = torch.tensor(image_numpy_t).cuda()
    H=fake_img_mask.shape[2]
    W=fake_img_mask.shape[2]
    rect=lip_rect
    #for i4 in range(4):
    #    image_numpy_t[i4,76:110,22:90,0]=0
    '''

    for i4 in range(4):
        image_numpy_t[i4,72:99,22:90,0]=0
        image_numpy_t[i4,0:10,:,0]=1
        image_numpy_t[i4,:,0:12,0]=1
        image_numpy_t[i4,99:112,:,0]=1
        image_numpy_t[i4,:,100:112,0]=1
    '''

    img1=(image_numpy_t) * attention_images_face_
    img2=(1 - image_numpy_t) * attention_images_face_
    #print(face_real.shape)
    




    
    fake_imgs_, fake_img_mask_ =generator(Variable(face_real_.unsqueeze(0)),Variable(G_gt_aus.unsqueeze(0)))
    #print("fake_img_mask",fake_img_mask.shape) 4 450 450 1
    fake_img_mask_ = _do_if_necessary_saturate_mask(fake_img_mask_)
    image_numpy_ = fake_img_mask_.cpu().detach().numpy()
    image_numpy_t_ = np.transpose(image_numpy_, (0,2, 3, 1))
    image_numpy_t_ = torch.tensor(image_numpy_t_).cuda()
    #image_numpy_t_[0][76:110,22:90,0]=0
    imag_real_mask = (1 -image_numpy_t_[0]) * face_real
    



    '''
    G1=[]
    for i6 in range(4):
        #G_feature,G=lip_encoder((1 - image_numpy_t[i6]) * face_real,aud_para)
        G_feature,G=lip_encoder(img2[i6],aud_para)
        G1.append(G[0].permute(1,2,0))
    G1=torch.stack(G1)
    G1=G1.cuda()
    '''
    #target = torch.cat([torch.as_tensor(imageio.imread('/fs1/home/tjuvis_2022/lxx/bla.jpg')).unsqueeze(0) for i in range(4)],0)
    #target = target.float()/255.0
    #print(attention_images.shape)
    for i5 in range(0,4):
        with open(attention_pkl[i5], 'rb') as f:
            faces_info = pickle.load(f)
    
        for face_info in faces_info:
            h,w,x1, y1, x2, y2 = face_info['h'], face_info['w'],face_info['x1'], face_info['y1'], face_info['x2'], face_info['y2']
            resize=transforms.Resize([h,w])
            img__=img1[i5].permute(2,0,1)
            #print(attention_images[i5].shape,img1[i5].shape)
            img__= resize(img__)
            attention_images[i5][y1:y2, x1:x2] =img__.permute(1,2,0)
            #img___=img2[i5].permute(2,0,1)
            #img___= resize(img___)
            #target[i5][y1:y2, x1:x2] = img___.permute(1,2,0)
    #print(imag_real_mask.shape)
    #with open(attention_pkl[i5], 'rb') as f:
    #    faces_info = pickle.load(f)
    #for face_info in faces_info:
    #    h,w,x1, y1, x2, y2 = face_info['h'], face_info['w'],face_info['x1'], face_info['y1'], face_info['x2'], face_info['y2']
    #    target[y1:y2, x1:x2] = cv2.resize(G1,(h,w))
    #image_inputs1 = imag_real_mask.permute(2,0,1)
    #image_inputs1 = image_inputs1.unsqueeze(0).expand(1, image_inputs1.shape[0], image_inputs1.shape[1], image_inputs1.shape[2])
    #G_feature,_=lip_encoder(imag_real_mask,aud_para)
    #aud_para1=aud_para
    #G1=None
    


    fake_imgs_, fake_img_mask_ =generator(Variable(iden_.unsqueeze(0)),Variable(iden_gt_aus.unsqueeze(0)))
    #print("fake_img_mask",fake_img_mask.shape) 4 450 450 1
    fake_img_mask_ = _do_if_necessary_saturate_mask(fake_img_mask_)
    image_numpy_ = fake_img_mask_.cpu().detach().numpy()
    image_numpy_t_ = np.transpose(image_numpy_, (0,2, 3, 1))
    image_numpy_t_ = torch.tensor(image_numpy_t_).cuda()
    #image_numpy_t_[0][66:100,25:90,0]=0
    iden_imag_real_mask = (1 -image_numpy_t_[0]) * iden
    

    _,G=lip_encoder_new(iden_imag_real_mask,aud_para)
    G1 = G[0].permute(1,2,0)
    G_feature,_=lip_encoder(G1,aud_para)


    
    

    rgb_vp1=torch.cat((attention_embed_fn(attention_images),bc_viewpoints),-1)
    rgb_vp1 = rgb_vp1.permute(0, 3, 1, 2)
    #print("rgb_vp",rgb_vp.shape)4 66 450 450
    images_features1 = feature_extractor(rgb_vp1, attention_embed_ln)
    #print("images_features",images_features1.shape)#4 450 450 128

    #rgb_vp2=torch.cat((attention_embed_fn(target),bc_viewpoints),-1)
    #rgb_vp2 = rgb_vp2.permute(0, 3, 1, 2)
    #print("rgb_vp",rgb_vp.shape)4 66 450 450
    #G_feature = feature_extractor2(rgb_vp2, attention_embed_ln)
    #print("images_features",images_features1.shape)#4 450 450 128
    
    all_ret, loss_translation = batchify_rays(lip_rect, rays, bc_rgb, aud_para, chunk,attention_poses=attention_poses,intrinsic=intrinsic,images_features1=images_features1,G_feature=G_feature,**kwargs)

    #print(loss_translation)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'last_weight']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

    return ret_list + [ret_dict] + [loss_translation]+[G1]+[imag_real_mask]# + [mask_loss]

from skimage.metrics import structural_similarity

import lpips
def Cal_SSIM(imageA, imageB):
    imageA = imageA.cpu().detach().numpy()
    imageB = imageB.cpu().detach().numpy()
    #imageA=imageA.transpose((2,0,1))
    #imageB=imageB.transpose((2,0,1))

    #return skimage.measure.compare_ssim(imageB, imageA, data_range=255)
    # BGRshunxu
    #(B1, G1, R1) = cv2.split(imageA)
    #(B2, G2, R2) = cv2.split(imageB)
    imageA = cv2.cvtColor(imageA, cv2.COLOR_RGB2BGR)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_RGB2BGR)
    # convert the images to grayscale BGR2GRAY
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # method1
    (grayScore, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return grayScore
    
def Cal_LPIPS(imageA, imageB):
    loss_fn = lpips.LPIPS(net='alex', version=0.1)
    imageA = imageA.cpu().detach().numpy()
    imageB = imageB.cpu().detach().numpy()
    imageA=imageA.transpose((2,0,1))
    imageB=imageB.transpose((2,0,1))
    imageA=torch.Tensor(imageA).cuda()
    imageB=torch.Tensor(imageB).cuda()
    current_lpips_distance = loss_fn.forward(imageA, imageB)
    return current_lpips_distance


def render_train(img_i,iden,iden_,target,attention_pkl,i_pkl,iden_gt_aus,G_gt_aus,face_real,face_real_,attention_images_face,attention_images_face_,args,  torso_bcs, render_pose, aud_paras, bc_img, hwfcxy, attention_poses, attention_images,attention_images1,gt_aus,intrinsic,
                chunk, render_kwargs, gt_imgs=None,  render_factor=0, lip_rect=None):
    H, W, focal, cx, cy = hwfcxy
    i=img_i

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
    bc_img = torch.Tensor(imageio.imread(torso_bcs[i])).to(device).float() / 255.0
    #bc_img = torch.Tensor(imageio.imread('/fs1/home/tjuvis_2022/lxx/data_util/dataset/Obama_/torso_imgs/6558.jpg')).to(device).float() / 255.0
    '''
    rgb8 = to8b(images_refer[i].cpu().detach().numpy())
    filename = os.path.join('/fs1/home/tjuvis_2022/lxx/NeRF-att/cal_lpips/input_images/', '{:03d}.png'.format(i))
    imageio.imwrite(filename, rgb8)
    '''
    #print(bc_img.shape,torch.as_tensor(images_refer[i]).shape)
    #ssim = structural_similarity(torch.as_tensor(images_refer[i]).cpu().detach().numpy()*255.0, torch.as_tensor(images_refer[i]).cpu().detach().numpy()*255.0,multichannel=True)
    #print("ssim:",ssim.item())
    #lpips=Cal_LPIPS(torch.as_tensor(bc_img)*255.0, torch.as_tensor(images_refer[i])*255.0)
    
    #bc_img = torch.as_tensor(imageio.imread(os.path.join(args.basedir, 'bc.jpg'))).to(device).float()/255.0
    #print("hhh")
    '''
    transform_list = [#transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                           std=[0.5, 0.5, 0.5]),
                      ]
    _transform = transforms.Compose(transform_list)
    
    face_real_path=np.array(face_images_refer)[i]
    face_real_=cv_utils.read_cv2_img(face_real_path)
    face_real_ = _transform(Image.fromarray(face_real_)).cuda()
    
    face_real = torch.as_tensor(imageio.imread(face_real_path)).to(device).float()/255.0
    
    iden_real_path=np.array(face_images_refer)[0]
    iden_=cv_utils.read_cv2_img(iden_real_path)
    iden_ = _transform(Image.fromarray(iden_)).cuda()
    iden = torch.as_tensor(imageio.imread(iden_real_path)).to(device).float()/255.0
    
    G_gt_aus = torch.as_tensor(np.array(gt_aus1)[i]).cuda()
    G_gt_aus= G_gt_aus.type(torch.cuda.FloatTensor)
    iden_gt_aus = torch.as_tensor(np.array(gt_aus1)[0]).cuda()
    iden_gt_aus= iden_gt_aus.type(torch.cuda.FloatTensor)
    i_pkl = np.array(pkl)[i]
    '''
    c2w=render_pose
    rgb, disp, acc, last_weight, _, _,_,_ = render_dynamic_face_new(iden_gt_aus,iden,iden_,target,attention_pkl,i_pkl,G_gt_aus,face_real,face_real_,gt_aus,i,
        H, W, focal, cx, cy, chunk=chunk, c2w=c2w[:3, :4], aud_para=aud_paras, bc_rgb=bc_img,
        attention_poses=attention_poses, attention_images=attention_images,attention_images_head=attention_images1,attention_images_face = attention_images_face,attention_images_face_=attention_images_face_, intrinsic=intrinsic, render_pose=None, lip_rect = lip_rect, **render_kwargs)
    return rgb

def render_path(attention_pkl,pkl,gt_aus1,face_images_refer,attention_images_face,attention_images_face_,args,  images_refer,torso_bcs, render_poses, aud_paras, bc_img, hwfcxy, attention_poses, attention_images,attention_images1,gt_aus,intrinsic,
                chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, lip_rect=None):
    H, W, focal, cx, cy = hwfcxy

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    last_weights = []
    psnr_total=0
    ssim_total=0
    lpips_total=0
    #print("render_poses",render_poses.shape)#728 4 4
    
    for i, c2w in enumerate(tqdm(render_poses)):
        if i>=0:
            bc_img = torch.Tensor(imageio.imread(torso_bcs[i])).to(device).float() / 255.0
            #bc_img = torch.Tensor(imageio.imread(torso_bcs[6561])).to(device).float() / 255.0
            '''
            rgb8 = to8b(images_refer[i].cpu().detach().numpy())
            filename = os.path.join('/fs1/home/tjuvis_2022/lxx/NeRF-att/cal_lpips/input_images/', '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            '''
            #print(bc_img.shape,torch.as_tensor(images_refer[i]).shape)
            #ssim = structural_similarity(torch.as_tensor(images_refer[i]).cpu().detach().numpy()*255.0, torch.as_tensor(images_refer[i]).cpu().detach().numpy()*255.0,multichannel=True)
            #print("ssim:",ssim.item())
            #lpips=Cal_LPIPS(torch.as_tensor(bc_img)*255.0, torch.as_tensor(images_refer[i])*255.0)
            
            #bc_img = torch.as_tensor(imageio.imread(os.path.join(args.basedir, 'bc.jpg'))).to(device).float()/255.0
            #print("hhh")
            transform_list = [#transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
            _transform = transforms.Compose(transform_list)
            
            face_real_path=np.array(face_images_refer)[i]
            face_real_=cv_utils.read_cv2_img(face_real_path)
            face_real_ = _transform(Image.fromarray(face_real_)).cuda()
            
            face_real = torch.as_tensor(imageio.imread(face_real_path)).to(device).float()/255.0
            
            iden_real_path=np.array(face_images_refer)[8]
            iden_=cv_utils.read_cv2_img(iden_real_path)
            iden_ = _transform(Image.fromarray(iden_)).cuda()
            iden = torch.as_tensor(imageio.imread(iden_real_path)).to(device).float()/255.0
            
            G_gt_aus = torch.as_tensor(np.array(gt_aus1)[i]).cuda()
            G_gt_aus= G_gt_aus.type(torch.cuda.FloatTensor)
            iden_gt_aus = torch.as_tensor(np.array(gt_aus1)[8]).cuda()
            iden_gt_aus= iden_gt_aus.type(torch.cuda.FloatTensor)
            i_pkl = np.array(pkl)[i]
            rgb, disp, acc, last_weight, _, _,_,_ = render_dynamic_face_new(iden_gt_aus,iden,iden_,torch.as_tensor(images_refer[i]).cuda(),attention_pkl,i_pkl,G_gt_aus,face_real,face_real_,gt_aus,i,
                H, W, focal, cx, cy, chunk=chunk, c2w=c2w[:3, :4], aud_para=aud_paras[i], bc_rgb=bc_img,
                attention_poses=attention_poses, attention_images=attention_images,attention_images_head=attention_images1,attention_images_face = attention_images_face,attention_images_face_=attention_images_face_, intrinsic=intrinsic, render_pose=None, lip_rect = lip_rect, **render_kwargs)
            rgbs.append(rgb.cpu().numpy())
            
            img_loss = img2mse(torch.as_tensor(rgbs[i]).cuda(), torch.as_tensor(images_refer[i]).cuda())
            psnr = mse2psnr(img_loss)
            ssim = structural_similarity(torch.as_tensor(rgbs[i]).cpu().detach().numpy()*255.0, torch.as_tensor(images_refer[i]).cpu().detach().numpy()*255.0,multichannel=True)
            #lpips=Cal_LPIPS(torch.as_tensor(rgbs[i])*255.0, torch.as_tensor(images_refer[i])*255.0)
            psnr_total=psnr+psnr_total
            ssim_total=ssim+ssim_total
            #lpips_total=lpips_total+lpips
            #print("rgbs:",rgbs.shape)
            #print("images_refer:",images_refer.shape)
            print("psnr:",psnr.item())
            print("ssim:",ssim.item())
            disps.append(disp.cpu().numpy())
            last_weights.append(last_weight.cpu().numpy())
            if i == 0:
                print(rgb.shape, disp.shape)


            if savedir is not None:
                rgb8 = to8b(rgbs[-1])
                filename = os.path.join('/fs1/home/tjuvis_2022/lxx/NeRF-all/call_lpips/Obama_/test/', '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
                rgb8 = to8b(images_refer[i].cpu().numpy())
                filename = os.path.join('/fs1/home/tjuvis_2022/lxx/NeRF-all/call_lpips/Obama_/input/', 'gt{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
            if i == 50:
                break
        else:
            continue

    #print("avg_psnr:",(psnr_total/render_poses.shape[0]).item())
    print("avg_psnr: ",(psnr_total / 51))
    print("avg_ssim: ",(ssim_total / 51))
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    last_weights = np.stack(last_weights, 0)

    return rgbs, disps, last_weights

def load_ckpt(model, ckpt_path):
    old_state_dict = ckpt_path
    cur_state_dict = model.state_dict()
    for param in cur_state_dict:

        old_param = param
        if old_param in old_state_dict and cur_state_dict[param].size()==old_state_dict[old_param].size():
            print("loading param: ", param)
            model.state_dict()[param].data.copy_(old_state_dict[old_param].data)
        else:
            print("warning cannot load param: ", param)

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    attention_embed_fn, attention_embed_ln = get_embedder(5,0)
    attention_embed_fn_2, attention_embed_ln_2 = get_embedder(2,0,9)
    if args.dataset_type == 'llff' or args.use_quaternion or args.use_rotation_embed: #or args.dataset_type == 'shapenet':
        rotation_embed_fn, rotation_embed_ln = get_embedder(2,0,4)
    else:
        rotation_embed_fn, rotation_embed_ln = None, 0

    model_obj = Feature_extractor().to(device)#shoule be: num_embed * embed_ln + num_rot * rotation_embed_ln
    #model_obj__ = Feature_extractor().to(device)

    model_obj2 = Generator().to(device)
    
    grad_vars = list(model_obj.parameters())
    #grad_vars += list(model_obj__.parameters())
    #grad_vars += list(model_obj2.parameters())


    position_warp = Position_warp(255, args.num_reference_images).to(device)
    grad_vars += list(position_warp.parameters())

    hidden_dim = 128
    iters = 2
    num_slots = 2
    num_features = num_slots * hidden_dim
    attention_module = SlotAttention(num_slots, hidden_dim, 130, iters=iters).to(device)
    #attention_module2 = SlotAttention(num_slots, hidden_dim, 130, iters=iters).to(device)
    grad_vars += list(attention_module.parameters())
    #grad_vars += list(attention_module2.parameters())

    
    G_model = model_G.LipGeneratorRNN('reduce', 'reduce', 'reduce', 'GRU',
                                112,256, if_tanh = False)
    lip_encoder_new = model_G.LipGeneratorRNN('reduce', 'reduce', 'reduce', 'GRU',
                                112,256, if_tanh = False)
    img_encoder=model_G.ImageEncoder(112, 256, if_tanh = False)
    #grad_vars += list(G_model.parameters())
    #grad_vars += list(img_encoder.parameters())                          
    nerf_model = Face_Feature_NeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, dim_aud=args.dim_aud,
                     output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, dim_image_features = num_features).to(device)
    nerf_model_with_attention = nerf_attention_model(nerf_model, slot_att=attention_module,embed_fn=attention_embed_fn,
                                                     embed_ln=input_ch, embed_fn_2=attention_embed_fn_2, embed_ln_2=attention_embed_ln_2, coarse=True, num_samples=args.N_samples).to(device)
    grad_vars += list(nerf_model.parameters())

    models = {'model': nerf_model_with_attention, 'attention_model': attention_module}#,'feature_extractor': model_obj,'generator': model_obj2}

    nerf_model_fine = None
    if args.N_importance > 0:
        nerf_model_fine = Face_Feature_NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                                       input_ch=input_ch, dim_aud=args.dim_aud,
                                       output_ch=output_ch, skips=skips,
                                       input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, dim_image_features=num_features).to(device)
        nerf_model_fine_with_attention = nerf_attention_model(nerf_model_fine, attention_module,  attention_embed_fn, input_ch,
                                               attention_embed_fn_2, attention_embed_ln_2, coarse=False, num_samples=args.N_importance+args.N_samples).to(device)
        models['model_fine'] = nerf_model_fine_with_attention
        grad_vars += list(nerf_model_fine.parameters())

    #feature fusion module
    world_fn = lambda x: x
    index_func = make_indices
    def gather_indices(pts, attention_poses, intrinsic, images_features, aud_para, lip_rect):
        H,W = images_features.shape[1:3]
        H=int(H)
        W=int(W)

        indices = torch.round(index_func(pts, attention_poses, intrinsic, H, W)).int()
        #print("indices",indices.shape) 4 57600 2
        indices = indices.long()
        if not args.use_feature_map:
            features = [images_features[i][indices[i][:,0],indices[i][:,1]] for i in range(images_features.shape[0])]
            features = torch.cat([features[i].unsqueeze(0) for i in range(images_features.shape[0])], 0)
            #G_features_ = [G_features[i][indices[i][:,0],indices[i][:,1]] for i in range(G_features.shape[0])]
            #G_features_ = torch.cat([G_features_[i].unsqueeze(0) for i in range(G_features.shape[0])], 0)
        else:
            features = [images_features[i][torch.meshgrid(indices[i][:, 0], indices[i][:, 1])[0].reshape(-1,2)] for i in range(images_features.shape[0])]
            features = torch.cat([features[i].unsqueeze(0) for i in range(images_features.shape[0])], 0)
            #G_features_ = [G_features[i][torch.meshgrid(indices[i][:, 0], indices[i][:, 1])[0].reshape(-1,2)] for i in range(G_features.shape[0])]
            #G_features_ = torch.cat([G_features_[i].unsqueeze(0) for i in range(G_features.shape[0])], 0)

        #3d positional encoding
        embed_fn_warp, input_ch_warp = get_embedder(10, 0)
        translation = torch.stack([position_warp(embed_fn_warp(pts), aud_para.detach(), features[i].detach()) for i in range(args.num_reference_images)])#[4,65536,3]
        #print(translation)
        #rect = lip_rect
        #print("translation",translation.shape)4 57600 2
        loss_translation = torch.mean(translation**2,0)
        #loss_translation = 0
        #indices = indices + translation
        indices = torch.maximum(torch.minimum(indices, torch.Tensor([H - 1., W - 1.])), torch.Tensor([0, 0]))

        def grid_sampler_unnormalize(coord, size, align_corners):
            if align_corners:
                return 2*coord/(size-1)-1
            else:
                return (2*coord+1)/size-1

        indices_ = grid_sampler_unnormalize(indices, H, align_corners=False)

        if args.render_only:
            try:
                indices_ = indices_.reshape(indices_.shape[0], args.chunk, -1, indices_.shape[2])
            except:
                pdb.set_trace()
        else:
            indices_ = indices_.reshape(indices_.shape[0], args.N_rand, -1, indices_.shape[2])

        indices_ = torch.cat((indices_[:,:,:,1].unsqueeze(-1),indices_[:,:,:,0].unsqueeze(-1)),-1)
        #print("indices",indices.shape)4 57600 2
        features = nn.functional.grid_sample(images_features.permute(0,3,1,2), indices_, padding_mode='border', align_corners=False)
        #G_features_ = nn.functional.grid_sample(G_features.permute(0,3,1,2), indices_, padding_mode='border', align_corners=False)
        #print("features",features.shape)4 128 900 64
        features = features.reshape(args.num_reference_images, 128, -1).permute(0,2,1)
        #G_features_ = G_features_.reshape(1, 128, -1).permute(0,2,1)
        #print("features",features.shape)4 57600 128
        return torch.cat((features, indices.int().float()), -1), loss_translation
        #return torch.cat((features, indices.int().float()), -1)

    def network_query_fn(inputs, viewdirs, aud_para, network_fn, attention_poses, intrinsic, training, images_features1,G_feature,netchunk, lip_rect): \
        return run_network(inputs, viewdirs, aud_para, network_fn,
                           embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=netchunk, attention_poses = attention_poses,
                           intrinsic = intrinsic, training = training, images_features1 = images_features1,G_feature=G_feature, world_fn = world_fn, gather_func = gather_indices, lip_rect=lip_rect)

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in natsorted(
            os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    learned_codes_dict = None
    AudNet_state = None
    AudAttNet_state = None
    optimizer_aud_state = None
    optimizer_audatt_state = None
    optimizer_G_state = None
    optimizer_lip_state = None
    print("args.no_reload",args.no_reload)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        '''
        ckpts_pre=['/fs1/home/tjuvis_2022/lxx/DFRF/save/Obama_/pt/384000_head.tar']
        ckpt_path_pre = ckpts_pre[-1]
        print('pre Reloading from', ckpt_path_pre)
        ckpt_pre = torch.load(ckpt_path_pre)        
        
        load_ckpt(nerf_model_with_attention,ckpt_pre['network_fn_state_dict'])
        load_ckpt(model_obj,ckpt_pre['unet_state_dict'])
        load_ckpt(attention_module,ckpt_pre['attention_state_dict'])
        '''


      
        
        start = ckpt['global_step']
        #if args.render_only:
        #optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        

     

        # Load model
        nerf_model_with_attention.load_state_dict(ckpt['network_fn_state_dict'])
        model_obj.load_state_dict(ckpt['unet_state_dict'])
        #model_obj__.load_state_dict(ckpt['unet_state_dict__'])
        #img_encoder.load_state_dict(ckpt['img_encoder']) 
        
        model_obj2.load_state_dict(ckpt['unet_state_dict2'])  
        '''      

        ckpts_G=['/fs1/home/tjuvis_2022/lxx/NeRF-all/save/pt_Obama_/031000_head.tar']
        ckpt_path_G = ckpts_G[-1]
        print('G Reloading from', ckpt_path_G)
        ckpt_G = torch.load(ckpt_path_G)
        '''
        ckpts_lip=['/fs1/home/tjuvis_2022/lxx/NeRF-pre/save/out_pre/400000_head.tar']
        #ckpts_lip=['/fs1/home/tjuvis_2022/lxx/NeRF-pre/save/May/pt/400000_head.tar']
        ckpt_path_lip = ckpts_lip[-1]
        print('lip Reloading from', ckpt_path_lip)
        ckpt_lip = torch.load(ckpt_path_lip)
        lip_encoder_new.load_state_dict(ckpt_lip['lip_encoder'])
        G_model.load_state_dict(ckpt['lip_encoder'])
        #load_ckpt(G_model,ckpt_G['lip_encoder'])
        
        AudNet_state = ckpt_lip['network_audnet_state_dict']
        optimizer_aud_state = ckpt_lip['optimizer_aud_state_dict']
        
        #if args.render_only:
        #position_warp.load_state_dict(ckpt['position_warp_state_dict'])
        
        attention_module.load_state_dict(ckpt['attention_state_dict'])
        #attention_module2.load_state_dict(ckpt['attention_state_dict2'])
        #attention_module1.load_state_dict(ckpt['attention_state_dict1'])
        

        if nerf_model_fine is not None:
            print('Have reload the fine model parameters. ')
            nerf_model_fine_with_attention.load_state_dict(ckpt['network_fine_state_dict'])
            #load_ckpt(nerf_model_fine_with_attention,ckpt_pre['network_fine_state_dict'])
        
        if 'network_audattnet_state_dict' in ckpt_lip:
            AudAttNet_state = ckpt_lip['network_audattnet_state_dict']
        if 'optimizer_audatt_state_dict' in ckpt_lip:
            optimizer_audatt_state = ckpt_lip['optimizer_audatt_state_dict']
           
        if 'optimizer_G_state_dict' in ckpt:
            optimizer_G_state = ckpt['optimizer_G_state_dict']
        if 'optimizer_lip_state_dict' in ckpt:
            optimizer_lip_state = ckpt['optimizer_lip_state_dict']        
        
    models['optimizer'] = optimizer
    
    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': nerf_model_fine_with_attention,
        'N_samples': args.N_samples,
        'network_fn': nerf_model_with_attention,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'training': True,
        'feature_extractor': model_obj,
        #'feature_extractor2': model_obj__,
        'generator': model_obj2,
        'lip_encoder':G_model,
        'lip_encoder_new':lip_encoder_new,
        'img_encoder':img_encoder,

        'position_warp_model':position_warp,
        'attention_embed_fn': attention_embed_fn,
        'attention_embed_ln': attention_embed_ln,
        'rotation_embed_fn': rotation_embed_fn,
        'rotation_embed_ln': rotation_embed_ln,
        'use_render_pose': not args.no_render_pose
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['training'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, learned_codes_dict, \
        AudNet_state, optimizer_aud_state, AudAttNet_state, optimizer_audatt_state,optimizer_G_state, optimizer_lip_state,models


def raw2outputs(raw, z_vals, rays_d, bc_rgb, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-(act_fn(raw)+1e-6)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(
        dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    rgb = torch.cat((rgb[:, :-1, :], bc_rgb.unsqueeze(1)), dim=1)
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * \
        torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map, weights*z_vals


def render_rays(ray_batch,
                bc_rgb,
                aud_para,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                training=False,attention_poses=None,intrinsic=None,images_features1=None,G_feature=None,position_warp_model=None, lip_rect=None):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)
        t_rand[..., -1] = 1.0
        z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]

    raw, attention_cache, loss_translation = network_query_fn(pts, viewdirs, aud_para, network_fn, attention_poses, intrinsic, training, images_features1,G_feature, 900*64, lip_rect)
    #print("raw:",raw.shape)900 32 4
    rgb_map, disp_map, acc_map, weights, depth_map, depth_grid = raw2outputs(raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine

        raw, _, loss_translation = network_query_fn(pts, viewdirs, aud_para, run_fn, attention_poses, intrinsic, training, images_features1,G_feature, 900*64*3, lip_rect)
        rgb_map, disp_map, acc_map, weights, depth_map, depth_grid = raw2outputs(
            raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest=pytest)

    loss_translation = torch.mean(torch.mean(loss_translation, 1) * (1-torch.minimum(depth_grid.detach(), torch.Tensor([1])).reshape(-1)))
    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['last_weight'] = weights[..., -1]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")
    return ret, loss_translation


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,default='out-att/',
                        help='experiment name')
    parser.add_argument("--expname_finetune", type=str,default='pt_Obama_/pt_after_pre2/',
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='/fs1/home/tjuvis_2022/lxx/NeRF-all/save/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='/fs1/home/tjuvis_2022/lxx/data_util/dataset/',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=1024,
                        help='batch size (number of random rays per gradient step)')#1024
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=500,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_false',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    #parser.add_argument("--ft_path", type=str, default=None,
    #                    help='specific weights npy file to reload for coarse network')
    
    parser.add_argument("--ft_path", type=str, default='/fs1/home/tjuvis_2022/lxx/NeRF-all/save/pt_Obama_/show/143000_head.tar',
    #parser.add_argument("--ft_path", type=str, default='/fs1/home/tjuvis_2022/lxx/NeRF-all/save/pt_Obama_/new_pre3/080000_head.tar',
    #parser.add_argument("--ft_path", type=str, default='/fs1/home/tjuvis_2022/lxx/NeRF-all/save/May/a/080000_head.tar',
                        help='specific weights npy file to reload for coarse network')
    #parser.add_argument("--ft_path", type=str, default='/fs1/home/tjuvis_2022/lxx/NeRF-att/net_epoch_30_id_G_new.tar',
    #                    help='specific weights npy file to reload for coarse network')
    parser.add_argument("--N_iters", type=int, default=400000,
                        help='number of iterations')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_false',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='audface',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_false',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # face flags
    parser.add_argument("--with_test", type=int, default=0,
                        help='whether to use test set')
    parser.add_argument("--dim_aud", type=int, default=64,
                        help='dimension of audio features for NeRF')
    parser.add_argument("--sample_rate", type=float, default=0.95,
                        help="sample rate in a bounding box")
    parser.add_argument("--near", type=float, default=0.3,
                        help="near sampling plane")
    parser.add_argument("--far", type=float, default=0.9,
                        help="far sampling plane")
    parser.add_argument("--test_file", type=str, default='transforms_test.json',
                        help='test file')
    parser.add_argument("--aud_file", type=str, default='aud.npy',
                        help='test audio deepspeech file')
    parser.add_argument("--win_size", type=int, default=16,
                        help="windows size of audio feature")
    parser.add_argument("--smo_size", type=int, default=8,
                        help="window size for smoothing audio features")
    parser.add_argument('--nosmo_iters', type=int, default=50000,
                        help='number of iterations befor applying smoothing on audio features')#300000

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=1000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=150000,
                        help='frequency of testset saving')#10000
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')
    #add some paras
    parser.add_argument("--num_reference_images",   type=int, default=4,
                        help='the number of reference input images k')
    parser.add_argument("--select_nearest",   type=int, default=0,
                        help='whether to select the k-nearest images as the reference')
    parser.add_argument("--use_quaternion",   type=bool, default=False)
    parser.add_argument("--use_rotation_embed",   type=bool, default=False)
    parser.add_argument("--no_render_pose", type=bool, default=True)
    parser.add_argument("--use_warp", type=bool, default=True)
    parser.add_argument("--indices_before_iter", type=int, default=0)
    parser.add_argument("--translation_iter", type=int, default=0)
    parser.add_argument("--L2loss_weight", type=float, default=5e-9)
    parser.add_argument("--use_feature_map", type=bool, default=False)
    parser.add_argument("--selectimg_for_heatmap", type=int, default=0)
    parser.add_argument("--train_length", type=int, default=15)
    parser.add_argument("--need_torso", type=bool, default=True)
    parser.add_argument("--bc_type", type=str, default='torso_imgs')
    parser.add_argument("--refer_from_train", type=int, default=1)
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    print(args.near, args.far)
    # Load data

    if args.dataset_type == 'audface':
        if args.with_test == 1:
            head_images,face_images,images, poses, auds, bc_img, hwfcxy, lip_rects, torso_bcs, _, gt_aus1,pkl = \
                load_audface_data(args.datadir, args.testskip, args.test_file, args.aud_file, need_lip=True, need_torso = args.need_torso, bc_type=args.bc_type)
            #images = np.zeros(1)
        else:
            head_images,face_images,images, poses, auds, bc_img, hwfcxy, sample_rects, i_split, id_num, lip_rects, torso_bcs, gt_aus1,pkl = load_audface_data(
                args.datadir, args.testskip, train_length = args.train_length, need_lip=True, need_torso = args.need_torso, bc_type=args.bc_type)

        #print('Loaded audface', images['0'].shape, hwfcxy, args.datadir)
        if args.with_test == 0:
            print('Loaded audface', images['Obama_'].shape, hwfcxy, args.datadir)
            #all id has the same split, so this part can be shared
            i_train, i_val = i_split['Obama_']
        else:
            print('Loaded audface', len(images), hwfcxy, args.datadir)
        near = args.near
        far = args.far
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    #print(auds.shape)
    H, W, focal, cx, cy = hwfcxy
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    hwfcxy = [H, W, focal, cx, cy]

    intrinsic = np.array([[focal, 0., W / 2],
                          [0, focal, H / 2],
                          [0, 0, 1.]])
    intrinsic = torch.Tensor(intrinsic).to(device).float()

    # if args.render_test:
    #     render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    expname_finetune = args.expname_finetune
    os.makedirs(os.path.join(basedir, expname_finetune), exist_ok=True)
    f = os.path.join(basedir, expname_finetune, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname_finetune, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, learned_codes, \
    AudNet_state, optimizer_aud_state, AudAttNet_state, optimizer_audatt_state,optimizer_G_state,optimizer_lip_state, models = create_nerf(args)

    global_step = start

    AudNet = AudioNet(args.dim_aud, args.win_size).to(device)
    AudAttNet = AudioAttNet().to(device)
    optimizer_Aud = torch.optim.Adam(
        params=list(AudNet.parameters()), lr=args.lrate, betas=(0.9, 0.999))
    optimizer_AudAtt = torch.optim.Adam(
        params=list(AudAttNet.parameters()), lr=args.lrate, betas=(0.9, 0.999))

    optimizer_G = torch.optim.Adam(
        params=list(render_kwargs_train['lip_encoder'].parameters()), lr=args.lrate, betas=(0.9, 0.999))
    optimizer_lip = torch.optim.Adam(
        params=list(render_kwargs_train['lip_encoder_new'].parameters()), lr=args.lrate, betas=(0.9, 0.999))
    if AudNet_state is not None:
        AudNet.load_state_dict(AudNet_state, strict=False)
    if optimizer_aud_state is not None:
        optimizer_Aud.load_state_dict(optimizer_aud_state)
    if AudAttNet_state is not None:
        AudAttNet.load_state_dict(AudAttNet_state, strict=False)
    if optimizer_audatt_state is not None:
        optimizer_AudAtt.load_state_dict(optimizer_audatt_state)
    if optimizer_G_state is not None:
        optimizer_G.load_state_dict(optimizer_G_state)
    if optimizer_lip_state is not None:
        optimizer_lip.load_state_dict(optimizer_lip_state)
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            head_images_refer,face_images_refer,images_refer, poses_refer, auds_refer, bc_img_refer, _ , lip_rects_refer, _, _, gt_aus1,pkl = \
                load_audface_data(args.datadir, args.testskip, 'transforms_val.json', args.aud_file, need_lip=True, need_torso=False, bc_type=args.bc_type)

            images_refer = torch.cat([torch.Tensor(imageio.imread(images_refer[i])).cpu().unsqueeze(0) for i in
                                range(len(images_refer))], 0).float()/255.0

            poses_refer = torch.Tensor(poses_refer).float().cpu()
            # Default is smoother render_poses path
            #the data loader return these: images, poses, auds, bc_img, hwfcxy
            bc_img = torch.Tensor(bc_img).to(device).float() / 255.0
            poses = torch.Tensor(poses).to(device).float()
            auds = torch.Tensor(auds).to(device).float()
            testsavedir = os.path.join(basedir, expname_finetune, 'renderonly_{}_{:06d}'.format(
                'test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)

            print('test poses shape', poses.shape)
            #select reference images for the test set
            if args.refer_from_train:#1
                perm = [50,100,150,200]
                #perm1 = [50 100 150 200]
                perm = perm[0:args.num_reference_images]
                attention_images = images_refer[perm].to(device)

                attention_images_face_ = np.array(face_images_refer)[perm]
                attention_images_face_ = torch.cat([torch.as_tensor(imageio.imread(attention_images_face_[i])).unsqueeze(0) for i in range(args.num_reference_images)],0)
                attention_images_face_ = attention_images_face_.float()/255.0                   
                
                transform_list = [#transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
                _transform = transforms.Compose(transform_list)
                
                attention_images1 = np.array(head_images_refer)[perm]
                attention_pkl = np.array(pkl)[perm]
                attention_images_face= np.array(face_images_refer)[perm]
                b=[]
                b1=[]

                for t in range(args.num_reference_images):
                    a=cv_utils.read_cv2_img(attention_images1[t])
                    a1=cv_utils.read_cv2_img(attention_images_face[t])
                    a = _transform(Image.fromarray(a))
                    a1 = _transform(Image.fromarray(a1))
                    b.append(a)
                    b1.append(a1)
                b=torch.stack(b)
                b=b.cuda()
                b1=torch.stack(b1)
                b1=b1.cuda()

                
                #attention_images = attention_images.float()/255.0
                
                gt_aus = torch.as_tensor(np.array(gt_aus1)[perm]).cuda()
                gt_aus= gt_aus.type(torch.cuda.FloatTensor)
                attention_poses = poses_refer[perm, :3, :4].to(device)
            else:
                perm = np.random.randint(images_refer.shape[0]-1, size=4).tolist()
                attention_images_ = np.array(images)[perm]
                attention_images = torch.cat([torch.Tensor(imageio.imread(i)).unsqueeze(0) for i in
                                          attention_images_], 0).float() / 255.0
                attention_poses = poses[perm, :3, :4].to(device)

            auds_val = []
            if start < args.nosmo_iters:
                auds_val = AudNet(auds)
            else:
                print('Load the smooth audio for rendering!')
                for i in range(poses.shape[0]):
                    smo_half_win = int(args.smo_size / 2)
                    left_i = i - smo_half_win
                    right_i = i + smo_half_win
                    pad_left, pad_right = 0, 0
                    if left_i < 0:
                        pad_left = -left_i
                        left_i = 0
                    if right_i > poses.shape[0]:
                        pad_right = right_i - poses.shape[0]
                        right_i = poses.shape[0]
                    auds_win = auds[left_i:right_i]
                    if pad_left > 0:
                        auds_win = torch.cat(
                            (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
                    if pad_right > 0:
                        auds_win = torch.cat(
                            (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
                    auds_win = AudNet(auds_win)
                    #aud = auds_win[smo_half_win]
                    aud_smo = AudAttNet(auds_win)
                    auds_val.append(aud_smo)
                auds_val = torch.stack(auds_val, 0)

            #print(b.shape)
            with torch.no_grad():
                rgbs, disp, last_weight = render_path(attention_pkl,pkl,gt_aus1,face_images_refer,b1,attention_images_face_,args,  images_refer,torso_bcs, poses, auds_val, bc_img, hwfcxy, attention_poses,attention_images,b,gt_aus,
                            intrinsic, args.chunk, render_kwargs_test, gt_imgs=None, savedir=testsavedir, lip_rect=lip_rects_refer[perm])

            np.save(os.path.join(testsavedir, 'last_weight.npy'), last_weight)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(
                testsavedir, 'video.mp4'), to8b(rgbs), fps=25, quality=8)
            return


    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    print('N_rand', N_rand, 'no_batching',
          args.no_batching, 'sample_rate', args.sample_rate)
    use_batching = not args.no_batching

    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p, cx, cy)
                         for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], 0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = args.N_iters + 1
    print('Begin')
    print('TEST views are', i_train)
    print('TRAIN views are', i_val)

    start = start + 1
    psnr_total=0
    ssim_total=0
    for i in trange(0, len(i_train)):
        time0 = time.time()
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            #1.Select a id for training
            select_id = np.random.choice(id_num)
            select_id=str(select_id)
            select_id="Obama_"
            #bc_img_ = torch.Tensor(bc_img[str(select_id)]).to(device).float()/255.0
            poses_ = torch.Tensor(poses[str(select_id)]).to(device).float()
            auds_ = torch.Tensor(auds[str(select_id)]).to(device).float()
            i_train, i_val = i_split[str(select_id)]
            img_i = i#np.random.choice(i_train)
            image_path = images[str(select_id)][img_i]
            i_pkl=pkl[str(select_id)][img_i]
            face_real_path=face_images[str(select_id)][img_i]
            iden_path=face_images[str(select_id)][8]






            bc_img_ = torch.as_tensor(imageio.imread(torso_bcs[str(select_id)][img_i])).to(device).float()/255.0
            #bc_img_ = torch.as_tensor(imageio.imread(torso_bcs[str(select_id)][6561])).to(device).float()/255.0
            #bc_img_ = torch.as_tensor(imageio.imread(os.path.join(basedir, 'bc.jpg'))).to(device).float()/255.0

            target = torch.as_tensor(imageio.imread(image_path)).to(device).float()/255.0
            #print(face_real_path)
            #iden=torch.as_tensor(align_face(imageio.imread(iden_path))).to(device).float()/255.0
            #face_real = torch.as_tensor(align_face(imageio.imread(face_real_path))).to(device).float()/255.0
            iden=torch.as_tensor(imageio.imread(iden_path)).to(device).float()/255.0
            face_real = torch.as_tensor(imageio.imread(face_real_path)).to(device).float()/255.0
            pose = poses_[img_i, :3, :4]
            rect = sample_rects[str(select_id)][img_i]
            aud = auds_[img_i]

            #select the attention pose and image
            if args.select_nearest:
                current_poses = poses[str(select_id)][:, :3, :4]
                current_images = images[str(select_id)]  # top size was set at 4 for reflective ones
                current_images = torch.cat([torch.as_tensor(imageio.imread(current_images[i])).unsqueeze(0) for i in range(current_images.shape[0])], 0)
                current_images = current_images.float() / 255.0
                attention_poses, attention_images = get_similar_k(pose, current_poses, current_images, top_size=None, k = 20)
            else:
            
                #i_train_left = np.delete(i_train, np.where(np.array(i_train) == img_i))
                #perm = np.random.permutation(i_train_left)[:args.num_reference_images]#selete num_reference_images images from the training set as reference
                perm = [50,100,150,200]
                #perm = [10,20,30,60]
                attention_images = images[str(select_id)][perm]
                attention_images = torch.cat([torch.as_tensor(imageio.imread(attention_images[i])).unsqueeze(0) for i in range(args.num_reference_images)],0)
                attention_images = attention_images.float()/255.0

                attention_images_face_ = face_images[str(select_id)][perm]
                attention_images_face_ = torch.cat([torch.as_tensor(imageio.imread(attention_images_face_[i])).unsqueeze(0) for i in range(args.num_reference_images)],0)
                attention_images_face_ = attention_images_face_.float()/255.0   
            
                #i_train_left = np.delete(i_train, np.where(np.array(i_train) == img_i))
                #perm = np.random.permutation(i_train_left)[:args.num_reference_images]#selete num_reference_images images from the training set as reference
                transform_list = [#transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
                _transform = transforms.Compose(transform_list)
                
                attention_images1 = head_images[str(select_id)][perm]
                attention_pkl=pkl[str(select_id)][perm]
                attention_images_face = face_images[str(select_id)][perm]
                b=[]
                b1=[]
                face_real_=cv_utils.read_cv2_img(face_real_path)
                face_real_ = _transform(Image.fromarray(face_real_)).cuda()
                iden_=cv_utils.read_cv2_img(iden_path)
                iden_ = _transform(Image.fromarray(iden_)).cuda()
                for t in range(args.num_reference_images):
                    a=cv_utils.read_cv2_img(attention_images1[t])
                    a1=cv_utils.read_cv2_img(attention_images_face[t])
                    a = _transform(Image.fromarray(a))
                    a1 = _transform(Image.fromarray(a1))
                    b.append(a)
                    b1.append(a1)
                b=torch.stack(b)
                b=b.cuda()
                b1=torch.stack(b1)
                b1=b1.cuda() 
                
                #attention_images = attention_images.float()/255.0
                
                gt_aus = torch.as_tensor(gt_aus1[str(select_id)][perm]).cuda()
                gt_aus= gt_aus.type(torch.cuda.FloatTensor)
                G_gt_aus = torch.as_tensor(gt_aus1[str(select_id)][img_i]).cuda()
                G_gt_aus= G_gt_aus.type(torch.cuda.FloatTensor)
                iden_gt_aus = torch.as_tensor(gt_aus1[str(select_id)][8]).cuda()
                iden_gt_aus= iden_gt_aus.type(torch.cuda.FloatTensor)
                #print("gt_aus: ",gt_aus.shape)
                #gt_aus = gt_aus.contiguous().view(gt_aus.shape[0]*gt_aus.shape[1],-1)
                #gt_aus=gt_aus.contiguous().view(1, gt_aus.shape[0])
                
                attention_poses = poses[str(select_id)][perm, :3, :4]
                lip_rect = torch.Tensor(lip_rects[str(select_id)][perm])
                head_rect = torch.Tensor(sample_rects[str(select_id)][perm])

            attention_poses = torch.Tensor(attention_poses).to(device).float()
            if global_step >= args.nosmo_iters:
                smo_half_win = int(args.smo_size / 2)
                left_i = img_i - smo_half_win
                right_i = img_i + smo_half_win
                pad_left, pad_right = 0, 0
                if left_i < 0:
                    pad_left = -left_i
                    left_i = 0
                if right_i > i_train.shape[0]:
                    pad_right = right_i-i_train.shape[0]
                    right_i = i_train.shape[0]
                auds_win = auds_[left_i:right_i]
                if pad_left > 0:
                    auds_win = torch.cat(
                        (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
                if pad_right > 0:
                    auds_win = torch.cat(
                        (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
                auds_win = AudNet(auds_win)
                aud = auds_win[smo_half_win]
                aud_smo = AudAttNet(auds_win)
            else:
                aud = AudNet(aud.unsqueeze(0))
            if N_rand is not None:
                rays_o, rays_d = get_rays(
                    H, W, focal, torch.Tensor(pose), cx, cy)  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(
                        0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                if args.sample_rate > 0:
                    rect_inds = (coords[:, 0] >= rect[0]) & (
                        coords[:, 0] <= rect[0] + rect[2]) & (
                            coords[:, 1] >= rect[1]) & (
                                coords[:, 1] <= rect[1] + rect[3])
                    coords_rect = coords[rect_inds]
                    coords_norect = coords[~rect_inds]
                    rect_num = int(N_rand*args.sample_rate)
                    norect_num = N_rand - rect_num
                    select_inds_rect = np.random.choice(
                        coords_rect.shape[0], size=[rect_num], replace=False)  # (N_rand,)
                    # (N_rand, 2)
                    select_coords_rect = coords_rect[select_inds_rect].long()
                    select_inds_norect = np.random.choice(
                        coords_norect.shape[0], size=[norect_num], replace=False)  # (N_rand,)
                    # (N_rand, 2)
                    select_coords_norect = coords_norect[select_inds_norect].long(
                    )
                    select_coords = torch.cat(
                        (select_coords_rect, select_coords_norect), dim=0)
                else:
                    select_inds = np.random.choice(
                        coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()

                #print("select:",select_coords[:, 0].shape,select_coords[:, 1].shape,rays_o.shape) 900 900 450 450 3
                rays_o = rays_o[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0],
                                  select_coords[:, 1]]  # (N_rand, 3)
                #print(bc_img_.shape)
                bc_rgb = bc_img_[select_coords[:, 0],
                                select_coords[:, 1]]


        
        #####  Core optimization loop  #####
        if global_step >= args.nosmo_iters:
            auds_val=aud_smo
        else:
            auds_val=aud
                        
        with torch.no_grad():
            rgb_ = render_train(img_i,iden,iden_,target,attention_pkl,i_pkl,iden_gt_aus,G_gt_aus,face_real,face_real_,b1,attention_images_face_,args,  torso_bcs[str(select_id)], pose, auds_val, bc_img, hwfcxy, attention_poses,attention_images,b,gt_aus,
                            intrinsic, args.chunk, render_kwargs_test, gt_imgs=None,lip_rect=lip_rects[str(select_id)][perm])
        
        ssim = structural_similarity(torch.as_tensor(rgb_).cpu().detach().numpy()*255.0, torch.as_tensor(target).cpu().detach().numpy()*255.0,multichannel=True)
        
        img_loss = img2mse(torch.as_tensor(rgb_).cuda(), torch.as_tensor(target).cuda())
        psnr = mse2psnr(img_loss)
        
        psnr_total=psnr_total+psnr
        ssim_total=ssim_total+ssim
        if i % 1 == 0:
            rgb8 = to8b(rgb_.cpu().numpy())
            filename = os.path.join('/fs1/home/tjuvis_2022/lxx/NeRF-all/rebuttal/Obama/test/', '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            rgb8 = to8b(target.cpu().numpy())
            filename = os.path.join('/fs1/home/tjuvis_2022/lxx/NeRF-all/rebuttal/Obama/input/', '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)        
            tqdm.write(f"[TEST] Iter: {i} PSNR: {psnr.item()} SSIM: {ssim.item()}")
        if i == 300 :
            break
    print("avg_psnr: ",(psnr_total / 301))
    print("avg_ssim: ",(ssim_total / 301))


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()