import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from importlib import import_module

from sam_fact_tt_image_encoder import Fact_tt_Sam
from segment_anything import sam_model_registry

from trainer import trainer_run
from icecream import ic

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/mnt/weka/wekafs/rad-megtron/cchen/synapseCT/Training/2D_all_5slice', help='root dir for data')
parser.add_argument('--output', type=str, default='/mnt/weka/wekafs/rad-megtron/cchen/project_results/MA_SAM/results-1')
parser.add_argument('--num_classes', type=int, default=12, help='output channel of network')
parser.add_argument('--batch_size', type=int, default=3, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=8, help='total gpu')
parser.add_argument('--base_lr', type=float, default=0.0008, help='segmentation network learning rate')

parser.add_argument('--max_epochs', type=int,default=400, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int, default=300, help='maximum epoch number to train')

parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_name', type=str, default='vit_h', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='/mnt/weka/wekafs/rad-megtron/cchen/PretrainedModel/sam_vit_h_4b8939.pth', help='Pretrained checkpoint')
parser.add_argument('--adapt_ckpt', type=str, default=None, help='Finetuned checkpoint')
parser.add_argument('--rank', type=int, default=32, help='Rank for FacT')
parser.add_argument('--scale', type=float, default=1.0, help='Scale for FacT')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid when warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='sam_fact_tt_image_encoder')
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--lr_exp', type=float, default=7, help='The learning rate decay expotential')

# acceleration choices
parser.add_argument('--tf32', action='store_true', help='If activated, use tf32 to accelerate the training process')
parser.add_argument('--compile', action='store_true', help='If activated, compile the training model for acceleration')
parser.add_argument('--use_amp', action='store_true', help='If activated, adopt mixed precision for acceleration')
parser.add_argument('--skip_hard', action='store_true', help='If activated, adopt mixed precision for acceleration')

args = parser.parse_args()
args.warmup = True
args.AdamW = True
args.tf32 = True
args.compile = False
args.use_amp = True
args.skip_hard = True

if __name__ == "__main__":
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0., 0., 0.],
                                                                pixel_std=[1., 1., 1.])

    pkg = import_module(args.module)
    net = pkg.Fact_tt_Sam(sam, args.rank, s=args.scale).cuda()
    if args.compile:
        net = torch.compile(net)

    if args.adapt_ckpt is not None:
        net.load_parameters(args.adapt_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4

    config_file = os.path.join(args.output, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer_run(args, net, args.output, multimask_output, low_res)
