# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import os
import torch
import time
# import timm
# import torchvision
# import utils\
from config import cfg
from model_test import get_model




torch.autograd.set_grad_enabled(False)


T0 = 10
T1 = 60

def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.Conv2d):
            child.bias = torch.nn.Parameter(torch.zeros(child.weight.size(0)))
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)

def compute_throughput_cpu(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    # warmup
    start = time.time()
    while time.time() - start < T0:
        model(inputs)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)


def compute_throughput_cuda(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast():
        while time.time() - start < T0:
            model(inputs)
    timing = []
    if device == 'cuda:0':
        torch.cuda.synchronize()
    with torch.cuda.amp.autocast():
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)


for device in ['cuda:0', 'cpu']:
# for device in ['cuda:0']:
    if 'cuda' in device and not torch.cuda.is_available():
        print("no cuda")
        continue

    if device == 'cpu':
        os.system('echo -n "nb processors "; '
                  'cat /proc/cpuinfo | grep ^processor | wc -l; '
                  'cat /proc/cpuinfo | grep ^"model name" | tail -1')
        print('Using 1 cpu thread')
        torch.set_num_threads(1)
        compute_throughput = compute_throughput_cpu
    else:
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        compute_throughput = compute_throughput_cuda

    # for n, batch_size0, resolution in [      
    #     ('timm.models.resnet50', 1024, 224),
    #     ('timm.models.vit_deit_tiny_distilled_patch16_224', 2048, 224),
    #     ('timm.models.vit_deit_small_distilled_patch16_224', 2048, 224),
    #     ('levit.LeViT_128S', 2048, 224),
    #     ('levit.LeViT_128', 2048, 224),
    #     ('levit.LeViT_192', 2048, 224),
    #     ('levit.LeViT_256', 2048, 224),
    #     ('levit.LeViT_384', 1024, 224),
    #     ('timm.models.efficientnet_b0', 1024, 224),
    #     ('timm.models.efficientnet_b1', 1024, 240),
    #     ('timm.models.efficientnet_b2', 512, 260),
    #     ('timm.models.efficientnet_b3', 512, 300),
    #     ('timm.models.efficientnet_b4', 256, 380),
    # ]:
    #     pass
    resolution = 256
    if device == 'cpu':
        batch_size = 1
    else:
        batch_size = 16
        torch.cuda.empty_cache()
    inputs = torch.randn(batch_size, 3, resolution,
                            resolution, device=device)
    # inputs = {'img':inputs}
    # model = eval(n)(num_classes=1000)
    model = get_model(6890, 29, 'test', device)

    replace_batchnorm(model)
    model.to(device)
    model.eval()
    model = torch.jit.trace(model, inputs)
    compute_throughput('model', model, device,
                        batch_size, resolution=resolution)