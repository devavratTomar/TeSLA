
import os, re
import copy
import random
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as tvF
import torchvision

import numpy as np
import pandas as pd
import json

from .distributed import init_distributed_mode


COLORS = 2*torch.tensor([[128,64,128],
                         [244,35,232],
                         [70,70,70],
                         [102,102,156],
                         [190,153,153],
                         [153,153,153],
                         [250,170,30],
                         [220,220,0],
                         [107,142,35],
                         [152,251,152],
                         [70,130,180],
                         [220,20,60],
                         [255,0,0],
                         [0,0,142],
                         [0,0,70],
                         [0,60,100],
                         [0,80,100],
                         [0,0,230],
                         [119,11,32],
                         [0,0,0]], dtype=float)/255.0 - 1.0


def normalize_fn(mean, std):
    return lambda x : tvF.normalize(x, mean, std)

def denormalize_fn(mean, std):
    return lambda x : tvF.normalize(x, -torch.tensor(mean)/torch.tensor(std), 1/torch.tensor(std))

def covariance(features):
    assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / (n - 1)
    return cov


def coral(cs, ct):
    d = cs.shape[0]
    loss = (cs - ct).pow(2).sum() / (4. * d ** 2)
    return loss


def linear_mmd(ms, mt):
    loss = (ms - mt).pow(2).mean()
    return loss

def clip_by_norm(feats):
    with torch.no_grad():
        norm_feats = torch.norm(feats, p=2, dim=-1, keepdim=True)
        norm_feats = torch.clip(norm_feats, min=1.0)

    feats = feats / norm_feats
    return feats

def clip_gradient(optimizer):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, 1.0, norm_type=2.0, error_if_nonfinite=True)

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]

def natural_sort(items):
    new_items = items.copy()
    new_items.sort(key=natural_keys)
    return new_items

def getcolorsegs(seg):
    seg = seg.detach().cpu()
    seg[seg==255] = 19
    color_seg = COLORS[seg].permute([2, 0, 1]) if seg.dim()==2 else COLORS[seg].permute([0, 3, 1, 2])
    color_seg = color_seg.clamp(-1, 1) * 0.5 + 0.5
    return color_seg

def overlay_segs(img, seg, alpha=0.2):
    """
    imgs should be in range [-1, 1] and shape C x H x W or B x C x H x W.
    seg should have integer range with same spatial shape as H x W or B x H x W.
    """
    img = img.detach().cpu()
    seg = seg.detach().cpu()
    seg[seg==255] = 19

    assert img.size()[-2:] == seg.size()[-2:]
    if img.size()[-3] == 1:
        img = img.repeat([3, 1, 1]) if img.dim() == 3 else img.repeat([1, 3, 1, 1])

    mask = (seg != 0)

    if seg.dim() == 3:
        mask = mask.unsqueeze(1)

    color_seg = COLORS[seg].permute([2, 0, 1]) if seg.dim()==2 else COLORS[seg].permute([0, 3, 1, 2])

    ### colors from 0 and 1
    color_seg = color_seg.clamp(-1, 1) * 0.5 + 0.5
    img = img.clamp(-1, 1) * 0.5 + 0.5
    
    merged = mask*(alpha*color_seg + (1-alpha)*img) + (~mask) * img

    return merged


def setup_experiment(opt):
    # Setup SEED
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)

    random.seed(opt.seed)
    torchvision.torch.manual_seed(opt.seed)
    torchvision.torch.cuda.manual_seed(opt.seed)

    # Create directory
    ensure_dir(opt.experiment_dir)

    # Init distribution
    opt.dist=False
    if torch.cuda.is_available():
        init_distributed_mode(opt)
        opt.dist = opt.world_size>1
        opt.batch_size = opt.batch_size//opt.world_size
        cudnn.benchmark = True
    else:
        raise NotImplementedError("This script needs GPU. No GPU available or torch.cuda.is_available is False !")


def load_results(roots, with_feats=False):
    res_dict = {}
    for root in roots:
        method = "/".join(root.split("/")[-4:-2])
        metadata = pd.read_csv(os.path.join(root, "metadata.csv"))
        if with_feats:
            features_w = torch.load(os.path.join(root, "Feat_w.pth"))
            if os.path.exists(os.path.join(root, "Feat_h.pth")):
                features_h = torch.load(os.path.join(root, "Feat_h.pth"))
            else:
                features_h = copy.deepcopy(features_w)
            res_dict[method] = {"metadata" : metadata, "features_w" : features_w, "features_h" : features_h}
        else:
            res_dict[method] = {"metadata" : metadata}
    return res_dict

def compare_results(res_dict, pred_field="Pred"):
    df = pd.DataFrame()
    for method, data in res_dict.items():
        metadata = data["metadata"]
        if pred_field in metadata:
            metadata["Correct"] = metadata["Label"] == metadata[pred_field]
        else:
            metadata["Correct"] = metadata["Label"] == metadata["Pred"]
        perclass_acc = metadata[["Label", "Correct"]].groupby(["Label"]).mean().reset_index()
        perclass_acc.loc[len(perclass_acc)] = {"Label":"Avg.", "Correct":perclass_acc["Correct"].mean()}
        perclass_acc.loc[len(perclass_acc)] = {"Label":"Acc.", "Correct": metadata["Correct"].mean()}
        df[method] = perclass_acc[["Label", "Correct"]].groupby(["Label"]).mean()
    return df

def compare_results_with_stds(roots, seeds, pred_field="Pred"):
    dfs = []
    for seed in seeds:
        new_roots = []
        for root in roots:
            new_root = root.split("/")
            new_root.insert(-1, seed)
            new_roots.append("/".join(new_root))

        res_dict = load_results(new_roots)
        df = compare_results(res_dict, pred_field=pred_field)
        dfs.append(df)
    group_df = (
        # combine dataframes into a single dataframe
        pd.concat(dfs)
        # replace 0 values with nan to exclude them from mean calculation
        .groupby("Label")
    )
    return group_df.mean(), group_df.std()


def load_segmentation_source_model(net, path):
    weights = torch.load(path, map_location='cpu')
    net.load_state_dict(weights)


def json_load(file_path):
    with open(file_path, 'r') as fp:
        return json.load(fp)


def offline(trloader, net, class_num=10):
    net.eval()
    feat_stack = [[] for i in range(class_num)]

    with torch.no_grad():
        for batch_idx, (inputs, labels, index) in enumerate(trloader):
            predict_logit, feat = net(inputs.cuda(), True)
            pseudo_label = predict_logit.max(dim=1)[1]

            for label in pseudo_label.unique():
                label_mask = pseudo_label == label
                feat_stack[label].extend(feat[label_mask, :])
    ext_mu = []
    ext_cov = []
    ext_all = []

    for feat in feat_stack:
        ext_mu.append(torch.stack(feat).mean(dim=0))
        ext_cov.append(covariance(torch.stack(feat)))
        ext_all.extend(feat)

    ext_all = torch.stack(ext_all)
    ext_all_mu = ext_all.mean(dim=0)
    ext_all_cov = covariance(ext_all)
    net.train()

    return ext_mu, ext_cov, ext_all_mu, ext_all_cov
