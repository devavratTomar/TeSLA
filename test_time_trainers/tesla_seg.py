from copy import deepcopy
import math
import random
import torch
import torch.utils.data as data
import torchvision.utils as tvu
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import torch.nn.functional as F
from medpy.metric.binary import dc
import torchmetrics
from loguru import logger

from tqdm import tqdm

import dataloaders.segmentation.mr_segmentation as mr_seg_dataset
import dataloaders.segmentation.visda.cityscapes as rgb_seg_dataset
from networks.segmentation.unet import UNet
from networks.segmentation.deeplabv3 import get_deeplab_v3, DeepLabEncapsulator
from networks.augmentation.segoptaug import SegOptAug, RefinedPseudoLabels

from losses.seg_losses import EntropyClassMarginals, EntropyLoss

from utilities.utils import ensure_dir, normalize_fn, denormalize_fn, load_segmentation_source_model, overlay_segs, clip_gradient, getcolorsegs
from utilities.metric_tracker import MetricTracker
from utilities.distributed import concat_all_gather, sync_batchnorms, model_to_DDP, DistributedEvalSampler
from networks.ema_seg import ModelEMA
from utilities.metrics import *


class PolicyReplay:
    def __init__(self, buffer_size=64) -> None:
        self.buffer_size = buffer_size
        self.policies = []

    def sample(self, policy_module):
        current_params = {k: v.cpu() for k, v in policy_module.state_dict().items()}

        # add current policy params to the buffer
        if len(self.policies) < self.buffer_size:
            self.policies.append(current_params)
            return_params = current_params
        else:
            if random.random() > 0.5:
                return_params = current_params
            else:
                # randomly replace the past policy from the buffer
                idx = random.randint(0, self.buffer_size-1)
                return_params = self.policies[idx]
                self.policies[idx] = current_params

        # put return_params on the gpu
        return_params = {k: v.cuda() for k, v in return_params.items()}
        policy_module.load_state_dict(return_params)


class TeSLA_Seg(object):
    def __init__(self, opt):

        # Setup attributes
        self.opt = opt
        
        # policy replay
        self.policy_replay = PolicyReplay()
        
        # source model
        if opt.dataset_type == "med":
            self.net = UNet(self.opt.n_channels, self.opt.n_classes)
        else:
            self.net = get_deeplab_v3(self.opt.n_classes)
        
        self.orig_state_dict = deepcopy(self.net.state_dict())
        # load source model
        load_segmentation_source_model(self.net, opt.pretrained_source_path)
        

        # freeze last layer of the model
        self.set_parameters(self.net)

        # encapsulate
        if opt.dataset_type != "med":
            self.net = DeepLabEncapsulator(self.net)
        
        # put in train mode and setup gpu
        self.net = self.net.cuda()
        self.net.train()

        # ema model for knowledge distillation
        self.ema_net = ModelEMA(self.net, decay=opt.ema_momentum).cuda()
        self.ema_net.train()
        # self.track_running_stats(self.ema_net, False)

        self.seg_refiner = RefinedPseudoLabels()

        if opt.dist:
            # Setup Dist
            self.net = sync_batchnorms(self.net)
            self.ema_net = sync_batchnorms(self.ema_net)
            self.net = model_to_DDP(self.net, opt.gpu)
        
        # initialize data loaders
        if opt.dataset_type == "med":
            if self.opt.n_epochs == 1:
                self.target_dataset = mr_seg_dataset.TestTimeVolumeDataset(self.opt.target_data_path, self.opt.target_sites, self.opt.dataset_name)
                self.target_eval_dataset = mr_seg_dataset.TestTimeDataset(self.opt.target_data_path, self.opt.target_sites, self.opt.dataset_name)
            else:
                self.target_dataset = mr_seg_dataset.TestTimeDataset(self.opt.target_data_path, self.opt.target_sites, self.opt.dataset_name)
                self.target_eval_dataset = mr_seg_dataset.TestTimeDataset(self.opt.target_data_path, self.opt.target_sites, self.opt.dataset_name)
        
        elif opt.dataset_type == "rgb":
            self.target_dataset = rgb_seg_dataset.CityscapesDataSet(root=self.opt.target_data_path,
                                                                    list_path=self.opt.target_list_path,
                                                                    set="train",
                                                                    info_path=self.opt.target_info_path,
                                                                    crop_size=opt.target_input_size,
                                                                    mean=(0.485, 0.456, 0.406),
                                                                    std=(0.229, 0.224, 0.225))

            self.target_eval_dataset = rgb_seg_dataset.CityscapesDataSet(root=self.opt.target_data_path,
                                                                         list_path=self.opt.target_eval_list_path,
                                                                         set="val",
                                                                         info_path=self.opt.target_info_path,
                                                                         crop_size=opt.target_input_size,
                                                                         mean=(0.485, 0.456, 0.406),
                                                                         std=(0.229, 0.224, 0.225))
        
        
        sampler_eval= DistributedEvalSampler(self.target_eval_dataset, shuffle=False) if self.opt.dist else None
        
        self.target_eval_loader = data.DataLoader(self.target_eval_dataset, batch_size=self.opt.batch_size, sampler=sampler_eval,
                                                    shuffle=False, drop_last=False, num_workers=2*self.opt.num_workers)

        shuffle = False if self.opt.n_epochs == 1 else True
        sampler = DistributedSampler(self.target_dataset, shuffle=shuffle) if self.opt.dist else None
        self.target_loader = data.DataLoader(self.target_dataset, batch_size=self.opt.batch_size, sampler=sampler,
                                             shuffle=False, drop_last=False, num_workers=2*self.opt.num_workers)
        
        # dictionary for nearest neighbours
        self.feats_nn_queue = {k:torch.Tensor().cuda() for k in range(self.opt.n_classes)} # [nclasses] 256 x feat_dim
        self.prob_nn_queue  = {k:torch.Tensor().cuda() for k in range(self.opt.n_classes)} # [nclasses] 256 x n_classes


        # Set augmentation module
        if opt.dataset_type == "med":
            self.normalize_fn = normalize_fn(self.target_dataset.mean, self.target_dataset.std)
            self.denormalize_fn = denormalize_fn(self.target_dataset.mean, self.target_dataset.std)
        else:
            self.normalize_fn = normalize_fn((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            self.denormalize_fn = denormalize_fn((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        self.hard_opt_aug = SegOptAug(opt.dataset_type, self.ema_net, opt.sub_policy_dim, opt.aug_mult, "Hard", self.normalize_fn, self.denormalize_fn, self.opt.lambda_1).cuda()
        
        if opt.dist:
            self.hard_opt_aug = model_to_DDP(self.hard_opt_aug, opt.gpu).module

        # losses
        self.crterian_cm = EntropyClassMarginals()

        # optimizer
        self.optimizer_net = torch.optim.AdamW(self.net.parameters(), lr=self.opt.lr, weight_decay=5e-4)
        # self.optimizer_net = torch.optim.SGD(self.net.parameters(), lr=self.opt.lr, momentum=0.9, weight_decay=5e-4)
        # self.optimizer_net = torch.optim.RMSprop(self.net.parameters(), lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)

        # track metrics
        self.metric_tracker = MetricTracker()

        if opt.dataset_type == "med":
            self.metric_seg = torchmetrics.Dice(num_classes=self.opt.n_classes, average=None, ignore_index=0).cuda()
            self.metric_seg_online = torchmetrics.Dice(num_classes=self.opt.n_classes, average=None, ignore_index=0).cuda()
        else:
            self.metric_seg = torchmetrics.JaccardIndex(num_classes=self.opt.n_classes + 1, average="none", ignore_index=self.opt.n_classes).cuda()
            self.metric_seg_online = torchmetrics.JaccardIndex(num_classes=self.opt.n_classes + 1, average="none", ignore_index=self.opt.n_classes).cuda()

        if opt.dist:
            self.metric_seg = model_to_DDP(self.metric_seg, opt.gpu)

        ## logging
        logger.add(os.path.join(self.opt.experiment_dir, f"training_logs_rank{self.opt.rank}.log"))

        self.train = self.train_volume if (self.opt.dataset_type == "med" and self.opt.n_epochs == 1) else self.train_image


        if self.opt.n_epochs != 1:
            self.update_bn_stats(self.net, self.target_loader)
            self.update_bn_stats(self.ema_net, self.target_loader)

        # self.ema_net.eval()

    @torch.no_grad()
    def update_bn_stats(self, net, data_loader):
        logger.info("Updating BN stats")
        for imgs, segs, _, names in tqdm(data_loader):
            imgs = imgs.cuda()
            net(imgs)

    
    def save_model(self, epoch):
        ensure_dir(os.path.join(self.opt.experiment_dir, "saved_models"))
        state_dict = self.ema_net.ema.state_dict()
        torch.save(state_dict, os.path.join(self.opt.experiment_dir, "saved_models", f"deeplab_epoch_{epoch}_lr_{self.opt.lr}"))


    def track_running_stats(self, net, v):
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.SyncBatchNorm):
                m.track_running_stats = v

    
    def set_parameters(self, net):
        try:
            backbone_params, classifier_params = net.get_backbone_classifier_params()
        except:
            backbone_params = list(net.backbone.parameters())
            classifier_params = list(net.classifier.parameters())
        
        for param in backbone_params:
            param.requires_grad=True
        
        for param in classifier_params:
            param.requires_grad=False

        # make bachnorm params fixed
        for m in net.modules():
            if (isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.SyncBatchNorm)):
                m.requires_grad_(False)
    
    
    @torch.no_grad()
    def evaluate_batch(self, net, batch, label, names=None):
        is_training = net.training
        preds = torch.zeros_like(label)
        preds_probs = torch.zeros(batch.size(0), self.opt.n_classes, batch.size(2), batch.size(3), dtype=torch.float32)
        if is_training:
            net.eval()
            if self.opt.dataset_type != "med":
                label[label==255] = 19
            
            for i, item in enumerate(batch):
                pred_prob = self.seg_refiner.get_pseudo_label(net, item.unsqueeze(0).cuda(), self.weak_mult)[0].cpu()
                preds_probs[i] = pred_prob
                preds[i] = torch.argmax(pred_prob, dim=0)
            
            self.metric_seg_online.update(preds.cuda(), label.cuda())

            # save if names are given
            if names:
                ensure_dir(os.path.join(self.opt.experiment_dir, "Final", "predictions"))
                ensure_dir(os.path.join(self.opt.experiment_dir, "Final", "predictions_probs"))
                for i in range(len(batch)):
                    np.save(os.path.join(self.opt.experiment_dir, "Final", "predictions_probs", names[i][0]) , preds_probs[i].cpu().numpy())
                    np.save(os.path.join(self.opt.experiment_dir, "Final", "predictions", names[i][0]) , preds[i].cpu().numpy())
        
        if is_training:
            net.train()

        iou_scores_array = self.metric_seg_online.compute()
        class_avg = iou_scores_array.nanmean()

        return (100*iou_scores_array).cpu().numpy().tolist(), 100*class_avg.cpu()
    

    @torch.no_grad()
    def evaluate(self, net, data_loader, epoch):
        is_training = net.training
        if is_training:
            net.eval()
        
        self.metric_seg.reset()
        ensure_dir(os.path.join(self.opt.experiment_dir, f"Epoch_{epoch}_Final", "predictions"))
        ensure_dir(os.path.join(self.opt.experiment_dir, f"Epoch_{epoch}_Final", "predictions_probs"))
        
        for imgs, segs, _, names in tqdm(data_loader):
            imgs = imgs.cuda()
            if self.opt.dataset_type != "med":
                segs[segs==255] = 19
            
            preds_probs = self.seg_refiner.get_pseudo_label(net, imgs, self.weak_mult)
            preds = torch.argmax(preds_probs, dim=1)
            self.metric_seg.update(preds, segs.cuda())
        
        # log results
        iou_scores_array = 100*self.metric_seg.compute()
        class_avg = iou_scores_array.nanmean()
        iou_str = ""
        for item in iou_scores_array:
            iou_str += " & %4.1f" % item
        logger.info("Classwise: %s Avg: %4.1f" % (iou_str, class_avg))
        # save to file
        iou_scores_array = iou_scores_array.cpu().numpy()
        np.save(os.path.join(self.opt.experiment_dir, f"{epoch}_class_iou.npy"), iou_scores_array)
        
        if is_training:
            net.train()
    

    def lr_scheduler(self, optimizer, iter_num, max_iter, gamma=10, power=0.75):
        decay = (1 + gamma * iter_num / max_iter) ** (-power)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.opt.lr * decay
        return optimizer


    def cross_entropy_loss(self, p, q):
        pixel_ce =  -torch.sum(p * torch.log(q + 1e-6), dim=1) # batch x h x w
        loss     = torch.mean(pixel_ce, dim=(-1, -2))
        return loss

    
    @torch.no_grad()
    def get_slice_index(self, net, img):
        is_training = net.training
        if is_training:
            net.eval()
        out = []
        for i in range(img.size()[0]):
            pred = net(img[i].clone().cuda().unsqueeze(0))["out"][0]
            pred = torch.argmax(pred, dim=0)
            if pred.sum() > 256:
                out.append(i)
        
        if is_training:
            net.train()

        return out


    def train_volume(self):
        """
        Code for training the test time policy networks on the target site.
        """
        logger.info("Training Test Time Policy Network")
        n_iters = 1
        n_epochs = 1
        max_iter = len(self.target_loader)*self.opt.n_epochs
        logger.info("Accuracy on Test Set")
        sample_size = 8

        self.weak_mult = self.opt.weak_mult

        while n_epochs <= self.opt.n_epochs:
            # Set epoch
            if n_epochs == 1:
                self.weak_mult = 1
            else:
                self.weak_mult = self.opt.weak_mult
            if self.opt.dist:
                self.target_loader.sampler.set_epoch(n_epochs)

            self.hard_opt_aug.policy_predictor.reset_weights()            
            
            for x, label, _, names in self.target_loader:
                # reset augmentors
                # batch size is one
                x = x[0]
                label = label[0]

                if self.opt.apply_lr_scheduler:
                    self.lr_scheduler(self.optimizer_net, n_iters, max_iter)
                
                slice_indices = list(range(x.size(0)))
                n_repeat = 8
                
                ## randomly slect batch size images and update the model for 10 iterations.
                for _ in range(n_repeat):
                    # make saved gradients zero
                    self.optimizer_net.zero_grad()
                    rand_index = random.choices(slice_indices, k=sample_size)
                    batch_x = x[rand_index]
                    batch_x = batch_x.cuda()

                    with torch.no_grad():
                        # get pseudo labels using the easy augmentations w.r.t. ema model
                        labels_batch_x_ema = self.seg_refiner.get_pseudo_label(self.ema_net, batch_x, self.weak_mult)
                        soft_pseudo_labels = labels_batch_x_ema

                    # Optimize adversarial augmentations
                    hard_losses = self.hard_opt_aug.optimize(batch_x) #hard_pseudo_labels

                    if self.opt.replay_policy:
                        self.policy_replay.sample(self.hard_opt_aug.policy_predictor)
                
                    # sample hard augmentations
                    x_aug_hard, x_aug_hard_pseudo_labels = self.hard_opt_aug.sample(batch_x, soft_pseudo_labels) # batch, aug_mult, ch, h, w
                
                    x_aug_hard = x_aug_hard.detach() # no gradients
                    self.metric_tracker.update_metrics(hard_losses)

                    pred_no_ema = F.softmax(self.net(batch_x)["out"], dim=1)

                    pred_all_gpus = concat_all_gather(pred_no_ema, rank=self.opt.rank)
                    loss_cm_cent  = self.opt.world_size *  self.crterian_cm(pred_all_gpus)
                
                    (loss_cm_cent.mean()).backward()

                    # update logger
                    self.metric_tracker.update_metrics({
                        "div": (loss_cm_cent.mean()).detach().cpu().item(),
                    })

                    # accumulate gradients for no augmentations
                    pred_normal = F.softmax(self.net(batch_x)["out"], dim=1)
                    loss_teach_normal = self.cross_entropy_loss(pred_normal, soft_pseudo_labels)
                    loss_teach_normal.mean().backward()

                    self.metric_tracker.update_metrics({
                        "teach_N": loss_teach_normal.mean().detach().cpu().item()
                    })
                

                    # teacher student knowledge distillation
                    for j in range(self.opt.aug_mult):
                        pred_aug = F.softmax(self.net(x_aug_hard[:, j])["out"], dim=1)
                        label_aug = x_aug_hard_pseudo_labels[:, j]

                        loss_teach = F.kl_div(torch.log(pred_aug + 1e-6), label_aug, reduction="none").sum(dim=1).mean(dim=(-1, -2))
                        (self.opt.lambda_2*loss_teach.mean() /self.opt.aug_mult).backward()
                        
                        # update logger
                        self.metric_tracker.update_metrics({
                            "teach": loss_teach.detach().mean().cpu().item(),
                        })

                    clip_gradient(self.optimizer_net)
                
                    self.optimizer_net.step()

                    # update teacher
                    self.ema_net.update(self.net.module)

                # batch evaluate
                self.metric_seg_online.reset()
                if self.opt.n_epochs != 1:
                    names = None
                
                iou, avg = self.evaluate_batch(self.ema_net, x, label, names)

                # print metrics on terminal
                if n_iters % 20 == 0:
                    loss_str = "%d>>>" % n_iters
                    for k, v in self.metric_tracker.current_metrics().items():
                        loss_str += " %s: %.3f" % (k, v)
                    logger.info(loss_str)
                
                iou_str = ""
                for item in iou:
                    iou_str += " & %4.1f" % item
                logger.info("Online eval: %s Avg: %4.1f" % (iou_str, avg))
                n_iters += 1
            
            # time.sleep(2)
            # Offline Evaluation
            if self.opt.n_epochs != 1:
                logger.info(f"Evaluating after {n_epochs} {n_iters} epochs on Test Set ...")
                self.evaluate(self.ema_net, self.target_eval_loader, n_epochs)

            ## save model
            self.save_model(n_epochs)
            
            # Next epoch
            n_epochs +=1




    def train_image(self):
        """
        Code for training the test time policy networks on the target site.
        """
        logger.info("Training Test Time Policy Network")
        n_iters = 1
        n_epochs = 1
        max_iter = len(self.target_loader)*self.opt.n_epochs

        self.weak_mult = self.opt.weak_mult

        while n_epochs <= self.opt.n_epochs:
            # Set epoch
            if (n_epochs == 1 and self.opt.n_epochs != 1) :#or (self.opt.n_epochs == 1 and n_iters <= 60):
                self.weak_mult = 1
            else:
                self.weak_mult = self.opt.weak_mult
            if self.opt.dist:
                self.target_loader.sampler.set_epoch(n_epochs)

            self.hard_opt_aug.policy_predictor.reset_weights()            
            self.metric_seg_online.reset()

            for x, label, _, _ in self.target_loader:
                ## apply lr scheduler
                if self.opt.apply_lr_scheduler:
                    self.lr_scheduler(self.optimizer_net, n_iters, max_iter)
                
                ## put current queue and labels on gpu
                batch_x = x.cuda()

                with torch.no_grad():
                    # get pseudo labels using the easy augmentations w.r.t. ema model
                    labels_batch_x_ema = self.seg_refiner.get_pseudo_label(self.ema_net, batch_x, self.weak_mult)
                    soft_pseudo_labels = labels_batch_x_ema

                # Optimize adversarial augmentations
                hard_losses = self.hard_opt_aug.optimize(batch_x) #hard_pseudo_labels

                # replay policy buffer
                if self.opt.replay_policy:
                    self.policy_replay.sample(self.hard_opt_aug.policy_predictor)
                
                # sample hard augmentations
                x_aug_hard, x_aug_hard_pseudo_labels = self.hard_opt_aug.sample(batch_x, soft_pseudo_labels) # batch, aug_mult, ch, h, w
                
                x_aug_hard = x_aug_hard.detach() # no gradients
                self.metric_tracker.update_metrics(hard_losses)
                

                # make saved gradients zero
                self.optimizer_net.zero_grad()

                # marke marginal distribution uniform by maximizing entropy of class marginals
                # as batch is distributed over gpus, collect them and pass correct gradients
                pred_no_ema = F.softmax(self.net(batch_x)["out"], dim=1)

                # gather pred from all gpus
                pred_all_gpus = concat_all_gather(pred_no_ema, rank=self.opt.rank)
                loss_cm_cent  = self.opt.world_size *  self.crterian_cm(pred_all_gpus)
                
                (loss_cm_cent.mean()).backward()

                # update logger
                self.metric_tracker.update_metrics({
                    "div": (loss_cm_cent.mean()).detach().cpu().item(),
                })

                # accumulate gradients for no augmentations
                pred_normal = F.softmax(self.net(batch_x)["out"], dim=1)
                loss_teach_normal = self.cross_entropy_loss(pred_normal, soft_pseudo_labels)
                loss_teach_normal.mean().backward()

                self.metric_tracker.update_metrics({
                    "teach_N": loss_teach_normal.mean().detach().cpu().item()
                })
                

                # teacher student knowledge distillation
                for j in range(self.opt.aug_mult):
                    pred_aug = F.softmax(self.net(x_aug_hard[:, j])["out"], dim=1)
                    label_aug = x_aug_hard_pseudo_labels[:, j]

                    loss_teach = F.kl_div(torch.log(pred_aug + 1e-6), label_aug, reduction="none").sum(dim=1).mean(dim=(-1, -2))
                    # loss_teach = self.cross_entropy_loss(label_aug, pred_aug).mean()
                    # loss_teach = F.nll_loss(torch.log(pred_aug + 1e-8), torch.argmax(label_aug, dim=1)).mean()
                    (self.opt.lambda_2*loss_teach.mean() /self.opt.aug_mult).backward()
                    
                    # update logger
                    self.metric_tracker.update_metrics({
                        "teach": loss_teach.detach().mean().cpu().item(),
                    })

                clip_gradient(self.optimizer_net)
                
                self.optimizer_net.step()

                # update teacher
                self.ema_net.update(self.net.module)

                # batch evaluate
                iou, avg = self.evaluate_batch(self.ema_net, batch_x, label)

                # print metrics on terminal
                if n_iters % 20 == 0:
                    loss_str = "%d>>>" % n_iters
                    for k, v in self.metric_tracker.current_metrics().items():
                        loss_str += " %s: %.3f" % (k, v)
                    logger.info(loss_str)
                
                iou_str = ""
                for item in iou:
                    iou_str += " & %4.1f" % item
                # print("Online eval: %s Avg: %4.1f" % (iou_str, avg))
                logger.info("Online eval: %s Avg: %4.1f" % (iou_str, avg))

                
                

                n_iters += 1
            
            
            # time.sleep(2)
            # Offline Evaluation
            if self.opt.n_epochs != 1:
                logger.info(f"Evaluating after {n_epochs} {n_iters} epochs on Test Set ...")
                self.evaluate(self.ema_net, self.target_loader, n_epochs)

            ## save model
            self.save_model(n_epochs)
            
            # Next epoch
            n_epochs +=1
