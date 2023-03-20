import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.augmentation.img_ops.autosegment import apply_invert_affine, apply_affine
from .aug_predictor import PolicyPredictor

import networks.augmentation.med_seg_policies as medpolicy
import networks.augmentation.rgb_seg_policies as rgbpolicy
from losses.seg_losses import EntropyLoss


class SegOptAug(nn.Module):

    def __init__(self, img_type:str, encoder : nn.Module, sub_policy_dim : int, aug_mult : int, name : str, normalize_fn, denormalize_fn, lambda_1=1.0) -> None:
        super().__init__()
        self.lambda_1 = lambda_1
        # Set Encoder
        self.encoder = encoder

        # policy getter
        self.get_sub_policies = medpolicy.get_sub_policies if img_type=="med" else rgbpolicy.get_sub_policies
        self.apply_augment_fn = medpolicy.apply_augment if img_type=="med" else rgbpolicy.apply_augment

        # Build sub Policy predictor Module
        self.aug_mult = aug_mult
        self.sub_policy_dim = sub_policy_dim
        self.name = name
        self.sub_policies = torch.tensor(self.get_sub_policies(self.sub_policy_dim))
        self.policy_predictor = PolicyPredictor(len(self.sub_policies), self.sub_policy_dim, self.name)

        # Set optimizer
        self.optimizer_policy = torch.optim.Adam([{"params": self.policy_predictor.policy_selection_weights, "lr":0.001},
                                                  {"params": self.policy_predictor.policy_mag_weights, "lr":0.01}])

        # Load Normalization layers
        self.normalize_fn = normalize_fn
        self.denormalize_fn = denormalize_fn

        # Set Loss Functions
        self.criterion_ent = EntropyLoss()
        self.criterian_l2 = torch.nn.MSELoss()

        # Set normalization hooker if model has normalization layers
        self.current_norm_inputs = {}

        # For removing hooks
        self.hook_handlers = {}
        self.register_norm_hooks()
        

    def norm_hook(self, idx):
        def hook(module, input, output):
            input = input[0]
            self.current_norm_inputs[idx] = [input.mean(dim=(-2, -1)), input.var(dim=(-2, -1))]
        return hook


    def register_norm_hooks(self):
        idx = 0
        for m in self.encoder.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.SyncBatchNorm) or isinstance(m, torch.nn.LayerNorm):
                self.hook_handlers[idx] = m.register_forward_hook(self.norm_hook(idx))
                idx += 1


    def compute_norm_stat_loss(self, pre_norm_feats_curr, pre_norm_feats_target):
        loss = torch.zeros(pre_norm_feats_curr[0][0].size(0)).cuda()
        
        for i in range(len(pre_norm_feats_curr)):
            target_mean = pre_norm_feats_target[i][0].detach()
            curr_mean = pre_norm_feats_curr[i][0]

            loss += (curr_mean - target_mean).pow(2).mean(dim=-1)
        return loss / len(pre_norm_feats_curr)
    

    def sample_apply_augmentation(self, apply_fn, sub_policies, x, sub_policy_indices, sub_policy_mags):
        """
        Applies given sub-policies on the single image x.
        x shape is [ch x h x w]
        sub_policy_indices int.
        sub_policy_mags shape is [sub_policy_dim=2]
        """
        fns_idxs = sub_policies[sub_policy_indices] # [sub_policy_dim]

        x = x.unsqueeze(0)
        inv_ops = []
        x_aug = x.clone()

        # apply sub-policy on the original image
        for fn, m in zip(fns_idxs, sub_policy_mags):
            m = m.unsqueeze(0)
            x_aug, inv = apply_fn(x_aug, fn, m) # inv of shape batch x 2 x 3
            
            inv_ops.append(inv)
        
        inv_ops = torch.stack(inv_ops, dim=1) # batch x [sub_policy_dim] x 2 x 3
        return x_aug, inv_ops # [aug_mult x ch x h x w]


    def forward(self):
        return self.policy_predictor()

    @torch.no_grad()
    def sample(self, x_cuda, pseudo_labels):
        x_aug = []
        pseudo_labels_aug = []
        # first sample opt_aug_mult subploicies without replacement
        prob_sub_policy, _ = self.policy_predictor() # [num_sub_policy]
        prob_sub_policy = prob_sub_policy.unsqueeze(0).repeat(x_cuda.size(0), 1)
        selected_sub_policy_idx = torch.multinomial(prob_sub_policy, self.aug_mult, replacement=False) # [batch , aug_mult]

        for k in range(self.aug_mult):
            curr_pseudo_labels = pseudo_labels.clone()
            prob_sub_policy, mag_sub_policy = self.policy_predictor() # [num_sub_policy] [num_sub_policies x policy_dim]
            prob_sub_policy = prob_sub_policy.unsqueeze(0).repeat(x_cuda.size(0), 1) # [batch, num_sub_policy]

            # select magnitude of current aug_mult
            curr_selected_sub_policy_index = selected_sub_policy_idx[:, k] # [batch]
            curr_selected_sub_policy_mag = mag_sub_policy[curr_selected_sub_policy_index,] # [batch, policy_dim]
            
            # apply augmentations based on selected policies
            x_cuda_denorm = self.denormalize_fn(x_cuda)
            x_augs_curr = []
            invs_curr = []
            
            for i, x_item in enumerate(x_cuda_denorm):
                x_aug_curr, inv_curr = self.sample_apply_augmentation(self.apply_augment_fn,
                                                                      self.sub_policies,
                                                                      x_item,
                                                                      curr_selected_sub_policy_index[i],
                                                                      curr_selected_sub_policy_mag[i, :])  
                
                x_augs_curr.append(x_aug_curr)
                invs_curr.append(inv_curr)
            
            x_augs_curr = torch.cat(x_augs_curr, dim=0)
            x_augs_curr = self.normalize_fn(x_augs_curr)
            invs_curr = torch.cat(invs_curr, dim=0)

            for i_policy_dim in range(self.sub_policy_dim):
                curr_pseudo_labels = apply_affine(curr_pseudo_labels, invs_curr[:, i_policy_dim])

            # gather current augmentations
            x_aug.append(x_augs_curr.detach())
            pseudo_labels_aug.append(curr_pseudo_labels.detach())
            
        
        x_aug = torch.stack(x_aug, dim=1) # [batch, aug_mult, ch, h, w]
        pseudo_labels_aug = torch.stack(pseudo_labels_aug, dim=1) #  [batch, aug_mult, n_classes, h, w]
        return x_aug.detach(), pseudo_labels_aug.detach()        


    def optimize(self, x_cuda):
        """
        Find Hard augmentations with respect to the momentum encoder.
        """
        avg_pred = 0

        # save original bn stats
        with torch.no_grad():
            self.encoder(x_cuda)
            orig_norm_stats = {}
            for k, v in self.current_norm_inputs.items():
                orig_norm_stats[k] = [v[0].detach(), v[1].detach()]
        
        self.optimizer_policy.zero_grad()

        # first sample opt_aug_mult subploicies without replacement
        with torch.no_grad():
            prob_sub_policy = torch.ones(len(self.sub_policies), dtype=torch.float32, device=x_cuda.device) / len(self.sub_policies)
            prob_sub_policy = prob_sub_policy.unsqueeze(0).repeat(x_cuda.size(0), 1)
            selected_sub_policy_idx = torch.multinomial(prob_sub_policy, self.aug_mult, replacement=False) # [batch , aug_mult]
        # Losses Tracker
        all_losses = {"H_ce_aug": [], "H_kl_aug": [], "loss_policy": [], "H_norm" : []}

        for k in range(self.aug_mult):
            prob_sub_policy, mag_sub_policy = self.policy_predictor() # [num_sub_policy] [num_sub_policies x policy_dim]
            prob_sub_policy = prob_sub_policy.unsqueeze(0).repeat(x_cuda.size(0), 1) # [batch, num_sub_policy]

            # select magnitude of current aug_mult
            curr_selected_sub_policy_index = selected_sub_policy_idx[:, k] # [batch]
            curr_selected_sub_policy_mag = mag_sub_policy[curr_selected_sub_policy_index,] # [batch, policy_dim]
            curr_selected_prob_sub_policy = prob_sub_policy[torch.arange(x_cuda.size(0)), curr_selected_sub_policy_index] # [batch]
            
            # apply augmentations based on selected policies
            x_cuda_denorm = self.denormalize_fn(x_cuda)
            x_augs_curr = []
            invs_curr = []
            for i, x_item in enumerate(x_cuda_denorm):
                x_aug_curr, inv_curr = self.sample_apply_augmentation(self.apply_augment_fn,
                                                                      self.sub_policies,
                                                                      x_item,
                                                                      curr_selected_sub_policy_index[i],
                                                                      curr_selected_sub_policy_mag[i, :])
                x_augs_curr.append(x_aug_curr)
                invs_curr.append(inv_curr)
            
            x_augs_curr = torch.cat(x_augs_curr, dim=0) # batch x ch x h x w
            x_augs_curr = self.normalize_fn(x_augs_curr)

            invs_curr = torch.cat(invs_curr, dim=0) # batch x [sub_policy_dim] x 2 x 3

            # get prediction of augmentation from the encoder
            pred = F.softmax(self.encoder(x_augs_curr)["out"], dim=1)

            # apply invs spatial operations for getting segmentations in the original space
            for i_policy_dim in reversed(range(self.sub_policy_dim)):
                pred = apply_invert_affine(pred, invs_curr[:, i_policy_dim])

            # update the avg_pred
            avg_pred = (avg_pred * k + pred.detach()) / (k+1)

            loss_kl = torch.nn.functional.kl_div(torch.log(pred + 1e-8),
                                                 avg_pred, log_target=False, reduction="none").sum(dim=(1, 2, 3))
            loss_kl = loss_kl / (pred.size(2) * pred.size(3))

            # maximize the cross entropy loss based on pseudo labels
            # loss_ce_aug = torch.nn.functional.nll_loss(torch.log(pred + 1e-8), pseudo_labels, reduction="none").mean(dim=(1, 2))
            loss_ce_aug = self.criterion_ent(pred)
            # print(loss_ce_aug.size())
            # minimize bn norm stats
            currn_norm_stats = self.current_norm_inputs
            loss_norm = self.lambda_1 * self.compute_norm_stat_loss(currn_norm_stats, orig_norm_stats) #self.compute_norm_stat_loss(currn_norm_stats, orig_norm_stats)
            # print(loss_norm.size())
            # print(loss_kl.size())
            (-loss_ce_aug.mean() + loss_norm.mean() - loss_kl.mean()).backward()
            loss_curr = (-loss_ce_aug + loss_norm - loss_kl).detach()

            # optimize the probability of selecting this policy
            loss_policy = torch.mean(loss_curr * torch.log(curr_selected_prob_sub_policy + 1e-8))
            loss_policy.backward()

            # gather current augmentations
            # x_aug.append(x_aug_curr.detach())
            # only track final loss for augs
            all_losses["H_ce_aug"].append(loss_ce_aug.mean().detach().cpu().item())
            all_losses["H_kl_aug"].append(loss_kl.mean().detach().cpu().item())
            all_losses["H_norm"].append(loss_norm.mean().detach().cpu().item())
            all_losses["loss_policy"].append(loss_policy.mean().detach().cpu().item())
        
        self.optimizer_policy.step()
        # x_aug = torch.stack(x_aug, dim=1) # [batch, aug_mult, ch, h, w]
        for k,v in all_losses.items():
            all_losses[k] = torch.tensor(v).mean()
        
        return all_losses


class RefinedPseudoLabels:
    def randomResizeCrop(self, x):
        # TODO: Investigate different scale for x and y
        delta_scale_x = 0.2
        delta_scale_y = 0.2

        scale_matrix_x = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        scale_matrix_y = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)

        translation_matrix_x = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        translation_matrix_y = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)

        delta_x = 0.5 * delta_scale_x * (2*torch.rand(x.size(0), 1, 1, device=x.device) - 1.0)
        delta_y = 0.5 * delta_scale_y * (2*torch.rand(x.size(0), 1, 1, device=x.device) -1.0)

        random_affine = (1 - delta_scale_x) * scale_matrix_x + (1 - delta_scale_y) * scale_matrix_y +\
                    delta_x * translation_matrix_x + \
                    delta_y * translation_matrix_y

        x = apply_affine(x, random_affine)
        return x, random_affine.detach()


    def randomHorizontalFlip(self, x):
        affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
        horizontal_flip = torch.tensor([-1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        # randomly flip some of the images in the batch
        mask = (torch.rand(x.size(0), device=x.device) > 0.5)
        affine[mask] = affine[mask] * horizontal_flip
        x = apply_affine(x, affine)
        return x, affine.detach()


    def randomVerticalFlip(self, x):
        affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
        vertical_flip = torch.tensor([1, 0, 0, 0, -1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        # randomly flip some of the images in the batch
        
        mask = (torch.rand(x.size(0), device=x.device) > 0.5)
        affine[mask] = affine[mask] * vertical_flip
        x = apply_affine(x, affine)
        return x, affine.detach()


    def randomRotate(self, x):
        affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
        rotation = torch.tensor([0, -1, 0, 1, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        # randomly flip some of the images in the batch
        mask = (torch.rand(x.size(0), device=x.device) > 0.5)
        affine[mask] = rotation.repeat(mask.sum(), 1, 1)

        x = apply_affine(x, affine)
        
        return x, affine.detach()

    def apply_invert_affine(self, x, affine):
        # affine shape should be batch x 2 x 3
        # x shape should be batch x ch x h x w

        # get homomorphic transform
        H = torch.nn.functional.pad(affine, [0, 0, 0, 1], "constant", value=0.0)
        H[..., -1, -1] += 1.0

        inv_H = torch.inverse(H)
        inv_affine = inv_H[:, :2, :3]

        grid = torch.nn.functional.affine_grid(inv_affine, x.size(), align_corners=False)
        x = torch.nn.functional.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

        return x

    @torch.no_grad()
    def get_pseudo_label(self, net, x, mult=3):
        preditions_augs = []
        is_training = net.training
        # if is_training:
        #     net.eval()
        outnet = net(x)
        preditions_augs.append(F.softmax(outnet["out"], dim=1))

        for i in range(mult-1):
            # x_aug, rotate_affine = self.randomRotate(x)
            # x_aug, vflip_affine = self.randomVerticalFlip(x_aug)
            # x_aug, hflip_affine = self.randomHorizontalFlip(x_aug)

            x_aug, hflip_affine = self.randomHorizontalFlip(x)
            x_aug, crop_affine  = self.randomResizeCrop(x_aug)

            # get label on x_aug
            outnet = net(x_aug)
            pred_aug = outnet["out"]
            
            pred_aug = F.softmax(pred_aug, dim=1)
            pred_aug = self.apply_invert_affine(pred_aug, crop_affine)
            pred_aug = self.apply_invert_affine(pred_aug, hflip_affine)

            preditions_augs.append(pred_aug)


        preditions = torch.stack(preditions_augs, dim=0).mean(dim=0) # batch x n_classes x h x w
        # renormalize the probability (due to interpolation of zeros, mean does not imply probability distribution over the classes)
        preditions = preditions / torch.sum(preditions, dim=1, keepdim=True)
        # if is_training:
        #     net.train()
        return preditions
