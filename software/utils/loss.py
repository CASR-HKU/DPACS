import os
from tensorboardX import SummaryWriter
import math
import torch
import torch.nn as nn


class SampleAdaptor:
    def __init__(self, weight, budget, config_file="config/sample/relu/default.json", num_epochs=100, **kwargs):
        self.budget = budget
        self.weight = weight
        self.relu = nn.ReLU()
        self.num_epochs = num_epochs
        self.parse_config(config_file)

    def parse_config(self, file):
        import json
        with open(file, "r") as load_f:
            load_dict = json.load(load_f)
        self.type = load_dict["type"]
        self.detach = bool(load_dict["detach"])
        self.over = load_dict["over"]
        self.under = load_dict["under"]
        self.begin_epoch = self.num_epochs * load_dict["begin_ratio"]
        self.flops_ratio_range = load_dict["flops_range"]
        if self.type == "relu":
            self.multiply = load_dict["multiply"]
            self.add = load_dict["add"]
        elif self.type == "distribute":
            over_param = load_dict["over_param"]
            self.over_A, self.over_B, self.over_C, self.over_D = \
                over_param["over_A"], over_param["over_B"], over_param["over_C"], over_param["over_D"]
            under_param = load_dict["under_param"]
            self.under_A, self.under_B, self.under_C, self.under_D, self.under_E, self.under_F \
                = under_param["under_A"], under_param["under_B"], under_param["under_C"], under_param["under_D"], \
                  under_param["under_E"], under_param["under_F"]

    def extract_meta(self, outputs, targets):
        pos = torch.softmax(outputs, dim=1)
        preds = torch.max(outputs, dim=1)[1]
        target_pos = []
        for p, target in zip(pos, targets):
            target_pos.append(p[target])
        return torch.max(pos, dim=1)[0], preds, torch.Tensor(target_pos).cuda()

    def flops_dist_limit(self, dists):
        dists = self.relu(dists - self.flops_ratio_range) + self.flops_ratio_range
        dists = self.relu(dists + self.flops_ratio_range) - self.flops_ratio_range
        return dists

    def update(self, outputs, targets, meta):
        if meta["epoch"] < self.begin_epoch:
            return torch.tensor([torch.tensor(1) for _ in range(meta["flops"].shape[-1])]).cuda()

        self.max_FLOPs = meta["flops_full"].sum((0))[0]
        flops_ratio = meta["flops"].sum(0)/self.max_FLOPs
        possibs, preds, target_pos = self.extract_meta(outputs, targets)
        conf_dist = possibs - target_pos
        flops_dist = flops_ratio - self.budget
        if self.flops_ratio_range > 0:
            flops_dist = self.flops_dist_limit(flops_dist)
        if self.type == "relu":
            alpha = self.over * flops_dist**2 * self.relu(target_pos - 0.5) ** 2 - \
                    self.under * conf_dist * self.relu(0.5 - conf_dist) ** 2 * flops_dist**2
            weights = self.weight * (torch.exp(self.multiply * alpha) + self.add)
        elif self.type == "distribute":
            correct_dist = self.over_A*possibs**2 + 2*self.over_B*possibs*flops_dist - self.over_B*flops_dist + \
                           self.over_C*possibs + self.over_D
            wrong_dist = self.under_A*conf_dist**2 + 2*self.under_B*conf_dist*flops_dist + self.under_C*flops_dist**2 \
                         + self.under_D*conf_dist + self.under_E*flops_dist + self.under_F
            whole_dist = correct_dist * (conf_dist == 0).int() * self.over + \
                         wrong_dist * (conf_dist != 0).int() * self.under
            weights = self.weight * torch.exp(whole_dist)
        else:
            raise NotImplementedError
        return weights.detach() if self.detach else weights


class WeightCurriculum:
    def __init__(self, num_epochs, n_weight, s_weight, c_weight, strategy="static", **kwargs):
        self.num_epochs = num_epochs
        self.strategy = strategy
        self.network_weight, self.spatial_weight, self.channel_weight = n_weight, s_weight, c_weight

    def update(self, epoch):
        if self.strategy == "static":
            return self.network_weight, self.spatial_weight, self.channel_weight
        else:
            raise NotImplementedError


class LayerCurriculum:
    def __init__(self, budget, num_epochs, backbone="resnet50", granularity="stage", layer_strategy="static", **kwargs):
        assert granularity in ["stage", "block"]
        assert backbone in ["resnet50", "resnet101", "MobileNetV2", "resnet32_BN", "resnet34", "resnet18",
                            "MobileNetV2_32x32"]
        self.backbone = backbone
        self.num_weights = 16 if backbone == "resnet50" else 36
        self.strategy = layer_strategy
        self.budget = budget
        self.num_epochs = num_epochs

    def update(self, epoch):
        if self.strategy == "static":
            return [self.budget for _ in range(self.num_weights)]
        elif self.strategy == "sensitive":
            budgets = []
            if self.backbone == "resnet50":
                stages = [3, 4, 6, 4]
                budget_diffs = [-0.1, 0.1, 0, -0.1]
                for s, b in zip(stages, budget_diffs):
                    for _ in range(s):
                        budgets.append(self.budget + b)
                return budgets
            elif self.backbone == "resnet32_BN":
                stages = [5, 5, 5]
                budget_diffs = [-0.1, 0.1, 0]
                for s, b in zip(stages, budget_diffs):
                    for _ in range(s):
                        budgets.append(self.budget + b)
                return budgets
        else:
            raise NotImplementedError


class SpatialLoss(nn.Module):
    '''
    Defines the sparsity loss, consisting of two parts:
    - network loss: MSE between computational budget used for whole network and target
    - block loss: sparsity (percentage of used FLOPS between 0 and 1) in a block must lie between upper and lower bound.
    This loss is annealed.
    '''

    def __init__(self, sparsity_target, unlimited_lower=False, layer_loss_method="flops", **kwargs):
        super(SpatialLoss, self).__init__()
        self.sparsity_target = sparsity_target
        self.lower_unlimited = unlimited_lower
        self.layer_loss_method = layer_loss_method
        self.LayerWeights = LayerCurriculum(sparsity_target, layer_strategy="static", **kwargs)
        self.relu = nn.ReLU()

    def calculate_layer_ratio(self, m_dil, m):
        c = m_dil.hard.sum((1,2,3)) * m_dil.flops_per_position + m.hard.sum((1,2,3)) * m.flops_per_position
        t = m_dil.total_positions/ m_dil.hard.shape[0] * m_dil.flops_per_position + \
            m.total_positions/ m.hard.shape[0] * m.flops_per_position
        if self.layer_loss_method == "flops":
            try:
                layer_perc = c / t
            except RuntimeError:
                layer_perc = torch.true_divide(c, t)
        elif self.layer_loss_method == "later_mask":
            layer_perc = m.hard.sum()/m.hard.numel()
        elif self.layer_loss_method == "front_mask":
            layer_perc = m_dil.hard.sum()/m_dil.hard.numel()
        else:
            raise NotImplementedError(self.layer_loss_method)
        return layer_perc, c, t

    def calculate_layer_ratio_whole(self, m_dil, m):
        c = m_dil.active_positions * m_dil.flops_per_position + m.active_positions * m.flops_per_position
        t = m_dil.total_positions * m_dil.flops_per_position + m.total_positions * m.flops_per_position
        if self.layer_loss_method == "flops":
            try:
                layer_perc = c / t
            except RuntimeError:
                layer_perc = torch.true_divide(c, t)
        elif self.layer_loss_method == "later_mask":
            layer_perc = m.hard.sum()/m.hard.numel()
        elif self.layer_loss_method == "front_mask":
            layer_perc = m_dil.hard.sum()/m_dil.hard.numel()
        else:
            raise NotImplementedError(self.layer_loss_method)
        return layer_perc, c, t

    def forward(self, meta, sample_weight=None):
        if sample_weight is None:
            sample_weight = torch.tensor([torch.tensor(1) for _ in range(meta["flops"].shape[-1])]).cuda()

        mask_percents, mask_percents = [], []
        loss_mask = torch.tensor(.0).to(device=meta['device'])
        layer_weights = self.LayerWeights.update(meta["epoch"])

        for i, (mask, weight) in enumerate(zip(meta['masks'], layer_weights)):
            m_dil, m = mask['dilate'], mask['std']
            if m is None:
                continue

            layer_perc, c, t = self.calculate_layer_ratio(m_dil, m)
            mask_percents.append(torch.mean(layer_perc))
            assert 0 <= torch.mean(layer_perc) <= 1, layer_perc
            loss_mask += (self.relu(layer_perc - weight)**2 * sample_weight).mean()  # upper bound
            if not self.lower_unlimited:
                loss_mask += (self.relu(weight - layer_perc)**2 * sample_weight).mean()  # lower bound

        loss_mask /= len(meta['masks'])
        return loss_mask, mask_percents


class FLOPsReductionLoss(nn.Module):
    def __init__(self, budget, unlimited_lower=False, **kwargs):
        super(FLOPsReductionLoss, self).__init__()
        self.budget = budget
        self.unlimited_lower = unlimited_lower
        self.sparse_loss = torch.zeros(1).cuda()
        self.relu = nn.ReLU()

    def forward(self, meta, sample_budgets=None, sample_weight=None):
        if sample_budgets is None:
            budgets = torch.tensor([self.budget for _ in range(meta["flops"].shape[-1])]).cuda()
        else:
            budgets = torch.tensor([self.budget + sample_budgets[idx] for idx in range(meta["flops"].shape[-1])]).cuda()

        if sample_weight is None:
            sample_weight = torch.tensor([torch.tensor(1) for _ in range(meta["flops"].shape[-1])]).cuda()

        sparse_loss = torch.zeros(1).cuda()
        flops, flops_full = torch.sum(meta["flops"], (0)), torch.sum(meta["flops_full"], (0))
        reduction_ratio = flops / flops_full
        if self.budget > 0:
            sparse_loss += ((self.relu(reduction_ratio - budgets) ** 2)*sample_weight).mean()
            if not self.unlimited_lower:
                sparse_loss += ((self.relu(budgets - reduction_ratio) ** 2)*sample_weight).mean()
        return sparse_loss, flops


class ChannelLoss(nn.Module):
    def __init__(self, budget, unlimited_lower=False, **kwargs):
        super(ChannelLoss, self).__init__()
        self.budget = budget
        self.unlimited_lower = unlimited_lower
        self.LayerWeights = LayerCurriculum(budget, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, meta, sample_weight=None):
        if sample_weight is None:
            sample_weight = torch.tensor([torch.tensor(1) for _ in range(meta["flops"].shape[-1])]).cuda()

        channel_loss, channel_percents = torch.zeros(1).cuda(), []
        if 0 < self.budget < 1:
            channel_cnt = 0
            for _, vector in meta["channel_prediction"].items():
                channel_cnt += 1
                channel_percent = torch.true_divide(vector.sum((1)), vector.shape[-1])
                channel_percents.append(torch.mean(channel_percent))
                assert 0 <= torch.mean(channel_percent) <= 1, torch.mean(channel_percent)
                channel_loss += ((self.relu(channel_percent - self.budget) ** 2) * sample_weight).mean()
                if not self.unlimited_lower:
                    channel_loss += ((self.relu(self.budget - channel_percent) ** 2) * sample_weight).mean()
            channel_loss /= channel_cnt
        else:
            channel_loss = meta["lasso_sum"]
        return channel_loss, channel_percents


class AdaptiveLoss(nn.Module):
    def __init__(self, budget, conf_thresh=0.8, cd_thresh=0.1, over_weight=1, under_weight=1, **kwargs):
        super(AdaptiveLoss, self).__init__()
        self.budget = budget
        self.conf_thresh = conf_thresh
        self.cd_thresh = cd_thresh
        self.over_weight = over_weight
        self.under_weight = under_weight

    def over_computing_loss(self, flops_ratio, prob_max, correctness, t="mul"):
        target_samples = (flops_ratio > self.budget) * correctness * prob_max > self.conf_thresh
        if t == "add":
            return sum(((flops_ratio - self.budget) + (prob_max - self.conf_thresh)) * target_samples)
        elif t == "mul":
            return sum(((flops_ratio - self.budget) * (prob_max - self.conf_thresh)) * target_samples)
        else:
            raise NotImplementedError

    def under_computing_loss(self, flops_ratio, correctness, conf_dist, t="mul"):
        target_samples = (flops_ratio < self.budget) * ~correctness * conf_dist > self.conf_thresh
        if t == "add":
            return sum(((self.budget - flops_ratio) + conf_dist) * target_samples)
        elif t == "mul":
            return sum(((self.budget - flops_ratio) * conf_dist) * target_samples)
        else:
            raise NotImplementedError
        
    def forward(self, meta, output, target):
        flops_ratio = torch.sum(meta["flops"], (0)) / torch.sum(meta["flops_full"], (0))
        prob_max = torch.softmax(output, dim=1).max(dim=1)[0]
        correctness = torch.max(output, dim=1)[1] == target
        target_conf = torch.tensor([p[t] for p, t in zip(torch.softmax(output, dim=1), target)]).cuda()
        confidence_dist = prob_max - target_conf
        return self.over_computing_loss(flops_ratio, prob_max, correctness) * self.over_weight + \
               self.under_computing_loss(flops_ratio, correctness, confidence_dist) * self.under_weight


class Loss(nn.Module):
    def __init__(self, network_weight, spatial_weight, channel_weight, network_budget, spatial_budget, channel_budget,
                 tensorboard_folder, num_epochs, loss_config="balance", adaptive_weight=0, record_gradient=False,
                 **kwargs):
        super(Loss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss().to(device="cuda:0")
        self.budget = network_budget
        self.loss_config = loss_config
        if network_budget != -1:
            if loss_config == "default":
                self.weights = WeightCurriculum(num_epochs, network_weight, spatial_weight, channel_weight)
                self.channel_loss = ChannelLoss(channel_budget, num_epochs=num_epochs, **kwargs)
                self.spatial_loss = SpatialLoss(spatial_budget, num_epochs=num_epochs, **kwargs)
                self.network_loss = FLOPsReductionLoss(network_budget, **kwargs)
            elif loss_config == "balance":
                self.network_loss = FLOPsReductionLoss(network_budget, **kwargs)
                kwargs.pop("unlimited_lower")
                self.weights = WeightCurriculum(num_epochs, network_weight, spatial_weight, channel_weight)
                self.channel_loss = ChannelLoss(math.sqrt(network_budget), num_epochs=num_epochs, unlimited_lower=True, **kwargs)
                self.spatial_loss = SpatialLoss(math.sqrt(network_budget), num_epochs=num_epochs, unlimited_lower=True, **kwargs)
            else:
                raise NotImplementedError
        self.a_weight = adaptive_weight
        if tensorboard_folder:
            os.makedirs(tensorboard_folder, exist_ok=True)
        self.tb_writer = SummaryWriter(tensorboard_folder) if tensorboard_folder else ""
        self.record_gradient = record_gradient

    def clear(self):
        self.n_loss, self.s_loss, self.c_loss, = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()
        self.percent_spatial, self.percent_channel = [], []

    def forward(self, output, target, meta, sample_budgets=None, phase="train"):
        self.clear()
        self.t_loss = self.task_loss(output, target)
        if self.budget != -1:
            self.n_weight, self.s_weight, self.c_weight = self.weights.update(meta["epoch"])
            self.n_loss, self.flops = self.network_loss(meta, sample_budgets)
            self.s_loss, self.percent_spatial = self.spatial_loss(meta)
            self.c_loss, self.percent_channel = self.channel_loss(meta)
            loss = self.t_loss + self.n_weight * self.n_loss + self.s_weight * self.s_loss + self.c_weight * self.c_loss
        else:
            loss = self.t_loss
            self.flops = torch.sum(meta["flops_full"], (0))
        return loss

