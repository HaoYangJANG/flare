"""Independent full-flow geometric quality field and bounded box guidance.

The field has its own image/query projections. Detector evidence is detached;
only candidate coordinates carry an inference gradient. All boxes are normalized
cxcywh relative to the unpadded image. This is endpoint-energy guidance, not a
claim of exact conditional score sampling. Targets are geometric localization
utility, NOT pure IoU: 0.9 * IoU + 0.1 / (1 + normalized squared center/log-size
distance). This supplies directional supervision even before boxes overlap.

Readiness is checkpointed successful optimizer updates, never forward calls.
The trainer alone increments successful_updates after a real field update;
sample_calls is a separate checkpointed private sampling sequence counter.
"""
import math
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


def local_boxes(anchor: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Center offsets in box-size units; size offsets in logarithmic units."""
    center = anchor[..., :2] + residual[..., :2] * anchor[..., 2:]
    size = anchor[..., 2:] * residual[..., 2:].clamp(-4., 4.).exp()
    return torch.cat((center.clamp(1e-6, 1.-1e-6), size.clamp(1e-6, 1.)), -1)


def box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Broadcasting cxcywh IoU, stable for tiny boxes and bounded in [0, 1].

    Compute overlap from size and center separation rather than subtracting
    rounded corner coordinates; a one-pixel box retains exact self overlap.
    """
    first, second = first.float(), second.float()
    a_size, b_size = first[..., 2:].clamp_min(0), second[..., 2:].clamp_min(0)
    separation = (first[..., :2] - second[..., :2]).abs()
    overlap = torch.minimum(torch.minimum(a_size, b_size), (a_size + b_size) / 2 - separation).clamp_min(0)
    inter = overlap.prod(-1)
    union = a_size.prod(-1) + b_size.prod(-1) - inter
    return torch.nan_to_num(inter / union.clamp_min(torch.finfo(inter.dtype).tiny), nan=0.).clamp(0, 1)


def localization_utility(first: torch.Tensor, target: torch.Tensor,
                         smooth_weight: float = .1) -> torch.Tensor:
    """Geometric utility in [0, 1], exactly 1 at GT, with nonoverlap ordering.

    Center distance is in GT-size units; size distance is the squared log
    ratio. The same target is used for value calibration and probe ranking.
    """
    first, target = first.float(), target.float()
    size = target[..., 2:].clamp_min(1e-6)
    center = (first[..., :2] - target[..., :2]) / size
    log_size = (first[..., 2:].clamp_min(1e-6) / size).log()
    distance = center.square().sum(-1) + log_size.square().sum(-1)
    return ((1 - smooth_weight) * box_iou(first, target)
            + smooth_weight / (1 + distance)).clamp(0, 1)


def bounded_ascent(
    boxes: torch.Tensor, energy: Callable, *, scale: float,
    max_center: float = .03, max_log_size: float = .08,
    backtracks: int = 2, min_gain: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """One bounded ascent step, with per-query finite checks and backtracking.

    No optimizer step, parameter .grad accumulation, random draws or GT access.
    Non-improving candidates return the exact input. The caller gates eligible
    queries and decides at which outer sampling times this operation is used.
    """
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("guidance scale must be finite and nonnegative")
    if not math.isfinite(min_gain) or min_gain < 0:
        raise ValueError('min_gain must be finite and nonnegative')
    for value in (max_center, max_log_size):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('movement limits must be finite and positive')
    if isinstance(backtracks, bool) or not isinstance(backtracks, int) or backtracks < 0:
        raise ValueError('backtracks must be a nonnegative integer')
    accepted = torch.zeros(boxes.shape[:-1], device=boxes.device, dtype=torch.bool)
    if scale == 0 or boxes.numel() == 0:
        return boxes, {"accepted": accepted}
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast(device_type=boxes.device.type, enabled=False):
        accepted = accepted.clone()
        work_dtype = torch.float64 if boxes.dtype == torch.float64 else torch.float32
        anchor = boxes.detach().to(work_dtype).clone()
        valid_anchor = (torch.isfinite(anchor).all(-1)
                        & (anchor[..., :2] >= 1e-6).all(-1)
                        & (anchor[..., :2] <= 1. - 1e-6).all(-1)
                        & (anchor[..., 2:] >= 1e-6).all(-1)
                        & (anchor[..., 2:] <= 1.).all(-1))
        if not valid_anchor.any():
            return boxes, {"accepted": accepted}
        residual = torch.zeros_like(anchor, requires_grad=True)
        initial = energy(local_boxes(anchor, residual))
        if initial.shape != anchor.shape[:-1]:
            raise ValueError("energy must return one scalar per box")
        if not initial.requires_grad:
            return boxes, {"accepted": accepted}
        gradient = torch.autograd.grad(initial.sum(), residual, create_graph=False, allow_unused=True)[0]
        if gradient is None:
            return boxes, {"accepted": accepted}
        finite = torch.isfinite(gradient).all(-1) & torch.isfinite(initial) & valid_anchor
        # Rescale first so a finite but large gradient cannot overflow its norm.
        magnitude = gradient.abs().amax(-1, keepdim=True)
        normalized = torch.nan_to_num(gradient / magnitude.clamp_min(1e-30))
        norm = normalized.norm(dim=-1, keepdim=True)
        finite &= (magnitude[..., 0] > 0) & (norm[..., 0] > 0)
        if not finite.any():
            return boxes, {"accepted": accepted}
        direction = normalized / norm.clamp_min(1e-30)
        budget = anchor.new_tensor([max_center,max_center,max_log_size,max_log_size])
        delta = direction * budget * min(float(scale), 1.)
        result = anchor.clone()
        actual_budget = torch.tensor([max_center, max_center, max_log_size, max_log_size],
                                     device=anchor.device, dtype=torch.float64) * min(float(scale), 1.)
        with torch.no_grad():
            for attempt in range(backtracks+1):
                pending = finite & ~accepted
                if not pending.any():
                    break
                candidate = local_boxes(anchor, delta * (0.5 ** attempt))
                # Judge the representable output, not a float32 improvement
                # that disappears (or overshoots the budget) after AMP rounding.
                candidate = candidate.to(boxes.dtype).to(work_dtype)
                actual, base = candidate.double(), anchor.double()
                actual_delta = torch.cat((
                    (actual[..., :2] - base[..., :2]) / base[..., 2:].clamp_min(1e-6),
                    (actual[..., 2:] / base[..., 2:].clamp_min(1e-6)).clamp_min(1e-12).log(),
                ), -1)
                within_budget = (actual_delta.abs() <= actual_budget).all(-1)
                evaluable = pending & within_budget & torch.isfinite(candidate).all(-1)
                if not evaluable.any():
                    continue
                value = energy(candidate)
                gain = value - initial.detach()
                take = evaluable & torch.isfinite(gain) & (gain > min_gain)
                result = torch.where(take[...,None], candidate, result)
                accepted |= take
    # Keep nonaccepted entries bit-identical, even if source had another dtype.
    result = torch.where(accepted[...,None], result.to(boxes.dtype), boxes).detach()
    return result, {"accepted": accepted.detach()}


def quality_targets(probes, targets, indices, num_classes, duplicate_iou=.5,
                    utility_smooth_weight=.1):
    """Fixed matched identity and geometric utility; ignore EACH duplicate probe.

    An unmatched anchor can enter or leave the duplicate-overlap region under a
    perturbation. Every probe has its own ignore decision; backgrounds retain
    exact zero values. A matched family always keeps its originally assigned GT.
    """
    batch, count, samples, _ = probes.shape
    values = probes.new_zeros(batch,count,samples,num_classes)
    valid = torch.ones_like(values, dtype=torch.bool)
    with torch.no_grad():
        for b, (src, dst) in enumerate(indices):
            gt = targets[b]['boxes'].to(probes)
            labels = targets[b]['labels'].to(probes.device).long()
            src, dst = src.to(probes.device), dst.to(probes.device)
            if gt.numel():
                overlaps = box_iou(probes[b,:,:,None], gt[None,None])
                duplicate = overlaps.amax(-1) >= duplicate_iou
                duplicate[src] = False
                valid[b] &= ~duplicate[..., None]
            if src.numel():
                utility = localization_utility(probes[b,src], gt[dst,None], utility_smooth_weight)
                values[b,src,:,labels[dst]] = utility
    return values, valid


class BoxQualityGuidance(nn.Module):
    """Field over one detached 256-channel res2 map and embed_dim queries.

    Probe families contain the anchor, +/- each axis at the configured guide
    radius (max_* times min(strength, 1)), +/- larger axes, mixed directions at
    both radii, and a GT endpoint for matched queries. mixed_probes counts paired
    directions per radius and must be even; all pairs share fixed semantic IDs.
    """
    def __init__(self, embed_dim: int, num_classes: int, channels: int = 32,
                 hidden_dim: int = 128, max_train_queries: int = 192,
                 strength: float = .5, start_progress: float = .25,
                 max_center: float = .03, max_log_size: float = .08,
                 backtracks: int = 2, score_power: float = 0.,
                 score_threshold: float = .05, warmup_updates: int = 1000,
                 ramp_updates: int = 1000,
                 loss_weight: float = .5, rank_weight: float = .1,
                 probe_center: float = .12, probe_log_size: float = .20,
                 seed: int = 42, mixed_probes: int = 8,
                 utility_smooth_weight: float = .1):
        super().__init__()
        for name, value in dict(embed_dim=embed_dim,num_classes=num_classes,channels=channels,
                                hidden_dim=hidden_dim,max_train_queries=max_train_queries).items():
            if isinstance(value,bool) or not isinstance(value,int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name,value in dict(strength=strength,score_power=score_power,loss_weight=loss_weight,rank_weight=rank_weight).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name,value in dict(max_center=max_center,max_log_size=max_log_size,probe_center=probe_center,probe_log_size=probe_log_size).items():
            if not math.isfinite(value) or not 0 < value <= .5:
                raise ValueError(f'{name} must lie in (0,.5]')
        if not 0 <= start_progress < 1 or not 0 <= score_threshold < 1:
            raise ValueError('invalid progress or score threshold')
        for value in (backtracks,warmup_updates,ramp_updates,seed,mixed_probes):
            if isinstance(value,bool) or not isinstance(value,int) or value < 0:
                raise ValueError('backtracks, warmup_updates, ramp_updates, seed and mixed_probes must be nonnegative integers')
        if mixed_probes % 2:
            raise ValueError('mixed_probes must be even to preserve symmetric mixed directions')
        if not math.isfinite(utility_smooth_weight) or not 0 < utility_smooth_weight <= 1:
            raise ValueError('utility_smooth_weight must lie in (0, 1]')
        self.num_classes, self.channels = num_classes, channels
        self.max_train_queries, self.seed = max_train_queries, seed
        self.strength, self.start_progress = strength, start_progress
        self.max_center, self.max_log_size, self.backtracks = max_center,max_log_size,backtracks
        self.score_power, self.score_threshold = score_power, score_threshold
        self.warmup_updates, self.ramp_updates = warmup_updates, ramp_updates
        self.loss_weight, self.rank_weight = loss_weight, rank_weight
        self.probe_center, self.probe_log_size = probe_center, probe_log_size
        self.mixed_probes, self.utility_smooth_weight = mixed_probes, utility_smooth_weight
        self.image_projection = nn.Conv2d(256,channels,1)
        self.query_projection = nn.Linear(embed_dim,32)
        self.mlp = nn.Sequential(nn.Linear(25*channels+25+32+num_classes+8,hidden_dim),
                                 nn.SiLU(),nn.Linear(hidden_dim,num_classes))
        # A small nonzero readout permits coordinate gradients from the first update.
        nn.init.normal_(self.mlp[-1].weight,std=.01)
        nn.init.constant_(self.mlp[-1].bias,-2.)
        self.register_buffer('successful_updates',torch.zeros((),dtype=torch.int64))
        self.register_buffer('sample_calls',torch.zeros((),dtype=torch.int64))
        self.last_diagnostics = {}

    def is_ready(self) -> bool:
        """Warmup completed; the first ramp point may still have zero strength."""
        return int(self.successful_updates.item()) >= self.warmup_updates

    def guidance_scale(self) -> float:
        """Configured strength times the linear post-warmup successful-step ramp."""
        updates = int(self.successful_updates.item())
        if updates < self.warmup_updates:
            return 0.
        ramp = 1. if self.ramp_updates == 0 else min(1., (updates - self.warmup_updates) / self.ramp_updates)
        return self.strength * ramp

    def project(self, features, image_sizes, padded_size):
        sizes = torch.as_tensor(image_sizes,device=features.device,dtype=torch.float32)
        padded = torch.as_tensor(padded_size,device=features.device,dtype=torch.float32)
        height,width=features.shape[-2:]
        y=(torch.arange(height,device=features.device)+.5)/height*padded[0]
        x=(torch.arange(width,device=features.device)+.5)/width*padded[1]
        valid=(y[None,:,None]<sizes[:,0,None,None]) & (x[None,None,:]<sizes[:,1,None,None])
        with torch.autocast(device_type=features.device.type,enabled=False):
            feature=features.detach().float().masked_fill(~valid[:,None],0.)
            return F.silu(self.image_projection(feature)).masked_fill(~valid[:,None],0.)

    def predict(self, projected, query, class_logits, boxes, progress, image_sizes, padded_size):
        """[B,Q,4] boxes -> [B,Q,C] quality logits, differentiable in boxes."""
        with torch.autocast(device_type=boxes.device.type,enabled=False):
            batch,count=boxes.shape[:2]
            b=boxes.float()
            size=torch.as_tensor(image_sizes,device=b.device,dtype=torch.float32)
            padded=torch.as_tensor(padded_size,device=b.device,dtype=torch.float32)
            pos=torch.linspace(-.6,.6,5,device=b.device)
            yy,xx=torch.meshgrid(pos,pos,indexing='ij')
            unit=torch.stack((xx,yy),-1).reshape(25,2)
            points=b[...,:2].unsqueeze(-2)+b[...,2:].unsqueeze(-2)*unit
            valid=((points>=0)&(points<=1)).all(-1)
            grid=2*points*(size[:,[1,0]]/padded[[1,0]])[:,None,None]-1
            sampled=F.grid_sample(projected.float(),grid.reshape(batch,count*25,1,2),
                                  mode='bilinear',padding_mode='zeros',align_corners=False)
            sampled=sampled.reshape(batch,self.channels,count,25).permute(0,2,3,1)
            sampled=sampled*valid[...,None]
            query_vector=self.query_projection(query.detach().float())
            p=progress.detach().to(b).reshape(batch,1,1).expand(batch,count,1)
            time=torch.cat((p,p.square(),(p*math.pi).sin(),(p*math.pi).cos()),-1)
            geometry=torch.cat((b[...,:2],b[...,2:].clamp_min(1e-6).log()),-1)
            descriptor=torch.cat((sampled.flatten(-2),valid.float(),query_vector,
                                  class_logits.detach().float().sigmoid(),geometry,time),-1)
            return self.mlp(descriptor)

    def training_losses(self, features, query, logits, boxes, progress, image_sizes,
                        padded_size, targets, indices):
        """Geometric value calibration and all-probe ordering; no detector grads.

        Sampling advances only sample_calls. It never advances readiness or the
        detector RNG, including under AMP, skipped updates and gradient accumulation.
        """
        # Keep a differentiable finite zero even for B=0/Q=0, including all field
        # parameters so distributed training does not gain conditionally unused heads.
        zero = sum(p.reshape(-1)[0] * 0 for p in self.parameters())
        total_quality, total_rank = zero, zero
        generator=torch.Generator().manual_seed((self.seed+int(self.sample_calls.item())) % (2**63-1))
        self.sample_calls.add_(1)
        if boxes.shape[0] == 0:
            return {'loss_quality':zero*self.loss_weight,
                    'loss_quality_rank':zero*self.rank_weight}
        projected=self.project(features,image_sizes,padded_size)
        for batch in range(boxes.shape[0]):
            src,dst=(v.to(boxes.device) for v in indices[batch])
            count=boxes.shape[1]
            positive=src[torch.randperm(len(src),generator=generator).to(src.device)]
            mask=torch.ones(count,device=boxes.device,dtype=torch.bool); mask[src]=False
            negative=torch.nonzero(mask,as_tuple=False).flatten()
            negative=negative[torch.randperm(len(negative),generator=generator).to(negative.device)]
            # Reserve at least a quarter for negatives when both kinds exist.
            cap=min(len(positive),max(1,self.max_train_queries*3//4))
            selected=torch.cat((positive[:cap],negative))[:self.max_train_queries]
            if len(selected)<min(count,self.max_train_queries):
                selected=torch.cat((selected,positive[cap:]))[:self.max_train_queries]
            anchor=boxes[batch:batch+1,selected].detach().float()
            n=len(selected)
            if n == 0:
                continue
            radius = anchor.new_tensor([self.max_center, self.max_center,
                                       self.max_log_size, self.max_log_size]) * min(self.strength, 1.)
            broad = anchor.new_tensor([self.probe_center, self.probe_center,
                                      self.probe_log_size, self.probe_log_size])
            axis = torch.eye(4, device=anchor.device, dtype=anchor.dtype)
            directions = [axis * radius, -axis * radius, axis * broad, -axis * broad]
            if self.mixed_probes:
                mixed = torch.rand(self.mixed_probes // 2, 4, generator=generator) * 2 - 1
                mixed = mixed / mixed.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                mixed = torch.cat((mixed, -mixed)).to(anchor)
                directions += [mixed * radius, mixed * broad]
            # Final zero is replaced only for matched queries, retaining background
            # zero targets for the unmatched families rather than assigning GT to them.
            offsets = torch.cat((anchor.new_zeros(1, 4), *directions, anchor.new_zeros(1, 4)))
            samples = len(offsets)
            probes=local_boxes(anchor[:,:,None],offsets[None,None]).detach()
            remap=torch.full((count,),-1,device=boxes.device,dtype=torch.long)
            remap[selected]=torch.arange(n,device=boxes.device)
            used=remap[src]>=0
            local_src,local_dst=remap[src[used]],dst[used]
            if len(local_src):
                probes[0,local_src,-1]=targets[batch]['boxes'][local_dst].to(probes)
            values,valid=quality_targets(probes,[targets[batch]],[(local_src,local_dst)],self.num_classes,
                                         utility_smooth_weight=self.utility_smooth_weight)
            # Repeat semantic context, keeping a fixed identity across nearby boxes.
            q=query[batch:batch+1,selected].detach()[:,:,None].expand(-1,-1,samples,-1).reshape(1,n*samples,-1)
            c=logits[batch:batch+1,selected].detach()[:,:,None].expand(-1,-1,samples,-1).reshape(1,n*samples,-1)
            pred=self.predict(projected[batch:batch+1],q,c,probes.reshape(1,n*samples,4),
                              progress[batch:batch+1],image_sizes[batch:batch+1],padded_size).reshape(1,n,samples,-1)
            focal=(pred.sigmoid()-values).abs().square()
            calibration=F.binary_cross_entropy_with_logits(pred,values,reduction='none')*focal
            # A dense set of unmatched proposals in an empty image must not
            # dominate the detector objective via a denominator of one.
            normalizer=max(len(local_src)*samples, n*samples/4, 1)
            total_quality=total_quality+(calibration*valid).sum()/normalizer
            if len(local_src):
                labels=targets[batch]['labels'][local_dst].to(boxes.device).long()
                prediction=pred[0,local_src,:,labels]
                truth=values[0,local_src,:,labels]
                # Every unordered pair participates, including anchor/local/mixed
                # probes. This avoids a far-axis-only signal and zero-IoU dead ranks.
                left,right=torch.triu_indices(samples,samples,offset=1,device=boxes.device)
                differences=truth[:,left]-truth[:,right]
                logit_difference=prediction[:,left]-prediction[:,right]
                total_rank=total_rank+(F.softplus(-differences.sign()*logit_difference)*differences.abs()).mean()
            else:
                total_rank=total_rank+pred.sum()*0
        batch_size=max(boxes.shape[0],1)
        return {'loss_quality':total_quality*self.loss_weight/batch_size,
                'loss_quality_rank':total_rank*self.rank_weight/batch_size}

    def guide(self, projected, query, logits, boxes, progress, image_sizes, padded_size):
        """Use only image evidence and predictions. No targets argument exists."""
        scale = self.guidance_scale()
        if scale == 0 or boxes.numel() == 0:
            return boxes,logits
        # All eligibility checks precede predict and coordinate autograd. In
        # particular early noisy phases do zero quality work after projection.
        with torch.no_grad():
            confidence = logits.detach().float().sigmoid().amax(-1)
            eligible = ((confidence >= self.score_threshold)
                        & (progress.detach().reshape(-1, 1) >= self.start_progress))
        if not eligible.any():
            return boxes,logits
        # Clones inside disabled inference mode are necessary when callers use inference_mode.
        with torch.inference_mode(False), torch.enable_grad():
            guided = boxes.detach().clone()
            output_logits = logits.detach().clone() if self.score_power > 0 else logits.detach()
            accepted = 0
            for batch in range(boxes.shape[0]):
                selected = torch.nonzero(eligible[batch], as_tuple=False).flatten()
                if selected.numel() == 0:
                    continue
                # Slice eligible queries before any field evaluation. Image context
                # is per image, while every query keeps its detector class identity.
                feature = projected[batch:batch+1].detach().float().clone()
                context = query[batch:batch+1,selected].detach().float().clone()
                classes = logits[batch:batch+1,selected].detach().float().clone()
                phase = progress[batch:batch+1].detach().float().clone()
                sizes = torch.as_tensor(image_sizes[batch:batch+1],device=boxes.device).clone()
                labels = classes.argmax(-1)
                def energy(candidate):
                    pred = self.predict(feature,context,classes,candidate,phase,sizes,padded_size)
                    return F.logsigmoid(pred.gather(-1,labels[...,None]).squeeze(-1))
                moved,diag = bounded_ascent(boxes[batch:batch+1,selected],energy,scale=scale,
                                           max_center=self.max_center,max_log_size=self.max_log_size,
                                           backtracks=self.backtracks)
                accepted += int(diag['accepted'].sum())
                with torch.no_grad():
                    guided[batch,selected] = moved[0]
                    if self.score_power > 0:
                        quality = self.predict(feature,context,classes,moved.float(),phase,sizes,padded_size).sigmoid()
                        probability = (classes.sigmoid() * quality.pow(self.score_power)).clamp(1e-6,1-1e-6)
                        output_logits[batch,selected] = torch.logit(probability)[0].to(logits.dtype)
            self.last_diagnostics={'eligible':int(eligible.sum()),'accepted':accepted,
                                   'scale':scale,'max_abs_delta':float((guided-boxes.detach()).abs().max())}
        return guided.detach(),output_logits.detach()
