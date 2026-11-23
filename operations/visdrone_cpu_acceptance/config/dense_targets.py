"""Keep the ordinary diffusion bank fixed-size while retaining every detection target."""
import torch

from projects.diffu_dino.modeling import DIFFUSION_DINO
from projects.diffu_dino.modeling.dino_diffu_det_noise import box_xyxy_to_cxcywh


class VisDroneDetector(DIFFUSION_DINO):
    def prepare_targets(self, instances):
        targets, diffusion = [], []
        for sample in instances:
            count = len(sample)
            if count < self.num_queries - 1:
                # Use the historical path, including its RNG sequence, for ordinary images.
                normal, padded = super().prepare_targets([sample])
                targets.extend(normal)
                diffusion.extend(padded)
                continue

            height, width = sample.image_size
            xyxy = sample.gt_boxes.tensor
            boxes = box_xyxy_to_cxcywh(xyxy / xyxy.new_tensor([width, height, width, height]))
            labels = sample.gt_classes
            if count > self.num_queries:
                # Both base samplers return indices into the leading GT rows.
                # Permute GT together so the selected bank is unbiased and those
                # indices still address the correct full-set detection/DN targets.
                order = torch.randperm(count, device=boxes.device)
                labels, boxes = labels[order], boxes[order]
            targets.append({"labels": labels, "boxes": boxes,
                            "image_size": boxes.new_tensor([height, width])})
            bank_labels = labels[:self.num_queries]
            bank_boxes = boxes[:self.num_queries]
            if count == self.num_queries - 1:
                pad_box = boxes.new_full((1, 4), 0.5)
                if self.noisy_gt:
                    pad_box = torch.randn_like(pad_box).sigmoid()
                bank_boxes = torch.cat((bank_boxes, pad_box))
                bank_labels = torch.cat((bank_labels, labels.new_full((1,), self.num_classes)))
            diffusion.append({"labels": bank_labels, "boxes": bank_boxes})
        return targets, diffusion
