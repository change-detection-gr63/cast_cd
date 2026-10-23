"""Cross-Attention Spatio-Temporal Change Detection model."""

import torch
import torchvision
from torch.nn import Conv2d, InstanceNorm2d, Module, ModuleList, PReLU, Sequential

from .blocks import CrossTemporalAttention, GDAF, PixelwiseLinear, UpConcat


def _get_backbone(bkbn_name, pretrained, output_layer_bkbn, freeze_backbone):
    try:
        entire_model = getattr(torchvision.models, bkbn_name)(
            weights="IMAGENET1K_V1" if pretrained else None).features
    except TypeError:
        entire_model = getattr(torchvision.models, bkbn_name)(pretrained=pretrained).features
    derived = ModuleList([])
    for name, layer in entire_model.named_children():
        derived.append(layer)
        if name == output_layer_bkbn: break
    if freeze_backbone:
        for p in derived.parameters(): p.requires_grad = False
    return derived


class CAST_CD(Module):
    """
    Cross-Attention Spatio-Temporal Change Detection (CAST-CD).
    Meaningful improvement over TinyCD featuring GDAF, CTA, Boundary Awareness, and Deep Supervision.
    """
    def __init__(self, bkbn_name="efficientnet_b4", pretrained=True,
                 output_layer_bkbn="3", freeze_backbone=False):
        super().__init__()
        self._backbone = _get_backbone(bkbn_name, pretrained, output_layer_bkbn, freeze_backbone)

        # Initial fusion
        self._first_fusion = Sequential(
            Conv2d(6, 3, 3, padding=1), PReLU(), InstanceNorm2d(3)
        )

        # GDAF replaces MixingMaskAttentionBlock for first 2 scales
        self._gdaf_modules = ModuleList([
            GDAF(24),  # Scale 1 (48 channel concat)
            GDAF(32)   # Scale 2 (64 channel concat)
        ])

        # Cross-Temporal Attention at deepest scale (56 channels)
        self._cta = CrossTemporalAttention(56, num_heads=4)

        # Up-samplers
        self._up = ModuleList([UpConcat(2, 56, 32, 64), UpConcat(2, 64, 24, 64), UpConcat(2, 64, 3, 32)])

        # Binary Change Head (No Sigmoid -> raw logits for BCEWithLogitsLoss)
        self.binary_head = PixelwiseLinear([32, 16, 8], [16, 8, 1], None)

        # Boundary Refinement Head
        self.boundary_head = Sequential(
            Conv2d(32, 16, 3, padding=1),
            InstanceNorm2d(16), PReLU(),
            Conv2d(16, 1, 1)  # Binary boundary logits
        )

        # Deep Supervision Heads
        self.ds_head2 = PixelwiseLinear([64, 16], [16, 1], None)
        self.ds_head3 = PixelwiseLinear([64, 16], [16, 1], None)

    def forward(self, ref, test):
        # 1. Encode
        f1_feats, f2_feats = self._encode(ref, test)

        # 2. Fuse (GDAF and CTA)
        fused_features = [self._first_fusion(torch.cat([ref, test], dim=1))]
        for i in range(len(f1_feats)-1):
            fused = self._gdaf_modules[i](f1_feats[i], f2_feats[i])
            fused_features.append(fused)

        # Deepest level CTA
        fused_deep = self._cta(f1_feats[-1], f2_feats[-1])
        fused_features.append(fused_deep)

        # 3. Decode
        upping = fused_features[-1]

        upping_s3 = self._up[0](upping, fused_features[-2])
        upping_s2 = self._up[1](upping_s3, fused_features[-3])
        upping_s1 = self._up[2](upping_s2, fused_features[-4])

        # Outputs
        binary_logits = self.binary_head(upping_s1)
        boundary_logits = self.boundary_head(upping_s1)

        if self.training:
            # Return auxiliary predictions + deepest features for contrastive loss
            ds_logits2 = self.ds_head2(upping_s2)
            ds_logits3 = self.ds_head3(upping_s3)
            return {
                'binary': binary_logits,
                'boundary': boundary_logits,
                'ds2': ds_logits2,
                'ds3': ds_logits3,
                'f1_deep': f1_feats[-1],
                'f2_deep': f2_feats[-1]
            }
        else:
            return binary_logits

    def _encode(self, ref, test):
        f1_feats = []
        f2_feats = []
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)
            if num != 0:
                f1_feats.append(ref)
                f2_feats.append(test)
        return f1_feats, f2_feats
