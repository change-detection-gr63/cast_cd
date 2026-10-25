# CAST-CD architecture audit

Source: `/Users/sagar/Downloads/castcd_all_in_one (2).ipynb`.

Verified against notebook code cells 3 (boundary extraction), 5 (model), 7 (losses/evaluation), and 9 (training). This is a source-code and diagram audit; the notebook was not trained and no PyTorch runtime shape test was executed.

## Correction to the highlighted gap

The previous panel placed an isolated loss formula next to a decoder block. A direct decoder-feature → total-loss arrow would be incorrect. The correct path is decoder outputs → prediction heads → individual losses, with corresponding targets → weighted total. Decoder detail and the complete training graph now occupy separate panels. Matching tensor symbols explicitly link panels (a) and (e).

## Verified architecture

- Shared EfficientNet-B4 stem and first three feature stages: 48 channels at H/2; 24 at H/2; 32 at H/4; 56 at H/8.
- Full-resolution fusion: both RGB images → concatenate → Conv 6→3 → PReLU → InstanceNorm.
- GDAF at 24 and 32 channels; CTA at 56 channels, with four heads of 14 channels each.
- Decoder configurations, extracted from the notebook AST: `UpConcat(2,56,32,64)`, `UpConcat(2,64,24,64)`, `UpConcat(2,64,3,32)`.
- Decoder: bilinear ×2, align_corners=True → crop to skip extent if needed → concatenate skip → depthwise 3×3 → PReLU → IN → 1×1 → PReLU → IN.
- Change head: 32→16→8→1; PReLU after every 1×1 convolution.
- DS₂ at H/2 and DS₃ at H/4: 64→16→1; PReLU after every 1×1 convolution.
- Boundary head: 3×3 32→16 → IN → PReLU → 1×1 16→1; no final activation.
- Inference: change logits → sigmoid → threshold strictly greater than 0.5.

Spatial fractions assume H and W divisible by 8, as stated in the diagram.

## Verified detailed dependencies

GDAF: both features feed absolute difference, the gate, and the temporal mean. The attended difference feeds both the gate and the blend. Gate, attended difference, and temporal mean feed the blend. Projection order is depthwise 3×3 → IN → PReLU → 1×1.

CTA: distinct Wq/Wk/Wv convolutions are reused across both temporal directions. Q and K normalize over the flattened spatial dimension. Attention is 14×14 per head; the learned temperature multiplies QK transpose before softmax. Both attention outputs feed the 112→56 projection. No residual or additional square-root scaling is depicted.

## Verified training graph

| Input | Target | Loss | Weight |
|---|---|---|---:|
| Change logits z | Original mask Y | BCEWithLogits | 1 |
| sigmoid(z) | Original mask Y | Focal Tversky | 0.5 |
| Boundary logits z_b | Morphological gradient of Y, 3×3 | BCEWithLogits | 0.3 |
| DS₂ logits z₂ | Nearest-resized Y at H/2 | BCEWithLogits | 0.4 |
| DS₃ logits z₃ | Nearest-resized Y at H/4 | BCEWithLogits | 0.2 |
| Original deepest features E₁,E₂ before CTA | Nearest-resized Y at H/8 | Contrastive alignment | 0.1 |

Focal Tversky: alpha=beta=0.5, gamma=0.75, smooth=1. CFA: normalize along channels; calculate L2 distance d; average `(1−Y₈)d² + Y₈ ReLU(2−d)²`. All six losses connect to the weighted sum and then the scalar total.

Boundary outputs do not refine the binary mask. DS predictions do not feed decoder stages. The boundary head executes in evaluation but its output is discarded; DS heads execute only in training. Thresholded masks do not enter any training loss.

## Verification performed

- Two independent audits compared the revised full architecture and training graph with the notebook source.
- XML parses successfully; cell IDs are unique.
- All 78 connectors attach to their intended source and destination shapes.
- All six individual losses have incoming prediction/features and target connections and outgoing weighted-sum connections.
- Preview inspected for clipping, glyph availability, and misleading connections.
- The preview and XML use the same layout specification. The XML was not imported and rendered in the draw.io editor during this audit.
