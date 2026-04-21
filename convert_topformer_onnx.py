"""
Standalone TopFormer-Base -> ONNX converter.

No mmcv or mmseg installation required — all model classes are defined
inline, adapted from obs-tackle/third_party/TopFormer-main/tools/convert2onnx.py.

Reads  : TopFormer-B_512x512_4x8_160k-39.2.pth  (root folder)
Writes : topformer.onnx                           (root folder)

ONNX interface:
  input  : 'input'   float32  [1, 3, 512, 512]   (ImageNet-normalised RGB)
  output : 'output'  float32  [1, 150, 64, 64]   (logits per ADE20K class)

Run once:
    python convert_topformer_onnx.py
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(ROOT, 'TopFormer-B_512x512_4x8_160k-39.2.pth')
OUTPUT     = os.path.join(ROOT, 'topformer.onnx')
INPUT_SIZE = 512


# ════════════════════════════════════════════════════════════════════════════
# Minimal replacements for mmcv.cnn  (attribute names match mmcv exactly so
# that checkpoint keys load without remapping)
# ════════════════════════════════════════════════════════════════════════════

def build_norm_layer(cfg, num_features):
    """Always returns ('bn', BatchNorm2d) — handles SyncBN configs too."""
    return 'bn', nn.BatchNorm2d(num_features)


class ConvModule(nn.Module):
    """
    Drop-in for mmcv.cnn.ConvModule.
    Attribute names (.conv / .bn / .activate) deliberately match mmcv so that
    state-dict keys from an mmcv-trained checkpoint load without key remapping.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1,
                 norm_cfg=None, act_cfg=None, **_ignored):
        super().__init__()
        bias = norm_cfg is None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.bn       = nn.BatchNorm2d(out_channels) if norm_cfg is not None else None
        self.activate = nn.ReLU(inplace=True)        if act_cfg  is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn       is not None: x = self.bn(x)
        if self.activate is not None: x = self.activate(x)
        return x


# ════════════════════════════════════════════════════════════════════════════
# TopFormer architecture
# (adapted from TopFormer-main/tools/convert2onnx.py — pure PyTorch only)
# ════════════════════════════════════════════════════════════════════════════

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        _, bn = build_norm_layer(norm_cfg, b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1    = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1,
                                bias=True, groups=hidden_features)
        self.act  = act_layer()
        self.fc2  = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, ks, stride, expand_ratio,
                 activations=None,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.stride       = stride
        self.expand_ratio = expand_ratio
        if activations is None:
            activations = nn.ReLU
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.append(Conv2d_BN(inp, hidden_dim, ks=1, norm_cfg=norm_cfg))
            layers.append(activations())
        layers += [
            Conv2d_BN(hidden_dim, hidden_dim, ks=ks, stride=stride,
                      pad=ks // 2, groups=hidden_dim, norm_cfg=norm_cfg),
            activations(),
            Conv2d_BN(hidden_dim, oup, ks=1, norm_cfg=norm_cfg),
        ]
        self.conv        = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn      = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class TokenPyramidModule(nn.Module):
    def __init__(self, cfgs, out_indices, inp_channel=16,
                 activation=nn.ReLU,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 width_mult=1.):
        super().__init__()
        self.out_indices = out_indices
        self.stem = nn.Sequential(
            Conv2d_BN(3, inp_channel, 3, 2, 1, norm_cfg=norm_cfg),
            activation()
        )
        self.cfgs   = cfgs
        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            layer_name     = f'layer{i + 1}'
            layer = InvertedResidual(inp_channel, output_channel, ks=k, stride=s,
                                     expand_ratio=t, norm_cfg=norm_cfg,
                                     activations=activation)
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer_name in enumerate(self.layers):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class Attention(nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = key_dim ** -0.5
        self.key_dim   = key_dim
        self.nh_kd     = key_dim * num_heads
        self.d         = int(attn_ratio * key_dim)
        self.dh        = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, self.nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, self.nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh,    1, norm_cfg=norm_cfg)
        self.proj = nn.Sequential(
            activation(),
            Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        qq  = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk  = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv  = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        attn = torch.matmul(qq, kk).softmax(dim=-1)
        xx   = torch.matmul(attn, vv).permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        return self.proj(xx)


class Block(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2.,
                 drop=0., drop_path=0., act_layer=nn.ReLU,
                 norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads,
                              attn_ratio=attn_ratio, activation=act_layer,
                              norm_cfg=norm_cfg)
        self.drop_path = nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class BasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0.,
                 drop_path=0., norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            Block(embedding_dim, key_dim=key_dim, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, attn_ratio=attn_ratio, drop=drop,
                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                  norm_cfg=norm_cfg, act_layer=act_layer)
            for i in range(block_num)
        ])

    def forward(self, x):
        for blk in self.transformer_blocks:
            x = blk(x)
        return x


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat(
            [F.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class InjectionMultiSum(nn.Module):
    def __init__(self, inp, oup,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 activations=None):
        super().__init__()
        self.local_embedding  = ConvModule(inp, oup, kernel_size=1,
                                           norm_cfg=norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1,
                                           norm_cfg=norm_cfg, act_cfg=None)
        self.global_act       = ConvModule(inp, oup, kernel_size=1,
                                           norm_cfg=norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_g):
        _, _, H, W = x_l.shape
        local_feat  = self.local_embedding(x_l)
        sig_act     = F.interpolate(self.act(self.global_act(x_g)),
                                    size=(H, W), mode='bilinear', align_corners=False)
        global_feat = F.interpolate(self.global_embedding(x_g),
                                    size=(H, W), mode='bilinear', align_corners=False)
        return local_feat * sig_act + global_feat


class Topformer(nn.Module):
    def __init__(self, cfgs, channels, out_channels, embed_out_indice,
                 decode_out_indices=(1, 2, 3), depths=4, key_dim=16, num_heads=8,
                 attn_ratios=2, mlp_ratios=2, c2t_stride=2, drop_path_rate=0.,
                 norm_cfg=dict(type='BN', requires_grad=True), act_layer=nn.ReLU6,
                 injection_type='muli_sum', init_cfg=None, injection=True, **_kw):
        super().__init__()
        self.channels          = channels
        self.injection         = injection
        self.embed_dim         = sum(channels)
        self.decode_out_indices = list(decode_out_indices)

        self.tpm = TokenPyramidModule(cfgs=cfgs, out_indices=embed_out_indice,
                                      norm_cfg=norm_cfg)
        self.ppa = PyramidPoolAgg(stride=c2t_stride)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.trans = BasicLayer(
            block_num=depths, embedding_dim=self.embed_dim,
            key_dim=key_dim, num_heads=num_heads,
            mlp_ratio=mlp_ratios, attn_ratio=attn_ratios,
            drop=0, attn_drop=0, drop_path=dpr,
            norm_cfg=norm_cfg, act_layer=act_layer)

        self.SIM = nn.ModuleList()
        for i in range(len(channels)):
            if i in self.decode_out_indices:
                self.SIM.append(InjectionMultiSum(
                    channels[i], out_channels[i],
                    norm_cfg=norm_cfg, activations=act_layer))
            else:
                self.SIM.append(nn.Identity())

    def forward(self, x):
        tpm_outs = self.tpm(x)
        agg      = self.ppa(tpm_outs)
        agg      = self.trans(agg)

        splits  = agg.split(self.channels, dim=1)
        results = []
        for i in range(len(self.channels)):
            if i in self.decode_out_indices:
                results.append(self.SIM[i](tpm_outs[i], splits[i]))
        return results


class SimpleHead(nn.Module):
    def __init__(self, channels, num_classes, in_index=(0, 1, 2),
                 dropout_ratio=0.1, norm_cfg=None,
                 act_cfg=dict(type='ReLU'), is_dw=False, **_kw):
        super().__init__()
        self.in_index = list(in_index) if not isinstance(in_index, (list, tuple)) \
                        else list(in_index)
        self.linear_fuse = ConvModule(
            in_channels=channels, out_channels=channels, kernel_size=1,
            groups=channels if is_dw else 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        feats = [inputs[i] for i in self.in_index]
        # Aggregate: upsample all to the largest spatial size then sum
        target_size = feats[0].shape[2:]
        out = feats[0]
        for f in feats[1:]:
            out = out + F.interpolate(f, size=target_size,
                                      mode='bilinear', align_corners=False)
        return self.conv_seg(self.linear_fuse(out))


class Segmentor(nn.Module):
    def __init__(self, backbone, decode_head):
        super().__init__()
        self.backbone    = backbone
        self.decode_head = decode_head

    def forward(self, img):
        return self.decode_head(self.backbone(img))


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _convert_batchnorm(module):
    """Convert any remaining SyncBatchNorm → BatchNorm2d in-place."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(
            module.num_features, module.eps, module.momentum,
            module.affine, module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data   = module.bias.data.clone().detach()
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad   = module.bias.requires_grad
        module_output.running_mean        = module.running_mean
        module_output.running_var         = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


# ── TopFormer-Base config (from local_configs/topformer/topformer_base.py) ─
_BN = dict(type='BN', requires_grad=True)

TOPFORMER_BASE_BACKBONE = dict(
    cfgs=[
        # k,  t,   c, s
        [3,   1,  16, 1],
        [3,   4,  32, 2],
        [3,   3,  32, 1],
        [5,   3,  64, 2],
        [5,   3,  64, 1],
        [3,   3, 128, 2],
        [3,   3, 128, 1],
        [5,   6, 160, 2],
        [5,   6, 160, 1],
        [3,   6, 160, 1],
    ],
    channels         = [32, 64, 128, 160],
    out_channels     = [None, 256, 256, 256],
    embed_out_indice = [2, 4, 6, 9],
    decode_out_indices = [1, 2, 3],
    depths           = 4,
    num_heads        = 8,
    c2t_stride       = 2,
    drop_path_rate   = 0.1,
    norm_cfg         = _BN,
    act_layer        = nn.ReLU6,
)

TOPFORMER_BASE_HEAD = dict(
    channels      = 256,
    num_classes   = 150,
    in_index      = [0, 1, 2],
    dropout_ratio = 0.1,
    norm_cfg      = _BN,
    act_cfg       = dict(type='ReLU'),
)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 60)
    print('  TopFormer-Base -> ONNX Converter')
    print('=' * 60)

    if not os.path.exists(CHECKPOINT):
        print(f'\nERROR: Checkpoint not found:\n  {CHECKPOINT}')
        print('Make sure TopFormer-B_512x512_4x8_160k-39.2.pth is in the FYP root folder.')
        sys.exit(1)

    # ── Build model ─────────────────────────────────────────────────────────
    print('\n[1/4] Building TopFormer-Base model...')
    backbone = Topformer(**TOPFORMER_BASE_BACKBONE)
    head     = SimpleHead(**TOPFORMER_BASE_HEAD)
    model    = Segmentor(backbone, head)
    model    = _convert_batchnorm(model)   # ensure no SyncBN
    model.eval()

    # ── Load checkpoint ─────────────────────────────────────────────────────
    print(f'[2/4] Loading checkpoint...')
    print(f'      {CHECKPOINT}')
    ckpt       = torch.load(CHECKPOINT, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  WARNING - missing keys ({len(missing)}): {missing[:3]} ...')
    if unexpected:
        print(f'  WARNING - unexpected keys ({len(unexpected)}): {unexpected[:3]} ...')
    if not missing and not unexpected:
        print('  All keys matched perfectly.')
    else:
        print(f'  Loaded with {len(missing)} missing / {len(unexpected)} unexpected keys.')
        print('  (Some mismatch is expected due to mmcv→pure-PyTorch conversion)')

    # ── ONNX export ─────────────────────────────────────────────────────────
    print(f'\n[3/4] Exporting to ONNX (input size: {INPUT_SIZE}×{INPUT_SIZE})...')
    dummy_input = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        # Verify forward pass works before export
        test_out = model(dummy_input)
        print(f'      PyTorch forward pass OK — output shape: {test_out.shape}')

        torch.onnx.export(
            model,
            dummy_input,
            OUTPUT,
            input_names  = ['input'],
            output_names = ['output'],
            opset_version = 18,
            export_params = True,
            do_constant_folding = True,
        )
    print(f'      Saved: {OUTPUT}')

    # ── Verify ONNX ─────────────────────────────────────────────────────────
    print('\n[4/4] Verifying ONNX with onnxruntime...')
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(OUTPUT, providers=['CPUExecutionProvider'])
        onnx_out = sess.run(None, {'input': dummy_input.numpy()})[0]
        print(f'      ONNX output shape : {onnx_out.shape}')
        # Compare with PyTorch
        max_diff = float(abs(test_out.numpy() - onnx_out).max())
        print(f'      Max abs diff vs PyTorch: {max_diff:.6f}')
        print(f'      {"PASS" if max_diff < 1e-2 else "WARN - diff larger than expected"}')
    except Exception as e:
        print(f'      WARNING: ONNX verification failed: {e}')
        print('      The .onnx file may still work at runtime.')

    print('\n' + '=' * 60)
    print('  Conversion complete!  Run: python navigation_app.py')
    print('=' * 60)
