import math
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch


class GaborConv2d(nn.Module):
    """
    Learnable Gabor Convolution (LGC) layer.
    Restriction: input channel must be 1.
    """
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_ratio = init_ratio if init_ratio > 0 else 1.0

        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False
        )
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)
        self.kernel = 0

    def genGaborBank(self, kernel_size, channel_in, channel_out, sigma, gamma, theta, f, psi):
        """Generate learnable Gabor filter bank."""
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax
        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).float()
        x_0 = torch.arange(xmin, xmax + 1).float()

        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1)
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize)
        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8 * sigma.view(-1, 1, 1, 1) ** 2)) \
            * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))
        gb = gb - gb.mean(dim=[2, 3], keepdim=True)
        return gb

    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in, self.channel_out,
                                   self.sigma, self.gamma, self.theta, self.f, self.psi)
        self.kernel = kernel
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)
        return out


class SELayer(nn.Module):
    """Squeeze‑and‑Excitation attention module for channel‑wise feature recalibration."""
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CompetitiveBlock(nn.Module):
    """
    Competitive Block (CB) = LGC + soft‑argmax + PPU downstream conv head.
    If if_second=True, apply two‑branch Gabor and concatenate competitive maps.
    """
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 init_ratio=1, o1=32, o2=24, if_second=False):
        super(CompetitiveBlock, self).__init__()
        assert channel_in == 1
        self.if_second = if_second
        self.channel_in = 1
        self.n_competitor = n_competitor
        self.init_ratio = init_ratio
        self.weight_chan = 0.5
        self.weight_spa = (1-self.weight_chan)/2

        self.gabor_conv2d_1 = GaborConv2d(
            channel_in=1, channel_out=n_competitor,
            kernel_size=ksize, stride=stride, padding=padding, init_ratio=init_ratio
        )
        self.gabor_conv2d_2 = GaborConv2d(
            channel_in=n_competitor, channel_out=n_competitor,
            kernel_size=ksize, stride=1, padding=ksize//2, init_ratio=init_ratio
        )
        self.argmax = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)

        self.se1 = SELayer(n_competitor)
        self.se2 = SELayer(n_competitor)

        self.conv1 = nn.Conv2d(n_competitor, o2, 5, 2, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(o1, o2, 1, 1, 0)
        self.conv1_2 = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.conv2_2 = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.gabor_conv2d_1(x)
        x1_1 = self.argmax(x)
        x1_2 = self.argmax_x(x)
        x1_3 = self.argmax_y(x)
        x_1 = self.weight_chan * x1_1 + self.weight_spa * (x1_2 + x1_3)
        x_1 = self.se1(x_1)
        if self.if_second:
            x = self.gabor_conv2d_2(x)
            x2_1 = self.argmax(x)
            x2_2 = self.argmax_x(x)
            x2_3 = self.argmax_y(x)
            x_2 = self.weight_chan * x2_1 + self.weight_spa * (x2_2 + x2_3)
            x_2 = self.se2(x_2)
            # x = torch.cat((x_1, x_2), dim=1)

            x = self.conv1_2(x_2)
            x = self.maxpool(x)
            x = self.conv2_2(x)
            return x
        x_1 = self.conv1(x_1)
        x_1 = self.maxpool(x_1)
        # x = self.conv2(x_1)
        return x_1


# helpers
"""Convert input argument to tuple.
If input is already a tuple, return it directly.
If input is scalar, convert to a tuple with two identical elements."""
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class FeedForward(nn.Module):
    """
    MLP feed‑forward module.
    Args:
        dim: feature dimension of input tensor
        dim_for_mlp: hidden dimension inside MLP block
        dropout: dropout probability
    """
    def __init__(self, dim, dim_for_mlp, dropout=0.1):
        super(FeedForward, self).__init__()

        self.dim = dim
        self.dim_for_mlp = dim_for_mlp
        self.dropout_rate = dropout

        self.net = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim_for_mlp),
            nn.GELU(),  # Non‑linear GELU activation
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dim_for_mlp, self.dim),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Multi‑head self‑attention module.
    Args:
        heads: number of attention heads
        dim: input feature dimension
        dim_for_head: feature dimension per single head
        dropout: dropout probability
    """
    def __init__(self, dim, heads, dim_for_head, dropout=0.1):
        super(Attention, self).__init__()

        self.dim = dim
        self.heads = heads
        self.dim_for_head = dim_for_head
        self.dropout_rate = dropout

        self.dim_for_head_inner = self.dim_for_head * self.heads

        self.project_out = not (self.heads == 1 and self.dim_for_head == self.dim)

        self.scale = self.dim_for_head ** -0.5

        self.norm = nn.LayerNorm(self.dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.to_qkv = nn.Linear(self.dim, self.dim_for_head_inner * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.dim_for_head_inner, self.dim),
            nn.Dropout(self.dropout_rate)
        ) if self.project_out else nn.Identity()

    def forward(self, x):
        # torch.Size([batch_size, num_patches + 1, dim])
        x = self.norm(x)  # Normalize input features

        # torch.Size([batch_size, num_patches + 1, dim_for_head_inner * 3])
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # torch.Size([batch_size, heads, num_patches + 1, dim_for_head_inner // heads = dim_for_head])
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # torch.Size([batch_size, heads, num_patches + 1, num_patches + 1])
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # torch.Size([batch_size, heads, num_patches + 1, dim_for_head_inner // heads = dim_for_head])
        out = torch.matmul(attn, v)

        # torch.Size([batch_size, num_patches + 1, heads * dim_for_head_inner // heads])
        out = rearrange(out, 'b h n d -> b n (h d)')

        # torch.Size([batch_size, num_patches + 1, dim])
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer encoder block stack.
    Args:
        depth: number of transformer encoder layers
        heads: number of multi‑attention heads
        dim: input feature dimension
        dim_for_head: feature dimension for each attention head
        dim_for_mlp: hidden dimension for MLP inside each block
        dropout: dropout probability
    """
    def __init__(self, dim, depth, heads, dim_for_head, dim_for_mlp, dropout=0.1):
        super(Transformer, self).__init__()

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_for_head = dim_for_head
        self.dim_for_mlp = dim_for_mlp
        self.dropout_rate = dropout

        self.norm = nn.LayerNorm(self.dim)

        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                Attention(heads=self.heads, dim=self.dim, dim_for_head=self.dim_for_head, dropout=self.dropout_rate),
                FeedForward(dim=self.dim, dim_for_mlp=self.dim_for_mlp, dropout=self.dropout_rate)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # torch.Size([batch_size, num_patches + 1, dim])
            x = attn(x) + x
            # torch.Size([batch_size, num_patches + 1, dim])
            x = ff(x) + x

        # torch.Size([batch_size, num_patches + 1, dim])
        return self.norm(x)


class ViT(nn.Module):
    """
    Vision Transformer encoder.
    Args:
        image_size: spatial resolution of input feature map
        patch_size: patch partition size; image_size must be divisible by patch_size
        channels: number of input feature channels
        num_classes: classification category count (unused here)
        depth: number of transformer encoder layers
        heads: number of multi‑attention heads
        dim: embedding dimension
        dim_for_head: feature dimension per attention head
        dim_for_mlp: hidden dimension of feed‑forward MLP
        pool: pooling strategy, choose from {'cls', 'mean'}
        dropout: dropout probability
        emb_dropout: dropout probability for patch embedding
    """
    def __init__(self, *, image_size, patch_size, channels, num_classes, depth, heads,
                 dim, dim_for_head, dim_for_mlp, pool='cls', dropout=0.1, emb_dropout=0.1):
        super(ViT, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.num_classes = num_classes
        self.depth = depth
        self.heads = heads
        self.dim = dim
        self.dim_for_head = dim_for_head
        self.dim_for_mlp = dim_for_mlp
        self.pool = pool
        self.dropout_rate = dropout
        self.emb_dropout_rate = emb_dropout

        self.image_height, self.image_width = pair(self.image_size)
        self.patch_height, self.patch_width = pair(self.patch_size)

        assert (self.image_height % self.patch_height == 0
                and self.image_width % self.patch_width == 0), ('Image dimensions must be '
                                                                'divisible by the patch size.')

        self.num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)
        self.patch_dim = self.channels * self.patch_height * self.patch_width

        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.dim),
            nn.LayerNorm(self.dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(self.emb_dropout_rate)

        self.transformer = Transformer(depth=self.depth, heads=self.heads, dim=self.dim,
                                       dim_for_head=self.dim_for_head, dim_for_mlp=self.dim_for_mlp,
                                       dropout=self.dropout_rate)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, feature_tensor):  # feature_tensor: torch.Size([batch_size, filter_num, 30, 30])
        # Patch embedding stage
        # torch.Size([batch_size, num_patches, dim])
        feature_tensor_for_vit = self.to_patch_embedding(feature_tensor)
        # torch.Size([batch_size, num_patches, dim])
        b, n, _ = feature_tensor_for_vit.shape
        # torch.Size([batch_size, num_patches, dim])
        # Do not use cls token
        feature_tensor_for_vit += self.pos_embedding[:, :n]
        feature_tensor_for_vit = self.dropout(feature_tensor_for_vit)

        # Transformer encoder forward
        feature_tensor_for_vit = self.transformer(feature_tensor_for_vit)

        # Return full patch token sequence without pooling
        return feature_tensor_for_vit


class ArcMarginProduct(nn.Module):
    """
    ArcFace angular margin classification head.
    During training: requires ground‑truth label and computes angular‑margin logits.
    During inference: outputs scaled cosine similarity without margin penalty.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if self.training:
            assert label is not None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            if not self.easy_margin:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            else:
                phi = torch.where(cosine > 0, phi, cosine)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            assert label is None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            output = self.s * cosine
        return output


class CTSNet(nn.Module):
    """
    Multi‑branch competitive feature extractor integrated with ViT branch for palmprint recognition.
    Reference: https://ieeexplore.ieee.org/document/9512475
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.cb1 = CompetitiveBlock(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=0, init_ratio=1)
        self.cb2 = CompetitiveBlock(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=0, init_ratio=0.5, if_second=True)
        self.cb3 = CompetitiveBlock(channel_in=1, n_competitor=18, ksize=7, stride=3, padding=0, init_ratio=0.25, if_second=True)

        self.vit1 = ViT(
            image_size=7, patch_size=1, channels=24, num_classes=num_classes,
            depth=2, heads=16, dim=512, dim_for_head=64, dim_for_mlp=384,
            dropout=0.1, emb_dropout=0.1
        )

        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(6936, 1024)
        self.fc3 = nn.Linear(7776, 1024)
        self.weight = 0.8

        self.drop = nn.Dropout(p=0.25)
        self.arclayer = ArcMarginProduct(1024, num_classes, s=30, m=0.5, easy_margin=False)

    def _forward_shared(self, x):
        """Shared feature extraction pipeline. Returns raw fused feature vector without ArcMargin head."""
        x1_1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x3 = x3.view(x3.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)

        # Feed feature map from CB1 into ViT encoder
        x1_1 = self.vit1(x1_1)
        x1_1 = x1_1.view(x1_1.shape[0], -1)

        x1 = x1_1
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)

        # Three branches output shape: [B,1024]
        stacked = torch.stack([x1, x2, x3], dim=1)  # [B,3,1024]
        score = stacked.mean(dim=-1)  # [B,3] sample‑adaptive scale‑competition weight
        attn = F.softmax(score, dim=-1)  # [B,3]
        attn = attn.unsqueeze(-1)  # [B,3,1]
        feat_concat = (stacked * attn).sum(dim=1)

        # feat_concat = self.weight * (x2*0.8+x3*0.2) + (1-self.weight) * x1

        return feat_concat

    def forward(self, x, labels=None):
        """Training‑mode forward pass: return ArcFace logits and raw feature vector."""
        feat = self._forward_shared(x)
        feat = self.drop(feat)
        logits = self.arclayer(feat, labels)
        return logits, feat

    def get_embedding(self, x):
        """Evaluation‑mode interface: return L2‑normalized embedding for feature matching."""
        with torch.no_grad():
            feat = self._forward_shared(x)
            feat = F.normalize(feat, p=2, dim=1)
        return feat


if __name__ == "__main__":
    import torch
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Build model instance
    model = CTSNet(num_classes=600).to(DEVICE)
    model.train()

    # Dummy input: batch=4, single‑channel grayscale palmprint image 128×128
    batch_size = 4
    dummy_img = torch.randn(batch_size, 1, 128, 128).to(DEVICE)
    dummy_labels = torch.randint(0, 600, (batch_size,)).to(DEVICE)

    # 1. Training forward (with label, output ArcFace logits + raw feature)
    with torch.no_grad():
        logits, feat_raw = model(dummy_img, dummy_labels)
    print("="*60)
    print(f"Train branch logits shape: {logits.shape}")
    print(f"Raw feature feat_raw shape: {feat_raw.shape}")

    # 2. Evaluation interface get_embedding (used for validation/test phase)
    emb_norm = model.get_embedding(dummy_img)
    print(f"Evaluation normalized embedding shape: {emb_norm.shape}")
    print(f"L2 norm check (sample 0): {torch.norm(emb_norm[0], p=2).item():.4f}")

    # 3. Single image test (simulate single sample loaded from DataLoader)
    single_img = torch.randn(1, 1, 128, 128).to(DEVICE)
    single_emb = model.get_embedding(single_img)
    print("-"*60)
    print(f"Single image normalized feature shape: {single_emb.shape}")

    # 4. Train mode forward (dropout activated)
    model.train()
    logits_train, feat_train = model(dummy_img, dummy_labels)
    print("-"*60)
    print(f"Train mode logits shape: {logits_train.shape}")
    print("8")
    print("\n✅ Model forward pass test completed.")
