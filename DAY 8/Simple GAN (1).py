"""
GENERATIVE AI COMPLETE DEMO
NVIDIA Internship Training

Topics:
1. Real MNIST Dataset
2. GAN Architecture
3. GAN Forward Pass
4. BERT Tokenization
5. BERT Embeddings
6. Cross Attention
7. Diffusion Scheduler
8. Forward Diffusion
9. Time Embeddings
10. Mini U-Net
11. Classifier-Free Guidance
12. Stable Diffusion Simulation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)

print("Using Device:", device)

if torch.cuda.is_available():
    print(
        "GPU Name:",
        torch.cuda.get_device_name(0)
    )


# =====================================================
# REAL DATASET
# =====================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

real_images, labels = next(iter(loader))

print("\nMNIST Batch Shape:")
print(real_images.shape)


# =====================================================
# GAN GENERATOR
# =====================================================

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(100, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):

        return self.model(z)


# =====================================================
# GAN DISCRIMINATOR
# =====================================================

class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(784, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.model(x)


G = Generator().to(device)
D = Discriminator().to(device)

noise = torch.randn(
    64,
    100,
    device=device
)

fake_images = G(noise)

scores = D(fake_images)

print("\nGenerated Images Shape:")
print(fake_images.shape)

print("Discriminator Scores Shape:")
print(scores.shape)


# =====================================================
# BERT TOKENIZATION
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

text = "A futuristic city with flying cars"

tokens = tokenizer(
    text,
    return_tensors="pt"
)

print("\nToken IDs:")
print(tokens["input_ids"])


# =====================================================
# BERT EMBEDDINGS
# =====================================================

bert = AutoModel.from_pretrained(
    "bert-base-uncased"
).to(device)

tokens = {
    k: v.to(device)
    for k, v in tokens.items()
}

with torch.no_grad():

    outputs = bert(**tokens)

text_embeddings = outputs.last_hidden_state

print("\nBERT Embedding Shape:")
print(text_embeddings.shape)


# =====================================================
# CROSS ATTENTION
# =====================================================

class CrossAttention(nn.Module):

    def __init__(
        self,
        dim=768
    ):

        super().__init__()

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(
        self,
        latent,
        text
    ):

        Q = self.q(latent)
        K = self.k(text)
        V = self.v(text)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            Q.shape[-1]
        )

        weights = F.softmax(
            scores,
            dim=-1
        )

        return torch.matmul(
            weights,
            V
        )


attention = CrossAttention().to(device)

latent_tokens = torch.randn(
    1,
    64,
    768,
    device=device
)

attention_output = attention(
    latent_tokens,
    text_embeddings
)

print("\nCross Attention Output Shape:")
print(attention_output.shape)


# =====================================================
# DIFFUSION SCHEDULER
# =====================================================

timesteps = 1000

betas = torch.linspace(
    1e-4,
    0.02,
    timesteps,
    device=device
)

alphas = 1 - betas

alpha_hat = torch.cumprod(
    alphas,
    dim=0
)

print("\nDiffusion Scheduler Created")


# =====================================================
# FORWARD DIFFUSION
# =====================================================

def forward_diffusion(
    x0,
    t
):

    noise = torch.randn_like(x0)

    sqrt_alpha_hat = torch.sqrt(
        alpha_hat[t]
    )[:, None, None, None]

    sqrt_one_minus = torch.sqrt(
        1 - alpha_hat[t]
    )[:, None, None, None]

    xt = (
        sqrt_alpha_hat * x0
        +
        sqrt_one_minus * noise
    )

    return xt, noise


sample = torch.randn(
    2,
    3,
    64,
    64,
    device=device
)

t = torch.randint(
    0,
    timesteps,
    (2,),
    device=device
)

xt, noise = forward_diffusion(
    sample,
    t
)

print("\nForward Diffusion Shape:")
print(xt.shape)


# =====================================================
# TIME EMBEDDING
# =====================================================

class TimeEmbedding(nn.Module):

    def __init__(
        self,
        dim
    ):

        super().__init__()

        self.dim = dim

    def forward(self, t):

        half = self.dim // 2

        emb = math.log(
            10000
        ) / (half - 1)

        emb = torch.exp(
            torch.arange(
                half,
                device=t.device
            ) * -emb
        )

        emb = t[:, None] * emb[None, :]

        return torch.cat(
            (
                torch.sin(emb),
                torch.cos(emb)
            ),
            dim=1
        )


time_embed = TimeEmbedding(128)

time_vector = time_embed(
    torch.tensor(
        [10, 100],
        device=device
    )
)

print("\nTime Embedding Shape:")
print(time_vector.shape)


# =====================================================
# MINI U-NET
# =====================================================

class MiniUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.down1 = nn.Conv2d(
            4,
            32,
            3,
            padding=1
        )

        self.down2 = nn.Conv2d(
            32,
            64,
            3,
            stride=2,
            padding=1
        )

        self.up = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.out = nn.Conv2d(
            32,
            4,
            1
        )

    def forward(self, x):

        x = F.relu(
            self.down1(x)
        )

        x = F.relu(
            self.down2(x)
        )

        x = F.relu(
            self.up(x)
        )

        return self.out(x)


unet = MiniUNet().to(device)

latent = torch.randn(
    1,
    4,
    64,
    64,
    device=device
)

pred_noise = unet(latent)

print("\nPredicted Noise Shape:")
print(pred_noise.shape)


# =====================================================
# CLASSIFIER FREE GUIDANCE
# =====================================================

eps_cond = torch.tensor(0.9)
eps_uncond = torch.tensor(0.3)

cfg_scale = 7.5

eps = (
    eps_uncond
    +
    cfg_scale *
    (
        eps_cond -
        eps_uncond
    )
)

print("\nCFG Output:")
print(eps.item())


# =====================================================
# STABLE DIFFUSION SIMULATION
# =====================================================

print("\nRunning Denoising Loop...")

latent = torch.randn(
    1,
    4,
    64,
    64,
    device=device
)

for step in range(20):

    predicted_noise = unet(latent)

    latent = latent - (
        0.05 * predicted_noise
    )

print("Final Latent Shape:")
print(latent.shape)


# =====================================================
# GPU MEMORY
# =====================================================

if torch.cuda.is_available():

    print(
        "\nAllocated Memory:",
        round(
            torch.cuda.memory_allocated() / 1e9,
            2
        ),
        "GB"
    )

print("\nDemo Completed Successfully")
