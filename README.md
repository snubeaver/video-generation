# video-generation
Personal Archive for Video Generation Models


## Image Generation Models

### 1. VQGAN (Vector Quantized Generative Adversarial Network):
VQGAN combines Vector Quantization (VQ) with GANs to create a [two-stage approach](https://medium.com/geekculture/vq-gan-explained-4827599b7cf2):

Stage 1 - Learning Discrete Codebook:

Uses Vector Quantization to learn a discrete codebook of image tokens
The encoder E maps images x to latent vectors z = E(x)
The latent vectors are quantized to their nearest codebook entries ẑ = q(z)
A decoder G reconstructs the image from quantized vectors
Trained with a combination of reconstruction loss, perceptual loss (using VGG features), and adversarial loss

Key features:

Codebook size typically 1024 or 16384 entries
Maintains spatial information through convolutional architecture
Uses commitment loss to keep encoder outputs close to codebook vectors
Transformer can be trained on these discrete codes for generation


### 2. [Stable Diffusion](https://poloclub.github.io/diffusion-explainer/):
Built on Latent Diffusion Models (LDM), with key architectural choices:

Compression Phase:

Uses autoencoder to compress images to lower-dimensional latent space (typically 4x downsampling)
Works in latent space of size h/8 × w/8 × 4 instead of pixel space
VAE architecture with KL-divergence regularization

Diffusion Process:

U-Net backbone with cross-attention layers for conditioning
Uses noise scheduling and learned denoising
Classifier-free guidance for improved sample quality
Conditioning through cross-attention mechanisms
Incorporates transformer blocks for processing text embeddings

Text Conditioning:

Uses CLIP text encoder (frozen) for text embeddings
Cross-attention layers inject text information into U-Net


### 3. Imagen:
Google's text-to-image model with several innovative components:

Architecture:

Uses a cascade of diffusion models at increasing resolutions
Base model: 64×64 pixels
Super-resolution model 1: 64×64 → 256×256
Super-resolution model 2: 256×256 → 1024×1024

Key Features:

Uses T5-XXL text encoder (frozen) instead of CLIP
Dynamic thresholding for improved sample quality
Classifier-free guidance with high guidance scales (∼7.5)
Noise conditioning augmentation during training
U-Net backbone with efficient attention mechanisms

### Notable differences and trade-offs:

Training Data Requirements:


VQGAN: Can train on relatively smaller datasets
Stable Diffusion: Requires large datasets but less than pixel-space models
Imagen: Requires massive high-quality datasets


### Computational Efficiency:


VQGAN: Most efficient for inference
Stable Diffusion: Moderate efficiency due to latent space
Imagen: Most computationally intensive due to cascade


### Quality vs Speed:


VQGAN: Fastest but lower quality
Stable Diffusion: Good balance of quality and speed
Imagen: Highest quality but slowest


### Architecture Complexity:


VQGAN: Simplest architecture
Stable Diffusion: Moderate complexity
Imagen: Most complex with multiple stages
