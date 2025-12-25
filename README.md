# Lab 4 – Deep Generative Models with PyTorch

## Objective
This laboratory work aims to explore and compare several **deep generative learning models** using the **PyTorch** framework.  
The focus is on understanding the behavior, advantages, and limitations of three fundamental architectures:

- AutoEncoder (AE)
- Variational AutoEncoder (VAE)
- Generative Adversarial Network (GAN)

Each model is implemented, trained, and evaluated on an appropriate dataset.

---

## Technical Stack
- Python  
- PyTorch  
- Kaggle (GPU environment)  
- NumPy  
- Matplotlib  
- Git & GitHub  

---

## Datasets

### MNIST Dataset
- Handwritten digit images (0–9)
- Grayscale images of size 28 × 28
- Used for AutoEncoder and Variational AutoEncoder experiments

### Abstract Art Gallery Dataset
- Collection of abstract art images
- Images resized to 64 × 64
- Used for Generative Adversarial Network training

---

## Part 1: AutoEncoder (AE)

### Model Architecture
- Input dimension: 784 (flattened image)
- Encoder: Fully connected layers
- Latent space dimension: 32
- Decoder: Symmetric fully connected layers
- Activation functions:
  - ReLU for encoder
  - Sigmoid for decoder

### Training Setup
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Batch size: 128
- Number of epochs: 20

### Results
The AutoEncoder converges to a very low reconstruction loss, showing its effectiveness in reproducing MNIST digit images.

![Loss Comparison](loss comparison.jpeg)

### Discussion
The AE focuses purely on minimizing reconstruction error.  
While reconstruction quality is high, the learned latent space is not organized, which limits its generative capability.

---

## Part 2: Variational AutoEncoder (VAE)

### Model Description
The VAE introduces a probabilistic latent representation:
- The encoder outputs a mean (μ) and a log-variance (log σ²)
- Latent variables are sampled using the reparameterization trick
- The decoder reconstructs the input from the sampled latent vector

Latent dimension: 32

### Loss Function
The VAE loss combines two components:

\[
\mathcal{L} = \text{Reconstruction Loss} + \text{KL Divergence}
\]

- Reconstruction loss: Binary Cross-Entropy
- KL Divergence: Regularization toward a standard normal distribution

### Training Parameters
- Optimizer: Adam
- Batch size: 128
- Epochs: 20

### KL Divergence Evolution
![KL Divergence](kl_divergence.png)

The KL divergence increases during early training and then stabilizes, indicating proper regularization of the latent space.

### Latent Space Visualization
![VAE Latent Space](vae_latent.png)

### Analysis
The latent space shows clear clustering corresponding to digit classes.  
This confirms that the VAE learns a **structured and continuous latent space**, suitable for generation and interpolation.  
Reconstructed images are slightly blurrier than those produced by the AE due to the imposed probabilistic constraint.

---

##  Part 3: Generative Adversarial Network (GAN)

### Architecture
- Generator: Transposed convolutional neural network
- Discriminator: Convolutional neural network
- Noise vector dimension: 100

### Training Strategy
- Loss function: Binary Cross-Entropy
- Label smoothing applied (real labels = 0.9)
- Different learning rates for Generator and Discriminator
- Two Generator updates per Discriminator update

### Training Dynamics
Several techniques were applied to improve training stability and reduce:
- Discriminator dominance
- Mode collapse
- Gradient instability

### Generated Samples
![GAN Generated Images](gan_generated.png)

### Qualitative Evaluation
The generated images exhibit:
- Visual diversity
- Abstract textures and patterns
- No significant mode collapse

Although some noise remains, the results are consistent with a basic GAN trained for a limited number of epochs.

---

##  Model Comparison

| Model | Advantages | Limitations |
|------|-----------|-------------|
| AutoEncoder | Accurate reconstruction | No structured latent space |
| Variational AutoEncoder | Structured and generative latent space | Slightly blurry outputs |
| GAN | Diverse and realistic generation | Training instability |

---

##  Conclusion
This laboratory work provided hands-on experience with three major deep generative modeling approaches.

- The AutoEncoder excels at reconstruction but lacks generative flexibility.
- The Variational AutoEncoder introduces probabilistic regularization, enabling meaningful latent representations.
- The Generative Adversarial Network demonstrates strong generative potential, despite its challenging training process.

All models were successfully implemented, trained, and analyzed, fulfilling the objectives of the lab.

---

## Author
**Douae El Hannach**  
Master IT Security & Big Data
