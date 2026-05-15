# vae-gan-tfg

Repository containing the code used for the simulations and experiments developed in the Bachelor's Thesis in Mathematics:

> **“Analysis and Implementation of Image Generation Models using Variational Autoencoders and Generative Adversarial Networks”**

This project includes several experiments related to Autoencoders, Convolutional Autoencoders (CAE), Variational Autoencoders (VAE), $\beta$-VAE, DCGANs, and hybrid VAE-GAN architectures.

The repository is organized according to the chapters and sections of the thesis manuscript.

---

# Project Structure

```text
vae-gan-tfg/
│
├── Chapter1/
│   └── Logic_Gates/
│       └── 0_graphics.py
│
├── Chapter2/
│   │
│   ├── 2_1_AE/
│   │   │
│   │   ├── 2_1_1_sigmoid/
│   │   │   └── 1_sigmoid_AE.py
│   │   │
│   │   ├── 2_1_2_tanh/
│   │   │   └── 2_tanh_AE.py
│   │   │
│   │   └── 2_1_3_relu/
│   │       ├── 3_ReLU_AE.py
│   │       ├── 4_sigmoid_AE.py
│   │       └── 5_tanh_AE.py
│   │
│   ├── 2_2_CAE/
│   │   └── 6_Overfitting_CAE.py
│   │
│   └── 2_3_VAE/
│       │
│       ├── 2_3_1_construc_reconstruc/
│       │   └── 7_reconstruction_regularization.py
│       │
│       ├── 2_3_2_esp_lat/
│       │   └── 8_esp_lat_norm_VAE.py
│       │
│       └── 2_3_3_b-VAE/
│           └── 9_overfitting_bvae_mse_kl_relu_tanh.py
│
├── Chapter3/
│   └── DCGAN/
│       └── 10_DCGAN.py
│
├── Chapter4/
│   └── 11_hybrid.py
│
└── .gitignore
```

---

# Requirements

Before running the code, it is recommended to:

- Install **Python 3.10** or later.
- Create a virtual environment:

```bash
python -m venv my_environment
```

- Activate the virtual environment on Windows:

```bash
my_environment\Scripts\activate
```

- Install the required libraries:

```bash
pip install numpy matplotlib tensorflow tensorboard scipy
```

---

# Project Objectives

The main goal of this project is to study, implement, and compare different image generation models, with special focus on:

- Classical Autoencoders.
- Variational Autoencoders (VAE).
- $\beta$-VAE models.
- Generative Adversarial Networks (GAN).
- Hybrid VAE-GAN architectures.

Additionally, the project analyzes:
- latent space representations,
- image reconstruction quality,
- regularization mechanisms,
- and generative capabilities of each architecture.

---

# Author

**Carlos Cruz Martínez**  
Double Degree in Telecommunications Engineering and Mathematics  
University of Málaga

