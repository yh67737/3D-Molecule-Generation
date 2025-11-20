# 🧬 AI-Aided Drug Design & 3D Molecule Generation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Research_Preview-yellow)]()

> **SITP Project (Student Innovation Training Program)** 

## 📖 Introduction

This repository contains the official implementation of my undergraduate research project on **3D Molecule Generation**. 

Traditional molecule generation often struggles to balance validity with 3D conformational stability. This project introduces a **novel hybrid framework** merging **Diffusion Models** and **Autoregressive (AR) methods**. By modeling the generation process as a **composite Markov chain**, we derive a novel two-stage **Variational Lower Bound (VLB)** to optimize the generation quality.

In addition, we provide an interactive property prediction system based on GCN/CVAE.

## 🚀 Key Features

### 1. Novel Generative Architecture
* **Composite Markov Chain:** Unifies Diffusion and AR processes for atom-level generation.
* **E-DiT Backbone:** Designed a novel **Equivariant Diffusion Transformer (E-DiT)** backbone to ensure rotation/translation equivariance in 3D space.
* **DDIM Sampler:** Integrated Denoising Diffusion Implicit Models for accelerated sampling.

### 2. Theoretical Innovation
* Derived a **Two-Stage VLB** (Variational Lower Bound) specifically tailored for this hybrid generation process, ensuring mathematically rigorous optimization.

### 3. High-Performance Engineering
* **Distributed Training:** Supported multi-GPU training pipelines for handling large-scale molecular datasets.
* **Interactive System:** Built a companion GCN/CVAE-based system for molecular property prediction, achieving **99% accuracy** on standard benchmarks.

## 🛠️ Model Architecture

The core model utilizes an **E-DiT (Equivariant Diffusion Transformer)** backbone. It processes molecular graphs while respecting SE(3) symmetry, ensuring that the generated 3D coordinates are physically valid.

