# UE5-ImageNet-OOD-7  
*Photorealistic, fully annotated out-of-distribution benchmark derived from seven ImageNet-1K classes*

[![Hugging Face](https://img.shields.io/badge/-HuggingFace-blue?logo=huggingface&style=flat)](https://huggingface.co/datasets/FlGutbier/UE5-ImageNet-OOD-7)  
Dataset size: **86016 images** :contentReference[oaicite:0]{index=0}  
License: **other** (see below) :contentReference[oaicite:1]{index=1}  

---

## Overview
`UE5-ImageNet-OOD-7` is an out-of-distribution (OOD) test set created entirely in **Unreal Engine 5** using a custom screenshot-automation plug-in.  
Each sample pairs a 512 × 512 PNG render with rich metadata (camera pose, light color, material, fog, background, etc.).  
The goal is to **stress-test ImageNet-trained classifiers** under controlled changes of viewpoint, illumination and scene composition while keeping label semantics identical.

## Key features
| Property | Value |
|----------|-------|
| Classes  | 7 ImageNet-1K categories |
| Images   | 86016 high-quality renders |
| Resolution | 512 × 512 (PNG, lossless) |
| Annotations | Per-image CSV: `class_id`, `background`, `material`, `camera_transform`, `light_rgb`, `fog` |
| Generation engine | Unreal Engine 5.4 with **Lumen**, HW ray tracing |
| Variation axes | Viewpoint, lighting color/intensity, materials, fog (as background simplifier), scene clutter |
| Intended use | OOD robustness evaluation, synthetic-to-real transfer, data-centric analysis |
