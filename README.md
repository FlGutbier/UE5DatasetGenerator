# A Synthetic OOD Benchmark Dataset and UE5 Plugin

This Codebase Contains:
  - Download link for the Dataset
  - Source Code for the Plugin and a packaged version for UE5
  - Python code to run evaluations using the dataset including a custom dataloader

## Plugin

The **UE5 Plugin** enables synthetic dataset generation by automating:

- Scene setup and variation
- Camera placement
- Object placement
- Screenshot capture
- Metadata export (e.g., class label, camera pose, lighting color)

The plugin supports batch rendering and is compatible with Unreal Engine 5.

## Dataset

The dataset consists of 86016 rendered images of 7 selected ImageNet classes under varying and combined conditions, including:

- Viewpoint changes
- Lighting variations 
- Fog
- Material and background shifts  

The goal is to provide a controlled benchmark for studying robustness, sensitivity, and generalization limits of image classifiers.
**Note:** Some assets used in the dataset are restricted for use in generative AI.

## Evaluation Code

Evaluation scripts are included to:

- Load the synthetic dataset
- Run inference with pretrained ImageNet classifiers (e.g., ResNet, ViT)
- Compute metrics such as Top-1/Top-5 Accuracy and AUC
- Analyze the impact of controlled variables (e.g., fog, angle)

All scripts are written in Python and use PyTorch.
