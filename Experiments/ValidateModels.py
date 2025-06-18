import torch
from ModelEvaluator import ModelEvaluator

def main():

    # Configuration
    CSV_FILE = r"d:/Bachelor thesis/Validate/analyze/Dataset/Metadata.csv"
    ROOT_DIR = r"d:/Bachelor thesis/Validate/analyze" 
    BATCH_SIZE = 128

    cnn_models = [
        "resnet50",
        "resnet101",
        "densenet201.tv_in1k",
        "convnext_base.fb_in1k",
        "convnext_large_mlp",
    ]
        
    vit_models = [
        "vit_base_patch16_224",
        "vit_large_patch16_224",
        "swin_base_patch4_window7_224",
    ]

    # custom : imageNet classes
    class_map = {
        0: 528,
        1: 526,
        2: 968,
        3: 954,
        4: 879,
        5: 587,
        6: 492,
    }
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: CUDA not available, using CPU. This will be slow.")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(device=device, cnn_models=cnn_models, vit_models=vit_models, class_map=class_map)
    
    # Run evaluation
    evaluator.run_evaluation(CSV_FILE, ROOT_DIR, BATCH_SIZE)


if __name__ == "__main__":
    main()

        