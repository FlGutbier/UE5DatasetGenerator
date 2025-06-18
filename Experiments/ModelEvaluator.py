import re
import json
import torch
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from CustomImageDataset import CustomImageDataset
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def collate_metadata(batch):
    """Custom collate function to properly handle metadata."""
    # Separate images, labels, and metadata
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    imagenet_labels = torch.tensor([item["imagenet_label"] for item in batch])
    indices = torch.tensor([item["idx"] for item in batch])
    
    # Collect metadata as list of dictionaries
    metadata_list = []
    for item in batch:
        meta_dict = {
            "idx": item["idx"],
            "image_path": item["image_path"],
            "object": item["object"],
            "level": item["level"],
            "material": item["material"],
            "camera_position": item["camera_position"],
            "light_color": item["light_color"],
            "fog": item["fog"],
        }
        metadata_list.append(meta_dict)
    
    return {
        "image": images,
        "label": labels,
        "imagenet_label": imagenet_labels,
        "idx": indices,
        "metadata": metadata_list,
    }


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        # Convert keys and values
        new_dict = {}
        for key, value in obj.items():
            # Convert numpy types in keys
            if hasattr(key, 'item'):  # numpy scalar
                new_key = key.item()
            elif isinstance(key, (np.integer, np.floating)):
                new_key = key.item()
            else:
                new_key = key
            
            new_dict[str(new_key)] = convert_numpy_types(value)
        return new_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class ModelEvaluator:
    def __init__(self, device="cuda", random_state=42, cnn_models = [], vit_models = [], class_map = {}):
        self.device = device
        self.random_state = random_state
        self.class_map = class_map

        # Model configurations
        self.cnn_models = cnn_models
        
        self.vit_models = vit_models
        
        # Create inverse mapping: ImageNet class -> custom class
        self.inverse_class_map = {v: k for k, v in self.class_map.items()}
        print(f"Inverse class mapping: {self.inverse_class_map}")
        
        # Define baseline conditions for analysis
        self.baseline_conditions = {
            "material": "Default",
            "light_color": "(R=1.0,G=1.0,B=1.0,A=0.0)",
            "fog": False,
            "camera_position": "X=0.000 Y=1.000 Z=0.200"
        }
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize from 512x512 to 224x224
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    def load_dataset(self, csv_file, root_dir, batch_size=64):
        """Load and return dataset and dataloader."""
        dataset = CustomImageDataset(
            csv_file=csv_file,
            root_dir=root_dir,
            transform=self.transform,
            class_map=self.class_map,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Keep False since we handle shuffling manually for KNN
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_metadata,
        )
        
        return dataset, dataloader

    def load_model(self, model_name):
        """Load pretrained model from timm."""
        model = timm.create_model(model_name, pretrained=True, num_classes=1000)
        model.eval()
        model.to(self.device)
        return model

    def evaluate_cnn(self, model, dataloader, model_name):
        """Evaluate CNN model following PUG paper approach."""
        all_predictions = []
        all_imagenet_predictions = []
        all_probs_mapped = []  # 7-class probabilities
        all_probs_full = []    # 1000-class probabilities  
        all_labels = []
        all_metadata = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
                images = batch["image"].to(self.device)
                labels = batch["label"]  # Original custom labels
                metadata = batch["metadata"]
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    probs_full = F.softmax(outputs, dim=1)  # Full 1000-class probabilities
                
                # Get ImageNet predictions (from all 1000 classes)
                imagenet_preds = torch.argmax(outputs, dim=1)
                
                # Map ImageNet predictions to custom classes (following PUG approach)
                mapped_preds = []
                for pred in imagenet_preds:
                    pred_item = pred.item()
                    if pred_item in self.inverse_class_map:
                        mapped_preds.append(self.inverse_class_map[pred_item])
                    else:
                        mapped_preds.append(999)  # Invalid class for unmapped predictions
                
                # Extract probabilities for our relevant ImageNet classes only
                mapped_indices = list(self.class_map.values())
                mapped_probs = probs_full[:, mapped_indices]
                
                # Don't normalize for CNN (following PUG paper approach)
                # The original probabilities are already properly normalized
                
                if batch_idx == 0:  # Debug first batch
                    print(f"\n=== CNN DEBUG - {model_name} ===")
                    print(f"Mapped probs sum (first sample): {mapped_probs[0].sum():.4f}")
                    print(f"Full probs sum (first sample): {probs_full[0].sum():.4f}")
                    print(f"=== END CNN DEBUG ===\n")
                
                all_predictions.extend(mapped_preds)
                all_imagenet_predictions.extend(imagenet_preds.cpu().numpy())
                all_probs_mapped.append(mapped_probs.cpu().numpy())  # 7-class subset
                all_probs_full.append(probs_full.cpu().numpy())      # 1000-class original
                all_labels.extend(labels.numpy())
                all_metadata.extend(metadata)
        
        all_probs_mapped = np.vstack(all_probs_mapped)
        all_probs_full = np.vstack(all_probs_full)
        
        return all_predictions, all_probs_mapped, all_probs_full, all_labels, all_metadata, all_imagenet_predictions

    def evaluate_vit(self, model, dataloader, model_name):
        """Evaluate ViT model with full probability storage."""
        all_predictions = []
        all_imagenet_predictions = []
        all_probs_mapped = []  # 7-class probabilities
        all_probs_full = []    # 1000-class probabilities
        all_labels = []
        all_metadata = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
                images = batch["image"].to(self.device)
                labels = batch["label"]  # Original custom labels
                metadata = batch["metadata"]
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    probs_full = F.softmax(outputs, dim=1)  # Full 1000-class probabilities
                
                # Get ImageNet predictions (from all 1000 classes)
                imagenet_preds = torch.argmax(outputs, dim=1)
                
                # Map ImageNet predictions to custom classes
                mapped_preds = []
                for pred in imagenet_preds:
                    pred_item = pred.item()
                    if pred_item in self.inverse_class_map:
                        mapped_preds.append(self.inverse_class_map[pred_item])
                    else:
                        mapped_preds.append(999)  # Invalid class for unmapped predictions
                
                # Extract probabilities for our relevant ImageNet classes only
                mapped_indices = list(self.class_map.values())
                mapped_probs = probs_full[:, mapped_indices]
                
                # Normalize extracted probabilities to sum to 1.0
                mapped_probs_sum = mapped_probs.sum(dim=1, keepdim=True)
                mapped_probs_sum[mapped_probs_sum == 0] = 1  # Avoid division by zero
                mapped_probs_normalized = mapped_probs / mapped_probs_sum
                
                if batch_idx == 0:  # Debug first batch
                    print(f"\n=== ViT NORMALIZATION DEBUG - {model_name} ===")
                    print(f"Original mapped probs sum (first sample): {mapped_probs[0].sum():.4f}")
                    print(f"Normalized mapped probs sum (first sample): {mapped_probs_normalized[0].sum():.4f}")
                    print(f"Full probs sum (first sample): {probs_full[0].sum():.4f}")
                    print(f"=== END NORMALIZATION DEBUG ===\n")
                
                all_predictions.extend(mapped_preds)
                all_imagenet_predictions.extend(imagenet_preds.cpu().numpy())
                all_probs_mapped.append(mapped_probs_normalized.cpu().numpy())  # 7-class normalized
                all_probs_full.append(probs_full.cpu().numpy())  # 1000-class original
                all_labels.extend(labels.numpy())
                all_metadata.extend(metadata)
        
        all_probs_mapped = np.vstack(all_probs_mapped)
        all_probs_full = np.vstack(all_probs_full)
        
        return all_predictions, all_probs_mapped, all_probs_full, all_labels, all_metadata, all_imagenet_predictions
    
    def calculate_metrics_robust(self, predictions, probs, labels, num_classes=7):
        """Calculate metrics with robust AUC calculation and detailed debugging."""
        pred_classes = np.array(predictions)
        true_classes = np.array(labels)
        probs = np.array(probs)
        
        print(f"\n=== DEBUG AUC CALCULATION ===")
        print(f"pred_classes shape: {pred_classes.shape}, unique: {np.unique(pred_classes)}")
        print(f"true_classes shape: {true_classes.shape}, unique: {np.unique(true_classes)}")
        print(f"probs shape: {probs.shape}")
        print(f"probs range: [{np.min(probs):.4f}, {np.max(probs):.4f}]")
        
        # Basic accuracy metrics
        valid_mask = pred_classes != 999
        valid_preds = pred_classes[valid_mask]
        valid_true = true_classes[valid_mask]
        
        print(f"Valid predictions: {len(valid_preds)}/{len(pred_classes)} ({len(valid_preds)/len(pred_classes)*100:.1f}%)")
        
        top1_acc_overall = accuracy_score(true_classes, pred_classes)
        
        # Top-5 accuracy
        top5_acc = 0.0
        if len(probs.shape) > 1 and len(valid_preds) > 0 and probs.shape[1] > 1:
            try:
                valid_probs = probs[valid_mask]
                k = min(5, valid_probs.shape[1])
                unique_classes = np.unique(valid_true)
                
                top5_acc = top_k_accuracy_score(
                    valid_true, 
                    valid_probs, 
                    k=k, 
                    labels=unique_classes
                )
            except Exception as e:
                print(f"Top-5 accuracy calculation failed: {e}")
                top5_acc = top1_acc_overall
        else:
            top5_acc = top1_acc_overall
        
        # AUC Calculation: FULL DATASET (not just valid predictions)
        auc_full_dataset = float("nan")
        try:
            # Create probability matrix for ALL samples
            all_probs = np.zeros((len(pred_classes), num_classes))
            
            # For valid predictions: use the actual probabilities
            if len(valid_preds) > 0:
                valid_probs = probs[valid_mask]
                # Basic normalization
                row_sums = valid_probs.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                valid_probs_normalized = valid_probs / row_sums
                all_probs[valid_mask] = valid_probs_normalized
            
            # For invalid predictions: assign very low uniform probabilities
            # This represents the model's uncertainty/rejection
            invalid_mask = ~valid_mask
            all_probs[invalid_mask] = 1.0 / num_classes  # Uniform = maximum uncertainty
            
            # Verify probabilities
            row_sums = all_probs.sum(axis=1)
            print(f"Full dataset probs row sums - min: {np.min(row_sums):.4f}, max: {np.max(row_sums):.4f}")
            
            # Calculate AUC on the full dataset
            unique_true_classes = np.unique(true_classes)
            if len(unique_true_classes) > 1:
                auc_full_dataset = roc_auc_score(
                    y_true=true_classes,  # All true labels
                    y_score=all_probs,    # All probabilities (including uniform for rejections)
                    multi_class='ovr',
                    average='macro',
                    labels=list(range(num_classes))
                )
                print(f"AUC (full dataset): {auc_full_dataset:.4f}")
            else:
                print("AUC: Cannot calculate - only one class present")
                
        except Exception as e:
            print(f"AUC calculation failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"=== END DEBUG ===\n")
        
        return {
            "top1_accuracy": top1_acc_overall,
            "top5_accuracy": top5_acc,
            "auc_full_dataset": auc_full_dataset,
            "valid_prediction_rate": len(valid_preds) / len(pred_classes),
        }

    def analyze_baseline_performance(self, df):
        """Analyze performance on baseline/default conditions."""
        print("\n" + "="*60)
        print("BASELINE PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Filter for baseline conditions
        baseline_mask = (
            (df['material'] == self.baseline_conditions['material']) &
            (df['light_color'] == self.baseline_conditions['light_color']) &
            (df['fog'] == self.baseline_conditions['fog']) & 
            (df['camera_position'] == self.baseline_conditions['camera_position'])
        )
        
        baseline_df = df[baseline_mask].copy()
        
        if len(baseline_df) == 0:
            print("No samples found matching baseline conditions!")
            return {}
        
        print(f"Found {len(baseline_df)} baseline samples out of {len(df)} total samples ({len(baseline_df)/len(df)*100:.1f}%)")
        
        results = {}
        
        # Analyze by model
        for model in df['model'].unique():
            model_baseline = baseline_df[baseline_df['model'] == model]
            if len(model_baseline) == 0:
                continue
                
            # Calculate baseline metrics
            valid_mask = model_baseline['is_valid_prediction'] == True
            valid_baseline = model_baseline[valid_mask]
            
            overall_acc = accuracy_score(model_baseline['true_label'], model_baseline['prediction'])
            valid_acc = accuracy_score(valid_baseline['true_label'], valid_baseline['prediction']) if len(valid_baseline) > 0 else 0.0
            valid_rate = len(valid_baseline) / len(model_baseline)
            
            # Per-class accuracy on baseline
            class_accuracies = {}
            for class_id in sorted(model_baseline['true_label'].unique()):
                class_samples = model_baseline[model_baseline['true_label'] == class_id]
                if len(class_samples) > 0:
                    class_acc = accuracy_score(class_samples['true_label'], class_samples['prediction'])
                    class_accuracies[int(class_id)] = {  # Convert to int for JSON compatibility
                        'accuracy': float(class_acc),
                        'n_samples': int(len(class_samples)),
                        'valid_predictions': int((class_samples['is_valid_prediction'] == True).sum())
                    }
            
            results[model] = {
                'overall_accuracy': float(overall_acc),
                'valid_accuracy': float(valid_acc),
                'valid_prediction_rate': float(valid_rate),
                'n_samples': int(len(model_baseline)),
                'class_accuracies': class_accuracies
            }
            
            print(f"\n--- {model} ---")
            print(f"Overall accuracy on baseline: {overall_acc:.3f}")
            print(f"Valid accuracy on baseline: {valid_acc:.3f}")
            print(f"Valid prediction rate: {valid_rate:.3f}")
            print("Per-class baseline accuracy:")
            for class_id, metrics in class_accuracies.items():
                print(f"  Class {class_id}: {metrics['accuracy']:.3f} ({metrics['valid_predictions']}/{metrics['n_samples']} valid)")
        
        return results

    def analyze_variable_impact(self, df):
        """Analyze the impact of each variable on model performance."""
        print("\n" + "="*60)
        print("VARIABLE IMPACT ANALYSIS")
        print("="*60)
        
        variables = ['material', 'light_color', 'fog', 'camera_position', 'level']
        results = {}
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model].copy()
            results[model] = {}
            
            print(f"\n--- {model} ---")
            
            # Get baseline accuracy for comparison
            baseline_mask = (
                (model_df['material'] == self.baseline_conditions['material']) &
                (model_df['light_color'] == self.baseline_conditions['light_color']) &
                (model_df['fog'] == self.baseline_conditions['fog'])
            )
            baseline_df = model_df[baseline_mask]
            baseline_acc = accuracy_score(baseline_df['true_label'], baseline_df['prediction']) if len(baseline_df) > 0 else 0.0
            
            for variable in variables:
                print(f"\nAnalyzing impact of {variable}:")
                
                # Get unique values for this variable
                unique_values = model_df[variable].unique()
                
                variable_results = {}
                
                for value in unique_values:
                    subset = model_df[model_df[variable] == value]
                    if len(subset) == 0:
                        continue
                    
                    # Calculate metrics for this subset
                    valid_mask = subset['is_valid_prediction'] == True
                    valid_subset = subset[valid_mask]
                    
                    overall_acc = accuracy_score(subset['true_label'], subset['prediction'])
                    valid_acc = accuracy_score(valid_subset['true_label'], valid_subset['prediction']) if len(valid_subset) > 0 else 0.0
                    valid_rate = len(valid_subset) / len(subset)
                    
                    # Calculate impact relative to baseline
                    acc_impact = overall_acc - baseline_acc
                    
                    variable_results[str(value)] = {
                        'overall_accuracy': float(overall_acc),
                        'valid_accuracy': float(valid_acc),
                        'valid_prediction_rate': float(valid_rate),
                        'accuracy_impact': float(acc_impact),
                        'n_samples': int(len(subset))
                    }
                    
                    print(f"  {value}: Acc={overall_acc:.3f} (Δ{acc_impact:+.3f}), Valid={valid_rate:.3f}, N={len(subset)}")
                
                results[model][variable] = variable_results
        
        return results

    def _format_label(self, value, variable):
        """Smart label formatting based on variable type."""
        if variable == 'light_color':
            # Handle RGBA values like "(R=0.0,G=1.0,B=0.0,A=1.0)"
            if 'R=' in value and 'G=' in value and 'B=' in value:
                # Extract RGBA values
                r_match = re.search(r'R=([\d.]+)', value)
                g_match = re.search(r'G=([\d.]+)', value)
                b_match = re.search(r'B=([\d.]+)', value)
                a_match = re.search(r'A=([\d.]+)', value)
                
                if all([r_match, g_match, b_match]):
                    r = float(r_match.group(1))
                    g = float(g_match.group(1))
                    b = float(b_match.group(1))
                    
                    # Convert to 0-255 scale if values are 0-1
                    if r <= 1.0 and g <= 1.0 and b <= 1.0:
                        r, g, b = int(r*255), int(g*255), int(b*255)
                    
                    return f"RGB({r},{g},{b})"
            return value[:15]
        
        elif variable == 'camera_position':
            # Handle camera positions like "X=0.500 Y=-0.500 Z=1.000"
            if 'X=' in value and 'Y=' in value and 'Z=' in value:
                # Extract XYZ values
                x_match = re.search(r'X=([-\d.]+)', value)
                y_match = re.search(r'Y=([-\d.]+)', value)
                z_match = re.search(r'Z=([-\d.]+)', value)
                
                if all([x_match, y_match, z_match]):
                    x = float(x_match.group(1))
                    y = float(y_match.group(1))
                    z = float(z_match.group(1))
                    return f"({x:.1f},{y:.1f},{z:.1f})"
            return value[:15]
        
        elif variable == 'material':
            # Truncate material names
            return value[:12]
        
        else:
            # Default truncation for other variables
            return value[:15]

    def generate_impact_visualizations(self, df, save_plots=True):
        """Generate visualizations for variable impact analysis."""
        print("\n" + "="*60)
        print("GENERATING IMPACT VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        variables = ['material', 'light_color', 'fog', 'camera_position', 'level']
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model].copy()
            model_clean = model.replace('/', '_')
            
            # Create a larger figure with more spacing
            fig, axes = plt.subplots(2, 3, figsize=(20, 14))
            fig.suptitle(f'Variable Impact Analysis - {model}', fontsize=16)
            
            # Adjust subplot spacing - extra bottom margin for diagonal labels
            plt.subplots_adjust(
                left=0.08, bottom=0.20, right=0.95, top=0.92, 
                wspace=0.3, hspace=0.4
            )
            
            for idx, variable in enumerate(variables):
                row = idx // 3
                col = idx % 3
                ax = axes[row, col]
                
                # Calculate accuracy by variable value
                accuracy_by_value = []
                value_names = []
                sample_counts = []
                
                for value in model_df[variable].unique():
                    subset = model_df[model_df[variable] == value]
                    if len(subset) > 0:
                        acc = accuracy_score(subset['true_label'], subset['prediction'])
                        accuracy_by_value.append(acc)
                        
                        # Smart label truncation based on variable type
                        label = self._format_label(str(value), variable)
                        value_names.append(label)
                        sample_counts.append(len(subset))
                
                # Create bar plot
                bars = ax.bar(range(len(accuracy_by_value)), accuracy_by_value)
                ax.set_xlabel(variable.replace('_', ' ').title(), fontsize=10)
                ax.set_ylabel('Accuracy', fontsize=10)
                ax.set_title(f'{variable.replace("_", " ").title()} Impact', fontsize=11)
                ax.set_xticks(range(len(value_names)))
                
                # Diagonal labels for better readability
                ax.set_xticklabels(
                    value_names, 
                    rotation=45,  # Diagonal rotation
                    ha='right',   # Right-aligned for diagonal text
                    va='top',     # Top-aligned
                    fontsize=8
                )
                
                # Set y-axis limits to accommodate annotations
                max_acc = max(accuracy_by_value) if accuracy_by_value else 1.0
                ax.set_ylim(0, max_acc * 1.15)
                
                # Add sample count annotations
                for i, (bar, count) in enumerate(zip(bars, sample_counts)):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2., 
                        height + 0.01,
                        f'n={count}', 
                        ha='center', 
                        va='bottom', 
                        fontsize=7
                    )
            
            # Remove empty subplot
            if len(variables) < 6:
                fig.delaxes(axes[1, 2])
            
            # Use tight_layout with padding
            plt.tight_layout(pad=2.0)
            
            if save_plots:
                plt.savefig(
                    f'variable_impact_{model_clean}.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none'
                )
                print(f"Saved plot: variable_impact_{model_clean}.png")
            
            plt.show()

    def save_predictions(self, predictions, probs, labels, metadata, model_name, imagenet_predictions=None):
        """Save predictions with metadata to CSV."""
        results = []
        
        for i, (pred, prob, label, meta) in enumerate(
            zip(predictions, probs, labels, metadata)
        ):
            result = {
                "model": model_name,
                "prediction": pred,
                "true_label": label,
                "confidence": np.max(prob) if len(prob) > 0 else 0.0,
                "is_valid_prediction": pred != 999,
                **meta,
            }
            
            # Add ImageNet prediction if available
            if imagenet_predictions is not None:
                result["imagenet_prediction"] = imagenet_predictions[i]
            
            # Add class probabilities
            for j, p in enumerate(prob):
                result[f"prob_class_{j}"] = p
            
            results.append(result)
        
        # Save to CSV
        df = pd.DataFrame(results)
        output_file = f"predictions_{model_name.replace('/', '_')}.csv"
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        return df

    def run_evaluation(self, csv_file, root_dir, batch_size=64):
        """Run complete evaluation pipeline."""
        print("Loading dataset...")
        dataset, dataloader = self.load_dataset(csv_file, root_dir, batch_size)
        
        results_summary = []
        all_predictions_dfs = []
        
        # Evaluate CNN models
        print("\n=== Evaluating CNN Models (PUG Paper Style) ===")
        for model_name in self.cnn_models:
            print(f"\nEvaluating {model_name}...")
            try:
                model = self.load_model(model_name)
                predictions, probs_mapped, probs_full, labels, metadata, imagenet_preds = self.evaluate_cnn(
                    model, dataloader, model_name
                )
                
                metrics = self.calculate_metrics_robust(predictions, probs_mapped, labels)
                metrics["model"] = model_name
                metrics["type"] = "CNN"
                results_summary.append(metrics)
                
                pred_df = self.save_predictions(
                    predictions, probs_mapped, labels, metadata, model_name, imagenet_preds
                )
                all_predictions_dfs.append(pred_df)
                
                print(f"Results: {metrics}")
                
                # Free memory
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Evaluate ViT models
        print("\n=== Evaluating ViT Models (ImageNet Classification) ===")
        for model_name in self.vit_models:
            print(f"\nEvaluating {model_name}...")
            try:
                model = self.load_model(model_name)
                predictions, probs_mapped, probs_full, labels, metadata, imagenet_preds = self.evaluate_vit(
                    model, dataloader, model_name
                )
                
                metrics = self.calculate_metrics_robust(predictions, probs_mapped, labels)
                metrics["model"] = model_name
                metrics["type"] = "ViT"
                results_summary.append(metrics)
                
                pred_df = self.save_predictions(
                    predictions, probs_mapped, labels, metadata, model_name, imagenet_preds
                )
                all_predictions_dfs.append(pred_df)
                
                print(f"Results: {metrics}")
                
                # Free memory
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save summary results
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv("evaluation_summary.csv", index=False)
        print("\n=== Evaluation Summary ===")
        print(summary_df.to_string(index=False))

        print(f"\nSummary saved to evaluation_summary.csv")
        
        # Perform detailed analysis if we have predictions
        if all_predictions_dfs:
            print("\n" + "="*80)
            print("PERFORMING DETAILED ANALYSIS")
            print("="*80)
            
            # Combine all predictions for analysis
            combined_df = pd.concat(all_predictions_dfs, ignore_index=True)
            
            # 1. Baseline performance analysis
            baseline_results = self.analyze_baseline_performance(combined_df)
            
            # 2. Variable impact analysis
            impact_results = self.analyze_variable_impact(combined_df)
            
            # 3. Generate visualizations
            self.generate_impact_visualizations(combined_df, save_plots=True)
            
            # Save detailed analysis results with proper type conversion
            analysis_results = {
                'baseline_performance': baseline_results,
                'variable_impact': impact_results
            }
            
            # Convert numpy types to JSON-serializable types
            analysis_results_clean = convert_numpy_types(analysis_results)
            
            # Save as JSON for further analysis
            with open('detailed_analysis_results.json', 'w') as f:
                json.dump(analysis_results_clean, f, indent=2)
            
            print(f"\nDetailed analysis results saved to detailed_analysis_results.json")