#!/usr/bin/env python3
"""
TensorFlow Lite Model Accuracy Analyzer with Loss Curves
Script complet pour tracer les courbes d'accuracy et loss
Combine la classe d'analyse avec le chargement d'images
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Pour sauvegarder sans affichage
from tflite_runtime.interpreter import Interpreter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
import seaborn as sns
import json
import os
from PIL import Image
from pathlib import Path


class TFLiteModelAnalyzer:
    """Analyze TensorFlow Lite model performance"""
    
    def __init__(self, model_path):
        """
        Initialize the analyzer
        
        Args:
            model_path: Path to .tflite model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model: {model_path}")
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print("\n=== Model Information ===")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Input dtype: {self.input_details[0]['dtype']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
        print(f"Output dtype: {self.output_details[0]['dtype']}")
        print("========================\n")
    
    def predict_single(self, input_data):
        """
        Run inference on a single sample
        
        Args:
            input_data: Single input sample (numpy array)
        
        Returns:
            Model prediction (probabilities)
        """
        # Ensure correct shape and dtype
        input_data = np.array([input_data], dtype=self.input_details[0]['dtype'])
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output_data[0]
    
    def predict_batch(self, X):
        """
        Run inference on multiple samples
        
        Args:
            X: Array of input samples
        
        Returns:
            Array of predictions (probabilities)
        """
        predictions = []
        total = len(X)
        
        print(f"Running inference on {total} samples...")
        for i, sample in enumerate(X):
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{total}")
            
            pred = self.predict_single(sample)
            predictions.append(pred)
        
        print(f"Inference complete!\n")
        return np.array(predictions)
    
    def compute_loss(self, predictions, y_true, num_classes):
        """
        Compute categorical cross-entropy loss
        
        Args:
            predictions: Model predictions (probabilities)
            y_true: True labels (integers)
            num_classes: Number of classes
        
        Returns:
            loss: Cross-entropy loss
        """
        # Convert labels to one-hot encoding
        y_true_one_hot = np.zeros((len(y_true), num_classes))
        y_true_one_hot[np.arange(len(y_true)), y_true] = 1
        
        # Clip predictions to avoid log(0)
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        
        # Compute categorical cross-entropy
        loss = -np.mean(np.sum(y_true_one_hot * np.log(predictions), axis=1))
        
        return loss
    
    def plot_accuracy_and_loss_curves(self, X_test, y_test, batch_size=32, 
                                       save_path_acc='accuracy_curve.png',
                                       save_path_loss='loss_curve.png',
                                       save_path_combined='accuracy_loss_combined.png'):
        """
        Plot accuracy and loss curves
        
        Args:
            X_test: Test data
            y_test: True labels
            batch_size: Batch size for computing metrics
            save_path_acc: Where to save accuracy plot
            save_path_loss: Where to save loss plot
            save_path_combined: Where to save combined plot
        """
        print("Computing accuracy and loss curves...")
        accuracies = []
        losses = []
        sample_counts = []
        n_samples = len(X_test)
        
        # Determine number of classes
        num_classes = len(np.unique(y_test))
        
        for i in range(batch_size, n_samples + 1, batch_size):
            # Get predictions up to current point
            batch_X = X_test[:i]
            batch_y = y_test[:i]
            
            preds = self.predict_batch(batch_X)
            pred_classes = np.argmax(preds, axis=1)
            
            # Compute accuracy
            acc = accuracy_score(batch_y, pred_classes)
            accuracies.append(acc)
            
            # Compute loss
            loss = self.compute_loss(preds, batch_y, num_classes)
            losses.append(loss)
            
            sample_counts.append(i)
            
            print(f"Samples: {i}/{n_samples}, Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        
        # Plot 1: Accuracy Curve
        plt.figure(figsize=(12, 6))
        plt.plot(sample_counts, accuracies, marker='o', linewidth=2, markersize=6, color='#2ecc71')
        plt.title('Model Accuracy Curve', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Test Samples', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        final_acc = accuracies[-1]
        plt.axhline(y=final_acc, color='r', linestyle='--', alpha=0.5, 
                    label=f'Final Accuracy: {final_acc:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path_acc, dpi=300, bbox_inches='tight')
        print(f"Accuracy curve saved to: {save_path_acc}")
        plt.close()
        
        # Plot 2: Loss Curve
        plt.figure(figsize=(12, 6))
        plt.plot(sample_counts, losses, marker='s', linewidth=2, markersize=6, color='#e74c3c')
        plt.title('Model Loss Curve', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Test Samples', fontsize=12)
        plt.ylabel('Loss (Cross-Entropy)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        final_loss = losses[-1]
        plt.axhline(y=final_loss, color='b', linestyle='--', alpha=0.5, 
                    label=f'Final Loss: {final_loss:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path_loss, dpi=300, bbox_inches='tight')
        print(f"Loss curve saved to: {save_path_loss}")
        plt.close()
        
        # Plot 3: Combined Accuracy and Loss
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Accuracy subplot
        ax1.plot(sample_counts, accuracies, marker='o', linewidth=2, markersize=6, color='#2ecc71')
        ax1.set_title('Model Accuracy Over Test Samples', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Test Samples', fontsize=11)
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        ax1.axhline(y=final_acc, color='r', linestyle='--', alpha=0.5, 
                    label=f'Final: {final_acc:.4f}')
        ax1.legend()
        
        # Loss subplot
        ax2.plot(sample_counts, losses, marker='s', linewidth=2, markersize=6, color='#e74c3c')
        ax2.set_title('Model Loss Over Test Samples', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Test Samples', fontsize=11)
        ax2.set_ylabel('Loss (Cross-Entropy)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=final_loss, color='b', linestyle='--', alpha=0.5, 
                    label=f'Final: {final_loss:.4f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path_combined, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {save_path_combined}\n")
        plt.close()
        
        return accuracies, losses, sample_counts
    
    def plot_confusion_matrix(self, X_test, y_test, class_names=None, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix
        
        Args:
            X_test: Test data
            y_test: True labels
            class_names: List of class names
            save_path: Where to save the plot
        """
        print("Computing predictions for confusion matrix...")
        predictions = self.predict_batch(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        
        cm = confusion_matrix(y_test, pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}\n")
        
        return cm
    
    def plot_per_class_metrics(self, X_test, y_test, class_names=None, 
                                save_path='per_class_metrics.png'):
        """
        Plot per-class accuracy, precision, recall, and F1-score
        
        Args:
            X_test: Test data
            y_test: True labels
            class_names: List of class names
            save_path: Where to save the plot
        """
        print("Computing per-class metrics...")
        predictions = self.predict_batch(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Get classification report as dict
        report = classification_report(y_test, pred_classes, 
                                       target_names=class_names,
                                       output_dict=True)
        
        # Extract metrics for each class
        classes = class_names if class_names else [str(i) for i in range(len(np.unique(y_test)))]
        precision = [report[c]['precision'] for c in classes]
        recall = [report[c]['recall'] for c in classes]
        f1 = [report[c]['f1-score'] for c in classes]
        
        # Create bar plot
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        ax.bar(x, recall, width, label='Recall', color='#2ecc71')
        ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics saved to: {save_path}\n")
        
        return report
    
    def generate_full_report(self, X_test, y_test, class_names=None, output_dir='results'):
        """
        Generate complete analysis report with accuracy and loss curves
        
        Args:
            X_test: Test data
            y_test: True labels
            class_names: List of class names
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("GENERATING FULL MODEL ANALYSIS REPORT (WITH LOSS CURVES)")
        print("="*70 + "\n")
        
        # Determine number of classes
        num_classes = len(np.unique(y_test))
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(num_classes)]
        
        # 1. Overall predictions
        print("Running full inference on test set...")
        predictions = self.predict_batch(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        
        # 2. Overall accuracy
        overall_acc = accuracy_score(y_test, pred_classes)
        print(f"\n{'='*70}")
        print(f"OVERALL ACCURACY: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
        print(f"{'='*70}\n")
        
        # 3. Overall loss
        overall_loss = self.compute_loss(predictions, y_test, num_classes)
        print(f"{'='*70}")
        print(f"OVERALL LOSS: {overall_loss:.4f}")
        print(f"{'='*70}\n")
        
        # 4. Accuracy and Loss curves
        accuracies, losses, samples = self.plot_accuracy_and_loss_curves(
            X_test, y_test, 
            save_path_acc=os.path.join(output_dir, 'accuracy_curve.png'),
            save_path_loss=os.path.join(output_dir, 'loss_curve.png'),
            save_path_combined=os.path.join(output_dir, 'accuracy_loss_combined.png')
        )
        
        # 5. Confusion matrix
        cm = self.plot_confusion_matrix(
            X_test, y_test, class_names,
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )
        
        # 6. Per-class metrics
        per_class_report = self.plot_per_class_metrics(
            X_test, y_test, class_names,
            save_path=os.path.join(output_dir, 'per_class_metrics.png')
        )
        
        # 7. Classification report
        print("="*70)
        print("CLASSIFICATION REPORT:")
        print("="*70)
        report = classification_report(y_test, pred_classes, 
                                       target_names=class_names,
                                       output_dict=True)
        print(classification_report(y_test, pred_classes, target_names=class_names))
        print("="*70 + "\n")
        
        # 8. Save results to JSON
        results = {
            'overall_accuracy': float(overall_acc),
            'overall_loss': float(overall_loss),
            'num_samples': int(len(X_test)),
            'num_classes': int(num_classes),
            'accuracy_curve': {
                'samples': [int(s) for s in samples],
                'accuracies': [float(a) for a in accuracies]
            },
            'loss_curve': {
                'samples': [int(s) for s in samples],
                'losses': [float(l) for l in losses]
            },
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 9. Save text summary
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MODEL EVALUATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total Samples: {len(X_test)}\n")
            f.write(f"Number of Classes: {num_classes}\n\n")
            f.write(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n")
            f.write(f"Overall Loss: {overall_loss:.4f}\n\n")
            f.write("="*70 + "\n")
            f.write("CLASSIFICATION REPORT:\n")
            f.write("="*70 + "\n")
            f.write(classification_report(y_test, pred_classes, target_names=class_names))
        
        print(f"\nJSON results saved to: {results_path}")
        print(f"Text summary saved to: {summary_path}")
        print(f"\nAll outputs saved in: {output_dir}/")
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70 + "\n")
        
        print("Generated files:")
        print("  - accuracy_curve.png (Accuracy over samples)")
        print("  - loss_curve.png (Loss over samples)")
        print("  - accuracy_loss_combined.png (Both metrics)")
        print("  - confusion_matrix.png")
        print("  - per_class_metrics.png")
        print("  - results.json (All metrics in JSON)")
        print("  - summary.txt (Text summary)")
        print()


def load_sample_data(test_images_dir='test_images', input_size=224):
    """
    Load test data from image folders
    
    Args:
        test_images_dir: Root directory containing class folders
        input_size: Size to resize images to
    
    Returns:
        X_test: Array of image data
        y_test: Array of labels
        class_names: List of class names
    """
    print(f"Loading test images from: {test_images_dir}")
    
    X_test = []
    y_test = []
    class_names = []
    
    # Get class folders
    class_dirs = sorted([d for d in os.listdir(test_images_dir) 
                         if os.path.isdir(os.path.join(test_images_dir, d))])
    
    if not class_dirs:
        print(f"ERROR: No class folders found in {test_images_dir}")
        print("Expected structure:")
        print("  test_images/")
        print("    class1/")
        print("      image1.jpg")
        print("      image2.png")
        print("    class2/")
        print("      image3.jpg")
        return None, None, None
    
    class_names = class_dirs
    
    # Load images from each class folder
    for class_idx, class_name in enumerate(class_names):
        folder_path = os.path.join(test_images_dir, class_name)
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        
        print(f"\nLoading {class_name}: {len(image_files)} images")
        
        for img_file in image_files:
            try:
                img_path = os.path.join(folder_path, img_file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((input_size, input_size))
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                X_test.append(img_array)
                y_test.append(class_idx)
            except Exception as e:
                print(f"  Error loading {img_file}: {e}")
                continue
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)
    
    print(f"\n✓ Loaded {len(X_test)} test samples from {len(class_names)} classes")
    print(f"Image shape: {X_test[0].shape}")
    print(f"Class names: {class_names}\n")
    
    return X_test, y_test, class_names


def main():
    """Main execution function"""
    
    # ===== CONFIGURATION =====
    # Modify these paths according to your setup
    MODEL_PATH = 'traffic_sign_model_v2.tflite'  # Change to your model
    TEST_IMAGES_DIR = 'test_images'  # Folder containing class subfolders
    OUTPUT_DIR = 'results'
    INPUT_SIZE = 224  # Match your model's input size
    
    print("\n" + "="*70)
    print("TRAFFIC SIGN MODEL ANALYZER - ACCURACY & LOSS CURVES")
    print("="*70 + "\n")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("\nAvailable .tflite files in current directory:")
        tflite_files = [f for f in os.listdir('.') if f.endswith('.tflite')]
        for f in tflite_files:
            print(f"  - {f}")
        return
    
    # Check if test images directory exists
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"ERROR: Test images directory not found: {TEST_IMAGES_DIR}")
        print("\nExpected structure:")
        print("  test_images/")
        print("    class1/")
        print("      image1.jpg")
        print("      image2.png")
        print("    class2/")
        print("      image3.jpg")
        return
    
    # Load test data
    X_test, y_test, class_names = load_sample_data(TEST_IMAGES_DIR, INPUT_SIZE)
    
    if X_test is None:
        return
    
    # Create analyzer
    print("Creating analyzer...")
    analyzer = TFLiteModelAnalyzer(MODEL_PATH)
    
    # Generate full report with loss curves
    print("Generating full report with accuracy and loss curves...")
    analyzer.generate_full_report(X_test, y_test, class_names, OUTPUT_DIR)
    
    print(f"\n✓ All results saved to: {OUTPUT_DIR}/")
    print("Open the PNG files to view the accuracy and loss curves")


if __name__ == '__main__':
    main()
