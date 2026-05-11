#!/usr/bin/env python3
"""
Générer des courbes d'Accuracy et Loss - Training vs Validation
Script pour visualiser les performances du modèle pendant l'entraînement
Avec 50 epochs comme dans votre exemple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import json
import os


def generate_synthetic_training_data(num_epochs=50, num_batches=100):
    """
    Générer des données de formation réalistes
    (Dans la pratique, vous chargerez vos vraies données)
    
    Args:
        num_epochs: Nombre d'epochs (50 par défaut)
        num_batches: Nombre de batches par epoch
    
    Returns:
        Données d'entraînement et validation
    """
    print(f"Génération des données de formation ({num_epochs} epochs)...\n")
    
    # Simulation du modèle qui apprend
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    epochs_list = list(range(1, num_epochs + 1))
    
    for epoch in range(1, num_epochs + 1):
        # Loss décroît exponentiellement (comme une courbe d'apprentissage réelle)
        train_loss = 2.5 * np.exp(-epoch / 10) + 0.05 * np.random.randn()
        val_loss = 2.2 * np.exp(-epoch / 10) + 0.1 * np.random.randn()
        
        # Accuracy augmente (inverse du loss)
        train_acc = 1 - (train_loss / 5.0)
        val_acc = 1 - (val_loss / 5.0)
        
        # Assurer que les valeurs sont réalistes
        train_loss = max(0.05, train_loss)
        val_loss = max(0.05, val_loss)
        train_acc = np.clip(train_acc, 0, 1)
        val_acc = np.clip(val_acc, 0, 1)
        
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        training_accuracy.append(train_acc)
        validation_accuracy.append(val_acc)
        
        if (epoch) % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs} - "
                  f"Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    print()
    return {
        'epochs': epochs_list,
        'training_loss': training_loss,
        'validation_loss': validation_loss,
        'training_accuracy': training_accuracy,
        'validation_accuracy': validation_accuracy
    }


def plot_loss_curves(data, output_dir='results', title_suffix=''):
    """
    Tracer les courbes de Loss (Training vs Validation)
    
    Args:
        data: Dictionnaire contenant les données d'entraînement
        output_dir: Dossier pour sauvegarder
        title_suffix: Suffixe pour le titre
    """
    epochs = data['epochs']
    training_loss = data['training_loss']
    validation_loss = data['validation_loss']
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, training_loss, label='Training', linewidth=2.5, 
             marker='o', markersize=4, color='#3498db')
    plt.plot(epochs, validation_loss, label='Validation', linewidth=2.5, 
             marker='s', markersize=4, color='#e74c3c')
    
    plt.title(f'Model Loss {title_suffix}', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Ajouter les valeurs finales
    final_train_loss = training_loss[-1]
    final_val_loss = validation_loss[-1]
    plt.text(0.98, 0.97, f'Final Train Loss: {final_train_loss:.4f}\nFinal Val Loss: {final_val_loss:.4f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    loss_path = os.path.join(output_dir, 'model_loss.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"✅ Loss curve saved: {loss_path}")
    plt.close()


def plot_accuracy_curves(data, output_dir='results', title_suffix=''):
    """
    Tracer les courbes d'Accuracy (Training vs Validation)
    
    Args:
        data: Dictionnaire contenant les données d'entraînement
        output_dir: Dossier pour sauvegarder
        title_suffix: Suffixe pour le titre
    """
    epochs = data['epochs']
    training_accuracy = data['training_accuracy']
    validation_accuracy = data['validation_accuracy']
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, training_accuracy, label='Training', linewidth=2.5, 
             marker='o', markersize=4, color='#2ecc71')
    plt.plot(epochs, validation_accuracy, label='Validation', linewidth=2.5, 
             marker='s', markersize=4, color='#f39c12')
    
    plt.title(f'Model Accuracy {title_suffix}', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Ajouter les valeurs finales
    final_train_acc = training_accuracy[-1]
    final_val_acc = validation_accuracy[-1]
    plt.text(0.98, 0.03, f'Final Train Acc: {final_train_acc:.4f}\nFinal Val Acc: {final_val_acc:.4f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    accuracy_path = os.path.join(output_dir, 'model_accuracy.png')
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    print(f"✅ Accuracy curve saved: {accuracy_path}")
    plt.close()


def plot_combined_curves(data, output_dir='results', title_suffix=''):
    """
    Tracer Loss et Accuracy ensemble (2 sous-graphiques)
    
    Args:
        data: Dictionnaire contenant les données d'entraînement
        output_dir: Dossier pour sauvegarder
        title_suffix: Suffixe pour le titre
    """
    epochs = data['epochs']
    training_loss = data['training_loss']
    validation_loss = data['validation_loss']
    training_accuracy = data['training_accuracy']
    validation_accuracy = data['validation_accuracy']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Loss
    ax1.plot(epochs, training_loss, label='Training', linewidth=2.5, 
             marker='o', markersize=4, color='#3498db')
    ax1.plot(epochs, validation_loss, label='Validation', linewidth=2.5, 
             marker='s', markersize=4, color='#e74c3c')
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Accuracy
    ax2.plot(epochs, training_accuracy, label='Training', linewidth=2.5, 
             marker='o', markersize=4, color='#2ecc71')
    ax2.plot(epochs, validation_accuracy, label='Validation', linewidth=2.5, 
             marker='s', markersize=4, color='#f39c12')
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    combined_path = os.path.join(output_dir, 'training_history_combined.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✅ Combined curves saved: {combined_path}")
    plt.close()


def plot_training_history_vertical(data, output_dir='results'):
    """
    Tracer Loss et Accuracy en vertical (comme votre image d'exemple)
    
    Args:
        data: Dictionnaire contenant les données d'entraînement
        output_dir: Dossier pour sauvegarder
    """
    epochs = data['epochs']
    training_loss = data['training_loss']
    validation_loss = data['validation_loss']
    training_accuracy = data['training_accuracy']
    validation_accuracy = data['validation_accuracy']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Subplot 1: Loss (en haut)
    ax1.plot(epochs, training_loss, label='Training', linewidth=2.5, 
             marker='o', markersize=5, color='#3498db')
    ax1.plot(epochs, validation_loss, label='Validation', linewidth=2.5, 
             marker='s', markersize=5, color='#e74c3c')
    ax1.set_title('Model Loss', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Accuracy (en bas)
    ax2.plot(epochs, training_accuracy, label='Training', linewidth=2.5, 
             marker='o', markersize=5, color='#2ecc71')
    ax2.plot(epochs, validation_accuracy, label='Validation', linewidth=2.5, 
             marker='s', markersize=5, color='#f39c12')
    ax2.set_title('Model Accuracy', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    vertical_path = os.path.join(output_dir, 'training_history_vertical.png')
    plt.savefig(vertical_path, dpi=300, bbox_inches='tight')
    print(f"✅ Vertical curves saved: {vertical_path}")
    plt.close()


def save_training_data_json(data, output_dir='results'):
    """
    Sauvegarder les données d'entraînement en JSON
    
    Args:
        data: Dictionnaire contenant les données
        output_dir: Dossier pour sauvegarder
    """
    json_data = {
        'epochs': data['epochs'],
        'training_loss': [float(x) for x in data['training_loss']],
        'validation_loss': [float(x) for x in data['validation_loss']],
        'training_accuracy': [float(x) for x in data['training_accuracy']],
        'validation_accuracy': [float(x) for x in data['validation_accuracy']],
        'final_metrics': {
            'final_training_loss': float(data['training_loss'][-1]),
            'final_validation_loss': float(data['validation_loss'][-1]),
            'final_training_accuracy': float(data['training_accuracy'][-1]),
            'final_validation_accuracy': float(data['validation_accuracy'][-1]),
            'total_epochs': len(data['epochs'])
        }
    }
    
    json_path = os.path.join(output_dir, 'training_history.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✅ Training data saved: {json_path}")


def generate_summary_report(data, output_dir='results'):
    """
    Générer un rapport textuel des résultats
    
    Args:
        data: Dictionnaire contenant les données
        output_dir: Dossier pour sauvegarder
    """
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL TRAINING SUMMARY - 50 EPOCHS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Epochs: {len(data['epochs'])}\n\n")
        
        f.write("INITIAL METRICS (Epoch 1):\n")
        f.write(f"  Training Loss: {data['training_loss'][0]:.4f}\n")
        f.write(f"  Validation Loss: {data['validation_loss'][0]:.4f}\n")
        f.write(f"  Training Accuracy: {data['training_accuracy'][0]:.4f}\n")
        f.write(f"  Validation Accuracy: {data['validation_accuracy'][0]:.4f}\n\n")
        
        f.write("FINAL METRICS (Epoch 50):\n")
        f.write(f"  Training Loss: {data['training_loss'][-1]:.4f}\n")
        f.write(f"  Validation Loss: {data['validation_loss'][-1]:.4f}\n")
        f.write(f"  Training Accuracy: {data['training_accuracy'][-1]:.4f}\n")
        f.write(f"  Validation Accuracy: {data['validation_accuracy'][-1]:.4f}\n\n")
        
        f.write("IMPROVEMENTS:\n")
        loss_improvement = data['training_loss'][0] - data['training_loss'][-1]
        acc_improvement = data['training_accuracy'][-1] - data['training_accuracy'][0]
        f.write(f"  Loss Reduction: {loss_improvement:.4f}\n")
        f.write(f"  Accuracy Gain: {acc_improvement:.4f}\n\n")
        
        f.write("OVERFITTING ANALYSIS:\n")
        final_loss_diff = abs(data['validation_loss'][-1] - data['training_loss'][-1])
        if final_loss_diff > 0.1:
            f.write(f"  ⚠️  Possible overfitting detected (Loss diff: {final_loss_diff:.4f})\n")
        else:
            f.write(f"  ✅ Good generalization (Loss diff: {final_loss_diff:.4f})\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✅ Summary report saved: {summary_path}")


def main():
    """Main function"""
    
    # Configuration
    NUM_EPOCHS = 50  # 50 epochs comme votre exemple
    OUTPUT_DIR = 'results'
    
    print("\n" + "=" * 70)
    print("TRAFFIC SIGN MODEL - TRAINING CURVES GENERATOR")
    print(f"Generating curves for {NUM_EPOCHS} epochs")
    print("=" * 70 + "\n")
    
    # Créer le dossier de résultats
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Générer les données
    data = generate_synthetic_training_data(num_epochs=NUM_EPOCHS)
    
    print("\nGénération des graphiques...\n")
    
    # Générer les courbes
    plot_loss_curves(data, OUTPUT_DIR)
    plot_accuracy_curves(data, OUTPUT_DIR)
    plot_combined_curves(data, OUTPUT_DIR)
    plot_training_history_vertical(data, OUTPUT_DIR)
    
    # Sauvegarder les données
    save_training_data_json(data, OUTPUT_DIR)
    generate_summary_report(data, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Fichiers générés dans: {OUTPUT_DIR}/")
    print("\n  Graphiques:")
    print("    - model_loss.png (Loss: Training vs Validation)")
    print("    - model_accuracy.png (Accuracy: Training vs Validation)")
    print("    - training_history_combined.png (Loss & Accuracy côte à côte)")
    print("    - training_history_vertical.png (Loss & Accuracy en vertical)")
    print("\n  Données:")
    print("    - training_history.json (Toutes les données)")
    print("    - training_summary.txt (Rapport textuel)")
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
