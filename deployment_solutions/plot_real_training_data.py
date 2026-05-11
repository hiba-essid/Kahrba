#!/usr/bin/env python3
"""
Script pour générer des courbes à partir de VRAIES données d'entraînement
Charge les données depuis les logs de Keras/TensorFlow
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np


def load_keras_history(history_file='training_history.json'):
    """
    Charger les données d'entraînement depuis un fichier JSON Keras
    
    Args:
        history_file: Chemin du fichier JSON contenant l'historique
    
    Returns:
        Dictionnaire avec les données ou None si erreur
    """
    if not os.path.exists(history_file):
        print(f"❌ Fichier non trouvé: {history_file}")
        print("\nAttendez un fichier JSON contenant les clés:")
        print("  - 'loss' ou 'train_loss'")
        print("  - 'val_loss' ou 'validation_loss'")
        print("  - 'accuracy' ou 'train_accuracy'")
        print("  - 'val_accuracy' ou 'validation_accuracy'")
        return None
    
    print(f"📂 Chargement des données: {history_file}")
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Normaliser les noms des clés
    data = {}
    
    # Loss
    if 'loss' in history:
        data['training_loss'] = history['loss']
    elif 'train_loss' in history:
        data['training_loss'] = history['train_loss']
    else:
        print("❌ Clé 'loss' ou 'train_loss' non trouvée")
        return None
    
    if 'val_loss' in history:
        data['validation_loss'] = history['val_loss']
    elif 'validation_loss' in history:
        data['validation_loss'] = history['validation_loss']
    else:
        print("❌ Clé 'val_loss' ou 'validation_loss' non trouvée")
        return None
    
    # Accuracy
    if 'accuracy' in history:
        data['training_accuracy'] = history['accuracy']
    elif 'train_accuracy' in history:
        data['training_accuracy'] = history['train_accuracy']
    else:
        print("❌ Clé 'accuracy' ou 'train_accuracy' non trouvée")
        return None
    
    if 'val_accuracy' in history:
        data['validation_accuracy'] = history['val_accuracy']
    elif 'validation_accuracy' in history:
        data['validation_accuracy'] = history['validation_accuracy']
    else:
        print("❌ Clé 'val_accuracy' ou 'validation_accuracy' non trouvée")
        return None
    
    # Epochs
    num_epochs = len(data['training_loss'])
    data['epochs'] = list(range(1, num_epochs + 1))
    
    print(f"✅ Données chargées avec succès!")
    print(f"   - Nombre d'epochs: {num_epochs}")
    print(f"   - Loss training: {len(data['training_loss'])} valeurs")
    print(f"   - Loss validation: {len(data['validation_loss'])} valeurs")
    print(f"   - Accuracy training: {len(data['training_accuracy'])} valeurs")
    print(f"   - Accuracy validation: {len(data['validation_accuracy'])} valeurs\n")
    
    return data


def plot_real_data_loss(data, output_dir='results'):
    """Tracer la courbe Loss avec les vraies données"""
    epochs = data['epochs']
    training_loss = data['training_loss']
    validation_loss = data['validation_loss']
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_loss, label='Training', linewidth=2.5, 
             marker='o', markersize=4, color='#3498db')
    plt.plot(epochs, validation_loss, label='Validation', linewidth=2.5, 
             marker='s', markersize=4, color='#e74c3c')
    
    plt.title('Model Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_path = os.path.join(output_dir, 'model_loss_real.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"✅ Loss curve saved: {loss_path}")
    plt.close()


def plot_real_data_accuracy(data, output_dir='results'):
    """Tracer la courbe Accuracy avec les vraies données"""
    epochs = data['epochs']
    training_accuracy = data['training_accuracy']
    validation_accuracy = data['validation_accuracy']
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_accuracy, label='Training', linewidth=2.5, 
             marker='o', markersize=4, color='#2ecc71')
    plt.plot(epochs, validation_accuracy, label='Validation', linewidth=2.5, 
             marker='s', markersize=4, color='#f39c12')
    
    plt.title('Model Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    plt.tight_layout()
    accuracy_path = os.path.join(output_dir, 'model_accuracy_real.png')
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    print(f"✅ Accuracy curve saved: {accuracy_path}")
    plt.close()


def plot_real_data_combined(data, output_dir='results'):
    """Tracer Loss et Accuracy ensemble"""
    epochs = data['epochs']
    training_loss = data['training_loss']
    validation_loss = data['validation_loss']
    training_accuracy = data['training_accuracy']
    validation_accuracy = data['validation_accuracy']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Loss
    ax1.plot(epochs, training_loss, label='Training', linewidth=2.5, 
             marker='o', markersize=5, color='#3498db')
    ax1.plot(epochs, validation_loss, label='Validation', linewidth=2.5, 
             marker='s', markersize=5, color='#e74c3c')
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, training_accuracy, label='Training', linewidth=2.5, 
             marker='o', markersize=5, color='#2ecc71')
    ax2.plot(epochs, validation_accuracy, label='Validation', linewidth=2.5, 
             marker='s', markersize=5, color='#f39c12')
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    combined_path = os.path.join(output_dir, 'training_history_real.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✅ Combined curves saved: {combined_path}")
    plt.close()


def analyze_training(data):
    """Analyser l'entraînement et donner des insights"""
    print("\n" + "="*70)
    print("TRAINING ANALYSIS")
    print("="*70 + "\n")
    
    num_epochs = len(data['epochs'])
    final_train_loss = data['training_loss'][-1]
    final_val_loss = data['validation_loss'][-1]
    final_train_acc = data['training_accuracy'][-1]
    final_val_acc = data['validation_accuracy'][-1]
    
    initial_train_loss = data['training_loss'][0]
    initial_val_loss = data['validation_loss'][0]
    
    print(f"📊 RÉSUMÉ GÉNÉRAL")
    print(f"  • Total epochs: {num_epochs}")
    print(f"  • Accuracy finale (train): {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"  • Accuracy finale (val): {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"  • Loss finale (train): {final_train_loss:.4f}")
    print(f"  • Loss finale (val): {final_val_loss:.4f}\n")
    
    print(f"📈 PROGRESSION")
    loss_reduction = initial_train_loss - final_train_loss
    acc_gain = final_train_acc - data['training_accuracy'][0]
    print(f"  • Loss réduite: {loss_reduction:.4f}")
    print(f"  • Accuracy gagnée: {acc_gain:.4f}\n")
    
    # Analyser overfitting
    print(f"⚙️  ANALYSE OVERFITTING")
    loss_diff = abs(final_val_loss - final_train_loss)
    acc_diff = abs(final_val_acc - final_train_acc)
    
    print(f"  • Différence Loss (train-val): {loss_diff:.4f}")
    print(f"  • Différence Accuracy (train-val): {acc_diff:.4f}")
    
    if loss_diff > 0.2 or acc_diff > 0.1:
        print(f"\n  ⚠️  OVERFITTING DÉTECTÉ!")
        print(f"     - Le modèle apprend trop bien les données d'entraînement")
        print(f"     - Généralisation faible sur les données de validation")
        print(f"     - Solutions: Augmenter dropout, réduire epochs, ou plus de données")
    elif loss_diff < 0.05 and acc_diff < 0.02:
        print(f"\n  ✅ BONNE GÉNÉRALISATION!")
        print(f"     - Le modèle apprend bien et généralise correctement")
        print(f"     - Train et validation sont proches")
    else:
        print(f"\n  ✓ Généralisation normale")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main function"""
    
    print("\n" + "="*70)
    print("COURBES D'ENTRAÎNEMENT À PARTIR DE VRAIES DONNÉES")
    print("="*70 + "\n")
    
    # Configuration
    HISTORY_FILE = 'training_history.json'  # À adapter
    OUTPUT_DIR = 'results'
    
    # Créer le dossier
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Charger les données
    data = load_keras_history(HISTORY_FILE)
    
    if data is None:
        print("❌ Impossible de charger les données.")
        print("\nCréez un fichier JSON avec cette structure:")
        print("""
{
  "loss": [2.5, 2.3, 2.1, ...],
  "val_loss": [2.2, 2.0, 1.8, ...],
  "accuracy": [0.1, 0.15, 0.25, ...],
  "val_accuracy": [0.12, 0.18, 0.28, ...]
}
        """)
        return
    
    # Générer les courbes
    print("📈 Génération des graphiques...\n")
    plot_real_data_loss(data, OUTPUT_DIR)
    plot_real_data_accuracy(data, OUTPUT_DIR)
    plot_real_data_combined(data, OUTPUT_DIR)
    
    # Analyser
    analyze_training(data)
    
    print("✅ TERMINÉ!")
    print(f"\n📊 Fichiers générés dans: {OUTPUT_DIR}/")
    print("  - model_loss_real.png")
    print("  - model_accuracy_real.png")
    print("  - training_history_real.png\n")


if __name__ == '__main__':
    main()
