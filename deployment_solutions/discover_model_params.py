#!/usr/bin/env python3
"""
Découvrir automatiquement les paramètres de votre modèle TFLite
INPUT_SIZE, Input Shape, et autres détails essentiels
"""

from tflite_runtime.interpreter import Interpreter
import os


def analyze_model(model_path):
    """
    Analyser un modèle TFLite et afficher ses paramètres
    
    Args:
        model_path: Chemin vers le fichier .tflite
    """
    print("\n" + "="*70)
    print(f"ANALYSE DU MODÈLE: {model_path}")
    print("="*70 + "\n")
    
    if not os.path.exists(model_path):
        print(f"❌ ERREUR: Fichier non trouvé: {model_path}")
        return
    
    try:
        # Charger le modèle
        print(f"📂 Chargement du modèle...")
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("✅ Modèle chargé avec succès!\n")
        
        # Obtenir les détails d'entrée et sortie
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Afficher les informations d'entrée
        print("=" * 70)
        print("📥 INFORMATIONS D'ENTRÉE (INPUT)")
        print("=" * 70)
        
        input_shape = input_details[0]['shape']
        print(f"Input Shape (forme complète): {input_shape}")
        print(f"Input dtype (type de données): {input_details[0]['dtype']}\n")
        
        # Décomposer la forme
        if len(input_shape) == 4:
            batch_size = input_shape[0]
            height = input_shape[1]
            width = input_shape[2]
            channels = input_shape[3]
            
            print("Détails décomposés:")
            print(f"  • Batch size (nombre d'images): {batch_size}")
            print(f"  • Hauteur: {height} pixels")
            print(f"  • Largeur: {width} pixels")
            print(f"  • Canaux: {channels} (RGB)")
            
            # Déterminer INPUT_SIZE
            if height == width:
                input_size = height
                print(f"\n🎯 INPUT_SIZE à utiliser dans le script: {input_size}")
            else:
                print(f"\n⚠️  ATTENTION: Hauteur ({height}) ≠ Largeur ({width})")
                print(f"   Les images doivent être redimensionnées à: {height}×{width}")
        
        # Afficher les informations de sortie
        print("\n" + "=" * 70)
        print("📤 INFORMATIONS DE SORTIE (OUTPUT)")
        print("=" * 70)
        
        output_shape = output_details[0]['shape']
        print(f"Output Shape: {output_shape}")
        print(f"Output dtype: {output_details[0]['dtype']}\n")
        
        # Déterminer le nombre de classes
        if len(output_shape) == 2:
            num_samples = output_shape[0]
            num_classes = output_shape[1]
            print(f"Détails décomposés:")
            print(f"  • Nombre d'échantillons: {num_samples}")
            print(f"  • Nombre de classes: {num_classes}")
            print(f"\n🏷️  Votre modèle peut classifier en {num_classes} classes")
        
        # Résumé pour copier-coller
        print("\n" + "=" * 70)
        print("✅ RÉSUMÉ - À COPIER DANS analyze_model_curves.py")
        print("=" * 70)
        
        if len(input_shape) == 4 and height == width:
            print(f"""
# === CONFIGURATION ===
MODEL_PATH = '{os.path.basename(model_path)}'
TEST_IMAGES_DIR = 'test_images'
OUTPUT_DIR = 'results'
INPUT_SIZE = {input_size}  # ← Votre valeur!
""")
        
        print("\n" + "=" * 70)
        print("Fichier de configuration:")
        print(f"  analyze_model_curves.py")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"❌ ERREUR lors du chargement du modèle: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    
    print("\n" + "🔍 DÉTECTEUR DE PARAMÈTRES DE MODÈLE TFLITE 🔍".center(70))
    
    # Chercher tous les fichiers .tflite
    tflite_files = [f for f in os.listdir('.') if f.endswith('.tflite')]
    
    if not tflite_files:
        print("\n❌ Aucun fichier .tflite trouvé dans le dossier courant!")
        print("\nPlacez vos fichiers .tflite dans le même dossier que ce script.")
        return
    
    print(f"\n📋 {len(tflite_files)} fichier(s) .tflite trouvé(s):\n")
    for i, f in enumerate(tflite_files, 1):
        print(f"  {i}. {f}")
    
    # Analyser chaque modèle
    print("\n" + "-"*70 + "\n")
    
    for model_file in tflite_files:
        analyze_model(model_file)
        print()
    
    # Afficher les instructions finales
    print("=" * 70)
    print("📝 INSTRUCTIONS FINALES")
    print("=" * 70)
    print("""
1. ✅ Notez la valeur INPUT_SIZE ci-dessus
2. ✅ Ouvrez le fichier: analyze_model_curves.py
3. ✅ Trouvez la section "=== CONFIGURATION ==="
4. ✅ Changez INPUT_SIZE à la valeur trouvée
5. ✅ Créez un dossier "test_images" avec des sous-dossiers par classe
6. ✅ Lancez: python analyze_model_curves.py

Besoin d'aide? Consultez le fichier: INPUT_SIZE_EXPLANATION.md
""")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
