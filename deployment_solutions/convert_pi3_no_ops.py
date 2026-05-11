import tensorflow as tf
import os
import json
import zipfile
import tempfile

"""
Raspberry Pi 3 Minimal Ops Conversion
======================================
Converts WITHOUT optimizations that create newer op versions.
This should create ONLY FULLY_CONNECTED version 10 or lower.
"""

print("="*70)
print("RASPBERRY PI 3 - MINIMAL OPS CONVERSION (NO OPTIMIZATIONS)")
print("="*70)

# Load the Keras model with config patching
print("\nLoading model...")
try:
    keras_path = "C:\\Users\\MSI\\Desktop\\PFA2\\traffic_sign_model_package_v2 (1)\\traffic_sign_model_package_v2\\best_model.keras"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(keras_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        config_path = os.path.join(tmpdir, 'config.json')
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        def remove_quant_config(obj):
            if isinstance(obj, dict):
                if 'quantization_config' in obj:
                    del obj['quantization_config']
                for value in obj.values():
                    remove_quant_config(value)
            elif isinstance(obj, list):
                for item in obj:
                    remove_quant_config(item)
        
        remove_quant_config(config_data)
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        patched_keras_path = "best_model_patched.keras"
        with zipfile.ZipFile(patched_keras_path, 'w') as zip_out:
            for file in os.listdir(tmpdir):
                file_path = os.path.join(tmpdir, file)
                zip_out.write(file_path, arcname=file)
        
        model = tf.keras.models.load_model(patched_keras_path, compile=False)
        os.remove(patched_keras_path)
    
    print("✓ Model loaded")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Convert WITHOUT optimizations (to avoid version 12 ops)
print("\nConverting with MINIMAL settings (no optimizations)...")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # CRITICAL: Restrict to ONLY basic ops - NO optimizations
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    
    # DO NOT USE optimizations - they create newer op versions!
    # converter.optimizations = []  # Explicitly NO optimizations
    
    # Disable all experimental features
    converter.experimental_enable_resource_variables = False
    converter.experimental_enable_per_channel_quantization = False
    
    # Set minimal inference type (keep as float for compatibility)
    # converter.inference_input_type = tf.float32
    # converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    print("✓ Conversion complete")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Save the model
print("\nSaving model (no optimizations)...")
try:
    output_path = 'C:\\Users\\MSI\\Desktop\\PFA2\\traffic_sign_model_package_v2 (1)\\traffic_sign_model_package_v2\\traffic_sign_model_pi3_minimal.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    file_size = os.path.getsize(output_path)
    print(f"✓ Saved: {output_path}")
    print(f"  Size: {file_size / (1024*1024):.2f} MB")
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. Copy to Raspberry Pi:")
    print(f"   scp {output_path} pi@YOUR_PI_IP:~/")
    print("\n2. Update predict_image_pi3.py:")
    print("   MODEL_PATH = 'traffic_sign_model_pi3_minimal.tflite'")
    print("\n3. Test on Pi:")
    print("   python3 predict_image_pi3.py")
    print("\n" + "="*70)
    
except Exception as e:
    print(f"✗ Error saving: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
