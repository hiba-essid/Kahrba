#!/usr/bin/env python3
"""
Traffic Sign Recognition Robot - Main Application
==================================================
Complete application for autonomous robot car with traffic sign recognition.

This is the main entry point for the robot car system. It integrates:
- Camera capture
- Image preprocessing
- TFLite model inference
- Motor control based on sign recognition

Hardware Requirements:
    - Raspberry Pi 3
    - Pi Camera or USB Webcam
    - L298N Motor Driver
    - DC Motors
    - Trained TFLite model

Usage:
    python main.py --model traffic_sign_model.tflite

Press Ctrl+C to stop the robot.
"""

import os
import sys
import time
import argparse
import threading
import json

# Camera and image processing
import cv2
import numpy as np

# TFLite inference
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("Warning: tflite_runtime not available. Using TensorFlow.")
    try:
        import tensorflow as tf
    except ImportError:
        print("Error: Neither tflite_runtime nor tensorflow is available!")
        sys.exit(1)

# Motor controller
from motor_controller import MotorController, ON_RASPBERRY_PI


class TrafficSignRecognizer:
    """
    Traffic sign recognition using TFLite model
    """
    
    def __init__(self, model_path):
        """
        Initialize the TFLite model
        
        Args:
            model_path: Path to .tflite model file
        """
        print(f"Loading model from: {model_path}")
        
        # Load TFLite model
        if TFLITE_AVAILABLE:
            self.interpreter = tflite.Interpreter(model_path=model_path)
        else:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
        
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.input_size = (self.input_shape[1], self.input_shape[2])
        self.input_dtype = self.input_details[0]['dtype']
        
        # Get quantization parameters if available
        self.quantization = self.input_details[0].get('quantization_parameters', None)
        self.is_quantized = self.quantization is not None and len(self.quantization.get('scales', [])) > 0
        
        if self.is_quantized:
            self.input_scale = self.quantization['scales'][0]
            self.input_zero_point = self.quantization['zero_points'][0]
            output_quant = self.output_details[0].get('quantization_parameters', {})
            self.output_scale = output_quant.get('scales', [1.0])[0]
            self.output_zero_point = output_quant.get('zero_points', [0])[0]
        
        print(f"Model loaded successfully!")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Input type: {self.input_dtype}")
        print(f"  Quantized: {self.is_quantized}")
        print(f"  Output classes: {self.output_details[0]['shape'][1]}")
    
    def preprocess(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb, self.input_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        # Quantize if needed
        if self.is_quantized:
            batched = (batched / self.input_scale + self.input_zero_point).astype(np.int8)
        
        return batched
    
    def predict(self, image):
        """
        Run inference on image
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            tuple: (predicted_class, confidence, inference_time_ms)
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        
        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dequantize if needed
        if self.is_quantized:
            output = (output.astype(np.float32) - self.output_zero_point) * self.output_scale
        
        # Get prediction
        predicted_class = int(np.argmax(output[0]))
        confidence = float(output[0][predicted_class])
        
        return predicted_class, confidence, inference_time


class TrafficSignRobot:
    """
    Main robot application class
    """
    
    # Default sign-to-action mapping
    # Modify based on your model's class definitions
    DEFAULT_SIGN_ACTIONS = {
        0: ('stop', 0),           # Stop sign
        1: ('forward', 30),       # Speed limit 20 -> go slow
        2: ('forward', 50),       # Speed limit 30 -> medium
        3: ('forward', 70),       # Speed limit 50 -> normal
        4: ('forward', 80),       # Speed limit 60 -> fast
        5: ('forward', 80),       # Speed limit 70
        6: ('forward', 80),       # Speed limit 80
        7: ('forward', 40),       # Speed limit 80 lifted -> slow
        8: ('forward', 50),       # Speed limit 100
        9: ('forward', 30),       # Speed limit 120 -> slow
        10: ('stop', 0),          # No passing
        11: ('stop', 0),          # No passing for trucks
        12: ('yield', 20),        # Priority at intersection
        13: ('yield', 20),        # Yield right-of-way
        14: ('stop', 0),          # Stop
        15: ('stop', 0),          # No vehicles
        16: ('stop', 0),          # No trucks
        17: ('stop', 0),          # No entry
        18: ('caution', 20),      # General caution
        19: ('caution', 20),      # Dangerous left curve
        20: ('caution', 20),      # Dangerous right curve
        21: ('caution', 20),      # Double curve
        22: ('caution', 20),      # Bumpy road
        23: ('caution', 20),      # Slippery road
        24: ('caution', 20),      # Road narrows
        25: ('caution', 20),      # Construction
        26: ('caution', 20),      # Traffic signal
        27: ('caution', 20),      # Pedestrians
        28: ('caution', 20),      # Children crossing
        29: ('caution', 20),      # Bicycles
        30: ('caution', 20),      # Ice/snow
        31: ('caution', 20),      # Wild animals
        32: ('forward', 40),      # End of limits
        33: ('turn_right', 40),   # Turn right ahead
        34: ('turn_left', 40),    # Turn left ahead
        35: ('forward', 40),      # Ahead only
        36: ('forward', 40),      # Go straight or right
        37: ('forward', 40),      # Go straight or left
        38: ('turn_right', 40),   # Keep right
        39: ('turn_left', 40),    # Keep left
        40: ('yield', 20),        # Roundabout
        41: ('stop', 0),          # End of no passing
        42: ('stop', 0),          # End of no passing for trucks
    }
    
    def __init__(self, model_path, camera_index=0, config_path=None):
        """
        Initialize robot system
        
        Args:
            model_path: Path to TFLite model
            camera_index: Camera device index (default: 0)
            config_path: Optional path to configuration file
        """
        print("="*60)
        print("Traffic Sign Recognition Robot")
        print("="*60)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize model
        self.recognizer = TrafficSignRecognizer(model_path)
        
        # Initialize motor controller
        self.motor = MotorController(
            left_pins=tuple(self.config.get('left_motor_pins', (17, 18, 27))),
            right_pins=tuple(self.config.get('right_motor_pins', (22, 23, 24)))
        )
        
        # Initialize camera
        self.camera_index = camera_index
        self.camera = None
        self._init_camera()
        
        # Sign actions (use default or load from config)
        self.sign_actions = self.config.get('sign_actions', self.DEFAULT_SIGN_ACTIONS)
        
        # Runtime parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.default_speed = self.config.get('default_speed', 30)
        self.turn_duration = self.config.get('turn_duration', 0.5)
        
        # State variables
        self.running = False
        self.current_sign = None
        self.current_confidence = 0.0
        self.fps = 0.0
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        print("\nRobot initialized successfully!")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Default speed: {self.default_speed}")
        print(f"  Running on Raspberry Pi: {ON_RASPBERRY_PI}")
        print("\nPress Ctrl+C to stop.\n")
    
    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _init_camera(self):
        """Initialize camera"""
        print(f"Initializing camera (index {self.camera_index})...")
        
        self.camera = cv2.VideoCapture(self.camera_index)
        
        # Set camera resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify camera is working
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError(f"Failed to initialize camera at index {self.camera_index}")
        
        print(f"Camera initialized: {frame.shape[1]}x{frame.shape[0]}")
    
    def extract_roi(self, frame):
        """
        Extract region of interest for sign detection
        
        Args:
            frame: Full camera frame
            
        Returns:
            ROI image for sign detection
        """
        h, w = frame.shape[:2]
        
        # Extract upper-center region where signs typically appear
        # Adjust these values based on camera mounting position
        roi = frame[int(h*0.1):int(h*0.6), int(w*0.2):int(w*0.8)]
        
        return roi
    
    def execute_action(self, action, speed):
        """
        Execute motor action
        
        Args:
            action: Action type ('stop', 'forward', 'backward', 'turn_left', 'turn_right', etc.)
            speed: Speed parameter (0-100)
        """
        if action == 'stop':
            self.motor.stop()
        elif action == 'forward':
            self.motor.forward(speed)
        elif action == 'backward':
            self.motor.backward(speed)
        elif action == 'turn_left':
            self.motor.turn_left(speed, duration=self.turn_duration)
        elif action == 'turn_right':
            self.motor.turn_right(speed, duration=self.turn_duration)
        elif action == 'yield':
            # Slow down
            self.motor.forward(speed)
        elif action == 'caution':
            # Slow down and proceed with caution
            self.motor.forward(speed)
        else:
            # Unknown action, default to stop
            self.motor.stop()
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Camera frame
            
        Returns:
            Processed frame with overlay
        """
        # Extract ROI
        roi = self.extract_roi(frame)
        
        # Run inference
        predicted_class, confidence, inference_time = self.recognizer.predict(roi)
        
        # Update state
        self.current_sign = predicted_class
        self.current_confidence = confidence
        
        # Get action for predicted sign
        if confidence >= self.confidence_threshold:
            if predicted_class in self.sign_actions:
                action, speed = self.sign_actions[predicted_class]
                self.execute_action(action, speed)
            else:
                # Unknown sign, use default
                self.motor.forward(self.default_speed)
        else:
            # Low confidence, use default behavior
            self.motor.forward(self.default_speed)
        
        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        
        # Draw overlay on frame
        self._draw_overlay(frame, predicted_class, confidence, inference_time)
        
        return frame
    
    def _draw_overlay(self, frame, predicted_class, confidence, inference_time):
        """Draw information overlay on frame"""
        h, w = frame.shape[:2]
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (int(w*0.2), int(h*0.1)), 
                     (int(w*0.8), int(h*0.6)), (0, 255, 0), 2)
        
        # Draw prediction info
        text_lines = [
            f"Sign: {predicted_class}",
            f"Confidence: {confidence:.2f}",
            f"Inference: {inference_time:.1f}ms",
            f"FPS: {self.fps:.1f}"
        ]
        
        y_offset = 30
        for i, text in enumerate(text_lines):
            cv2.putText(frame, text, (10, y_offset + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def run(self, display=True):
        """
        Main control loop
        
        Args:
            display: Whether to display video window
        """
        self.running = True
        
        print("Starting main control loop...")
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Warning: Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display if enabled
                if display and ON_RASPBERRY_PI:
                    cv2.imshow('Traffic Sign Robot', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        except KeyboardInterrupt:
            print("\nStopping robot...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the robot and cleanup"""
        self.running = False
        
        # Stop motors
        self.motor.stop()
        self.motor.cleanup()
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        print("Robot stopped and cleaned up.")


def main():
    parser = argparse.ArgumentParser(description='Traffic Sign Recognition Robot')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to TFLite model file')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration JSON file')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable video display window')
    args = parser.parse_args()
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Create and run robot
    robot = TrafficSignRobot(
        model_path=args.model,
        camera_index=args.camera,
        config_path=args.config
    )
    
    robot.run(display=not args.no_display)


if __name__ == "__main__":
    main()
