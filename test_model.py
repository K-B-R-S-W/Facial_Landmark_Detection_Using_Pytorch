"""
Test script to debug HRNetV2 model loading and forward pass
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def test_model_architecture():
    """Test if the model architecture is correct"""
    print("Testing HRNetV2 model architecture...")
    
    try:
        from src.landmark_detector import HRNetV2LandmarkDetector, HRNetV2_W32
        
        # Test backbone
        print("1. Testing HRNetV2 backbone...")
        backbone = HRNetV2_W32(num_classes=1000)
        print(f"✅ Backbone created successfully")
        
        # Test full model
        print("2. Testing full landmark detector...")
        model = HRNetV2LandmarkDetector(num_landmarks=68)
        print(f"✅ Full model created successfully")
        
        # Test forward pass with dummy input
        print("3. Testing forward pass...")
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Test backbone features with debugging
        with torch.no_grad():
            print("   3.1 Testing backbone feature extraction...")
            try:
                features = backbone(dummy_input, return_features=True)
                print(f"✅ Backbone forward pass successful")
                print(f"   Feature shapes: {[f.shape for f in features]}")
            except Exception as e:
                print(f"❌ Backbone forward pass failed: {e}")
                print("   Trying without return_features...")
                try:
                    _ = backbone(dummy_input, return_features=False)
                    print("✅ Backbone forward pass works without return_features")
                    # Let's check the transition layer sizes
                    print(f"   Transition1 length: {len(backbone.transition1)}")
                    print(f"   Transition2 length: {len(backbone.transition2)}")
                    print(f"   Transition3 length: {len(backbone.transition3)}")
                    raise e
                except Exception as e2:
                    print(f"❌ Backbone completely broken: {e2}")
                    raise e2
            
            # Test full model
            print("   3.2 Testing full landmark detector...")
            landmarks = model(dummy_input)
            print(f"✅ Full model forward pass successful")
            print(f"   Landmark shape: {landmarks.shape}")
            print(f"   Expected: (1, 68, 2)")
            
        return True
        
    except Exception as e:
        print(f"❌ Architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pretrained_loading(model_path):
    """Test pretrained weight loading"""
    print(f"\nTesting pretrained weight loading from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Load pretrained weights
        print("1. Loading pretrained weights...")
        pretrained_dict = torch.load(model_path, map_location='cpu')
        print(f"✅ Pretrained weights loaded")
        print(f"   Number of keys: {len(pretrained_dict.keys())}")
        
        # Show some key names
        keys = list(pretrained_dict.keys())
        print(f"   First 5 keys: {keys[:5]}")
        print(f"   Last 5 keys: {keys[-5:]}")
        
        # Test loading into model
        print("2. Testing weight loading into model...")
        from src.landmark_detector import HRNetV2_W32
        
        backbone = HRNetV2_W32(num_classes=1000)
        model_dict = backbone.state_dict()
        
        # Count matching keys
        matching_keys = 0
        shape_mismatches = 0
        
        for k, v in pretrained_dict.items():
            key = k.replace('module.', '')
            if key in model_dict:
                if v.shape == model_dict[key].shape:
                    matching_keys += 1
                else:
                    shape_mismatches += 1
        
        print(f"✅ Weight compatibility check complete")
        print(f"   Matching keys: {matching_keys}")
        print(f"   Shape mismatches: {shape_mismatches}")
        print(f"   Model keys: {len(model_dict)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pretrained loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_landmark_detector_init(model_path):
    """Test the full LandmarkDetector class initialization"""
    print(f"\nTesting LandmarkDetector initialization...")
    
    try:
        from src.landmark_detector import LandmarkDetector
        
        detector = LandmarkDetector(model_path=model_path, device='cpu')
        print(f"✅ LandmarkDetector initialized successfully")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect_landmarks(dummy_image)
        
        print(f"✅ Landmark detection test successful")
        if result is not None and 'landmarks' in result:
            landmarks = result['landmarks']
            print(f"   Landmarks shape: {landmarks.shape}")
            print(f"   Landmark range: [{landmarks.min():.2f}, {landmarks.max():.2f}]")
        else:
            print(f"   Result: {type(result)}")
            if hasattr(result, 'shape'):
                print(f"   Result shape: {result.shape}")
                print(f"   Result range: [{result.min():.2f}, {result.max():.2f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ LandmarkDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("HRNetV2 Model Testing Script")
    print("=" * 60)
    
    # Test model architecture first
    arch_test = test_model_architecture()
    
    if not arch_test:
        print("❌ Architecture test failed. Cannot continue.")
        return
    
    # Test pretrained loading
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "models", 
        "hrnetv2_w32_imagenet_pretrained.pth"
    )
    
    pretrained_test = test_pretrained_loading(model_path)
    
    if not pretrained_test:
        print("❌ Pretrained loading test failed.")
        print("   You can still run with randomly initialized weights.")
    
    # Test full detector
    detector_test = test_landmark_detector_init(model_path)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print(f"Architecture Test: {'✅ PASS' if arch_test else '❌ FAIL'}")
    print(f"Pretrained Test:   {'✅ PASS' if pretrained_test else '❌ FAIL'}")
    print(f"Detector Test:     {'✅ PASS' if detector_test else '❌ FAIL'}")
    print("=" * 60)
    
    if arch_test and detector_test:
        print("🎉 Model is ready for use!")
        if not pretrained_test:
            print("⚠️  Note: Using randomly initialized weights (no pretrained)")
    else:
        print("❌ Model has issues that need to be resolved")

if __name__ == "__main__":
    main()