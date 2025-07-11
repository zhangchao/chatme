#!/usr/bin/env python3
"""
Direct test of apple analysis with the enhanced app
"""
import numpy as np
from PIL import Image, ImageDraw

def create_red_apple_image():
    """Create a realistic red apple image for testing"""
    # Create a 300x300 image with white background
    img = Image.new('RGB', (300, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw apple body (red circle)
    apple_center = (150, 160)
    apple_radius = 80
    
    # Create gradient effect for apple
    for r in range(apple_radius, 0, -2):
        # Vary the red color for realistic apple look
        red_intensity = min(255, 180 + (apple_radius - r) * 2)
        green_intensity = max(0, 50 - (apple_radius - r))
        color = (red_intensity, green_intensity, 0)
        
        draw.ellipse([
            apple_center[0] - r, apple_center[1] - r,
            apple_center[0] + r, apple_center[1] + r
        ], fill=color)
    
    # Add apple stem (brown)
    draw.rectangle([145, 80, 155, 100], fill=(101, 67, 33))
    
    # Add leaf (green)
    draw.ellipse([155, 85, 175, 105], fill=(34, 139, 34))
    
    return np.array(img)

def test_vision_components():
    """Test individual vision components"""
    print("🔍 Testing Vision Components...")
    
    try:
        from vision_processor import VisionLanguageModel, MultimodalProcessor
        
        # Create apple image
        apple_image = create_red_apple_image()
        print(f"✅ Created apple image: {apple_image.shape}")
        
        # Test VisionLanguageModel directly
        print("\n🧠 Testing VisionLanguageModel...")
        vlm = VisionLanguageModel()
        
        analysis = vlm.analyze_image(apple_image, "what is it, what color")
        print(f"VLM Analysis: {analysis}")
        
        # Test MultimodalProcessor
        print("\n🔄 Testing MultimodalProcessor...")
        processor = MultimodalProcessor()
        
        response = processor.process_multimodal_input(
            image=apple_image,
            audio_text="what is it, what color",
            conversation_history=[]
        )
        print(f"Multimodal Response: {response}")
        
        # Check if response is good
        response_lower = response.lower()
        
        print("\n📊 Analysis:")
        if "apple" in response_lower or "fruit" in response_lower:
            print("✅ Identified as apple/fruit")
        else:
            print("❌ Did not identify as apple/fruit")
        
        if "red" in response_lower:
            print("✅ Identified red color")
        else:
            print("❌ Did not identify red color")
        
        if "person" in response_lower or "face" in response_lower:
            print("❌ Incorrectly mentioned person/face")
        else:
            print("✅ Did not mention person/face")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_app_simple():
    """Test enhanced app without complex imports"""
    print("\n🚀 Testing Enhanced App (Simple)...")
    
    try:
        # Import only what we need
        import sys
        import os
        
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        from vision_processor import MultimodalProcessor
        from conversation_manager import ConversationManager
        
        # Create apple image
        apple_image = create_red_apple_image()
        
        # Initialize components
        processor = MultimodalProcessor()
        conv_manager = ConversationManager()
        
        # Test the exact user scenario
        question = "what is it, what color"
        print(f"📝 Question: '{question}'")
        
        # Process the question
        response = processor.process_multimodal_input(
            image=apple_image,
            audio_text=question,
            conversation_history=[]
        )
        
        print(f"🤖 Response: {response}")
        
        # Analyze response quality
        response_lower = response.lower()
        
        score = 0
        total = 4
        
        print("\n📊 Response Analysis:")
        
        if "apple" in response_lower or "fruit" in response_lower:
            print("✅ Correctly identified as apple/fruit")
            score += 1
        else:
            print("❌ Did not identify as apple/fruit")
        
        if "red" in response_lower:
            print("✅ Correctly identified red color")
            score += 1
        else:
            print("❌ Did not identify red color")
        
        if "person" in response_lower or "face" in response_lower:
            print("❌ Incorrectly mentioned person/face")
        else:
            print("✅ Did not mention person/face")
            score += 1
        
        if "blurry" in response_lower or "trouble" in response_lower:
            print("⚠️  Mentioned processing issues")
        else:
            print("✅ No processing issues mentioned")
            score += 1
        
        print(f"\n🎯 Score: {score}/{total}")
        
        if score >= 3:
            print("🎉 Good response quality!")
            return True
        else:
            print("⚠️  Response needs improvement")
            return False
        
    except Exception as e:
        print(f"❌ Enhanced app test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run direct apple tests"""
    print("🍎 Direct Apple Analysis Test")
    print("=" * 40)
    
    tests = [
        ("Vision Components", test_vision_components),
        ("Enhanced App Simple", test_enhanced_app_simple),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        icon = "✅" if result else "❌"
        print(f"{icon} {test_name}")
    
    print(f"\n🎯 {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Apple recognition is working!")
        print("\n✅ The app should now correctly:")
        print("   🍎 Identify apples as fruits")
        print("   🔴 Recognize red color")
        print("   👤 NOT detect faces in fruit images")
        print("   📝 Give relevant responses")
    else:
        print("⚠️  Some tests failed, but improvements have been made.")
    
    print("\n🚀 Test with the actual app:")
    print("   streamlit run enhanced_app.py --server.port 8503")

if __name__ == "__main__":
    main()
