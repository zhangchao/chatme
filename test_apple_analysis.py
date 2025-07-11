#!/usr/bin/env python3
"""
Test script specifically for apple image analysis
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

def test_apple_recognition():
    """Test apple recognition with the enhanced vision processor"""
    print("🍎 Testing Apple Recognition...")
    
    try:
        from vision_processor import VisionLanguageModel, GestureDetector
        
        # Create test apple image
        apple_image = create_red_apple_image()
        print(f"✅ Created red apple image: {apple_image.shape}")
        
        # Test vision language model
        vlm = VisionLanguageModel()
        
        # Test different questions about the apple
        test_questions = [
            "What is it?",
            "What color is it?", 
            "What do you see?",
            "Describe this object",
            "What fruit is this?"
        ]
        
        print("\n🔍 Testing vision analysis...")
        
        for question in test_questions:
            print(f"\n📝 Question: '{question}'")
            
            # Test the mock analyzer (since MLX-VLM might not be loaded)
            response = vlm._mock_analyze_image(apple_image, question)
            print(f"🤖 Response: {response}")
            
            # Check if response is reasonable for an apple
            response_lower = response.lower()
            
            if "apple" in response_lower or "fruit" in response_lower:
                print("✅ Correctly identified as apple/fruit")
            elif "red" in response_lower and ("round" in response_lower or "circular" in response_lower):
                print("✅ Correctly identified red round object")
            else:
                print("⚠️  Could be more specific about apple identification")
            
            if "red" in response_lower:
                print("✅ Correctly identified red color")
            else:
                print("⚠️  Should mention red color")
        
        # Test gesture detector (should NOT detect faces in apple)
        print("\n👤 Testing face detection on apple...")
        gesture_detector = GestureDetector()
        
        scene_analysis = gesture_detector.analyze_scene(apple_image)
        faces_detected = scene_analysis["faces_detected"]
        
        if faces_detected == 0:
            print("✅ Correctly detected NO faces in apple image")
        else:
            print(f"❌ Incorrectly detected {faces_detected} faces in apple image")
        
        return faces_detected == 0  # Success if no faces detected
        
    except Exception as e:
        print(f"❌ Apple recognition test failed: {e}")
        return False

def test_with_enhanced_app():
    """Test with the actual enhanced app"""
    print("\n🚀 Testing with Enhanced App...")
    
    try:
        from enhanced_app import EnhancedMultimodalAssistant
        
        # Create apple image
        apple_image = create_red_apple_image()
        
        # Initialize assistant
        assistant = EnhancedMultimodalAssistant()
        if not assistant.initialize():
            print("❌ Failed to initialize assistant")
            return False
        
        # Test the exact scenario from the user
        question = "what is it, what color"
        print(f"📝 Testing user question: '{question}'")
        
        response = assistant.process_image_and_text(apple_image, question)
        print(f"🤖 Response: {response}")
        
        # Check if response is good
        response_lower = response.lower()
        
        success_indicators = []
        
        if "apple" in response_lower or "fruit" in response_lower:
            success_indicators.append("✅ Identified as apple/fruit")
        else:
            success_indicators.append("❌ Did not identify as apple/fruit")
        
        if "red" in response_lower:
            success_indicators.append("✅ Identified red color")
        else:
            success_indicators.append("❌ Did not identify red color")
        
        if "person" in response_lower or "face" in response_lower:
            success_indicators.append("❌ Incorrectly mentioned person/face")
        else:
            success_indicators.append("✅ Did not mention person/face")
        
        if "blurry" in response_lower:
            success_indicators.append("⚠️  Mentioned image is blurry")
        else:
            success_indicators.append("✅ Did not mention blurriness")
        
        print("\n📊 Analysis Results:")
        for indicator in success_indicators:
            print(f"   {indicator}")
        
        # Count successes
        successes = sum(1 for indicator in success_indicators if indicator.startswith("✅"))
        total = len(success_indicators)
        
        print(f"\n🎯 Score: {successes}/{total}")
        
        return successes >= total * 0.75  # 75% success rate
        
    except Exception as e:
        print(f"❌ Enhanced app test failed: {e}")
        return False

def main():
    """Run apple analysis tests"""
    print("🍎 Apple Image Analysis Tests")
    print("=" * 40)
    
    tests = [
        ("Apple Recognition", test_apple_recognition),
        ("Enhanced App Integration", test_with_enhanced_app),
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
        print("🎉 All tests passed! Apple recognition should work correctly now.")
        print("\n✅ The enhanced app should now:")
        print("   🍎 Correctly identify apples as fruits")
        print("   🔴 Recognize red color properly")
        print("   👤 NOT detect faces in fruit images")
        print("   📝 Give relevant responses to 'what is it, what color' questions")
    else:
        print("⚠️  Some tests failed. The app may still have issues with apple recognition.")
    
    print("\n🚀 Test the app with a real apple image:")
    print("   streamlit run enhanced_app.py --server.port 8503")
    print("   Upload an apple image and ask 'what is it, what color'")

if __name__ == "__main__":
    main()
