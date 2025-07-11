#!/usr/bin/env python3
"""
Test script for cat detection
"""
import numpy as np
from PIL import Image, ImageDraw

def create_cat_image():
    """Create a realistic cat image for testing"""
    # Create a 400x300 image with background
    img = Image.new('RGB', (400, 300), color=(50, 80, 50))  # Green background
    draw = ImageDraw.Draw(img)
    
    # Draw first cat (orange/ginger)
    # Cat body
    draw.ellipse([50, 150, 150, 220], fill=(200, 120, 60))  # Orange body
    
    # Cat head
    draw.ellipse([80, 100, 140, 160], fill=(210, 130, 70))  # Orange head
    
    # Cat ears
    draw.polygon([(85, 110), (95, 90), (105, 110)], fill=(190, 110, 50))
    draw.polygon([(115, 110), (125, 90), (135, 110)], fill=(190, 110, 50))
    
    # Cat eyes
    draw.ellipse([90, 120, 100, 130], fill=(0, 0, 0))  # Left eye
    draw.ellipse([120, 120, 130, 130], fill=(0, 0, 0))  # Right eye
    
    # Cat stripes (tabby pattern)
    for i in range(3):
        y = 160 + i * 15
        draw.line([(60, y), (140, y)], fill=(150, 80, 30), width=3)
    
    # Draw second cat (gray)
    # Cat body
    draw.ellipse([250, 140, 350, 210], fill=(120, 120, 120))  # Gray body
    
    # Cat head  
    draw.ellipse([280, 90, 340, 150], fill=(130, 130, 130))  # Gray head
    
    # Cat ears
    draw.polygon([(285, 100), (295, 80), (305, 100)], fill=(110, 110, 110))
    draw.polygon([(315, 100), (325, 80), (335, 100)], fill=(110, 110, 110))
    
    # Cat eyes
    draw.ellipse([290, 110, 300, 120], fill=(0, 0, 0))  # Left eye
    draw.ellipse([320, 110, 330, 120], fill=(0, 0, 0))  # Right eye
    
    return np.array(img)

def test_cat_detection():
    """Test cat detection with the enhanced vision processor"""
    print("🐱 Testing Cat Detection...")
    
    try:
        from vision_processor import VisionLanguageModel, MultimodalProcessor
        
        # Create test cat image
        cat_image = create_cat_image()
        print(f"✅ Created cat image: {cat_image.shape}")
        
        # Test vision language model
        vlm = VisionLanguageModel()
        
        # Test different questions about cats
        test_questions = [
            "What do you see?",
            "What is it?",
            "what do you see",
            "Describe this image",
            "What animals are in this image?"
        ]
        
        print("\n🔍 Testing vision analysis...")
        
        for question in test_questions:
            print(f"\n📝 Question: '{question}'")
            
            # Test the mock analyzer
            response = vlm._mock_analyze_image(cat_image, question)
            print(f"🤖 Response: {response}")
            
            # Check if response correctly identifies cats
            response_lower = response.lower()
            
            success_indicators = []
            
            if "cat" in response_lower:
                success_indicators.append("✅ Correctly identified as cat")
            else:
                success_indicators.append("❌ Did not identify as cat")
            
            if "two" in response_lower or "2" in response_lower:
                success_indicators.append("✅ Detected multiple cats")
            elif "cats" in response_lower:
                success_indicators.append("✅ Detected multiple cats (plural)")
            else:
                success_indicators.append("⚠️  Did not detect multiple cats")
            
            if "orange" in response_lower or "ginger" in response_lower:
                success_indicators.append("✅ Identified orange/ginger coloring")
            else:
                success_indicators.append("⚠️  Did not identify orange coloring")
            
            if "gray" in response_lower or "grey" in response_lower:
                success_indicators.append("✅ Identified gray coloring")
            else:
                success_indicators.append("⚠️  Did not identify gray coloring")
            
            if "fruit" in response_lower:
                success_indicators.append("❌ Incorrectly mentioned fruit")
            else:
                success_indicators.append("✅ Did not mention fruit")
            
            print("   Analysis:")
            for indicator in success_indicators:
                print(f"     {indicator}")
        
        # Test with multimodal processor
        print("\n🔄 Testing MultimodalProcessor...")
        processor = MultimodalProcessor()
        
        response = processor.process_multimodal_input(
            image=cat_image,
            audio_text="what do you see",
            conversation_history=[]
        )
        print(f"🤖 Multimodal Response: {response}")
        
        # Final analysis
        response_lower = response.lower()
        
        print("\n📊 Final Analysis:")
        
        if "cat" in response_lower:
            print("✅ Successfully identified cats in multimodal response")
            return True
        else:
            print("❌ Failed to identify cats in multimodal response")
            return False
        
    except Exception as e:
        print(f"❌ Cat detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run cat detection tests"""
    print("🐱 Cat Detection Test")
    print("=" * 40)
    
    success = test_cat_detection()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Cat detection is working!")
        print("✅ The app should now correctly identify cats instead of fruits")
        print("\n🚀 Test with the actual app:")
        print("   streamlit run enhanced_app.py --server.port 8503")
        print("   Upload a cat image and ask 'what do you see'")
    else:
        print("⚠️  Cat detection needs more work")
        print("❌ The app may still have issues with animal recognition")

if __name__ == "__main__":
    main()
