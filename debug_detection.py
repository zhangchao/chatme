#!/usr/bin/env python3
"""
Debug the detection system
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

def debug_detection():
    """Debug the detection system step by step"""
    print("🔍 Debugging Detection System...")
    
    try:
        from vision_processor import VisionLanguageModel
        
        # Create test cat image
        cat_image = create_cat_image()
        print(f"✅ Created cat image: {cat_image.shape}")
        
        # Test VLM directly
        vlm = VisionLanguageModel()
        
        # Debug each step
        print("\n🧪 Testing _detect_objects_advanced...")
        detected_objects = vlm._detect_objects_advanced(cat_image)
        print(f"Detected objects: {detected_objects}")
        
        print("\n🧪 Testing _detect_animals...")
        animals = vlm._detect_animals(cat_image)
        print(f"Detected animals: {animals}")
        
        print("\n🧪 Testing _detect_fruits...")
        fruits = vlm._detect_fruits(cat_image)
        print(f"Detected fruits: {fruits}")
        
        print("\n🧪 Testing _detect_common_objects...")
        objects = vlm._detect_common_objects(cat_image)
        print(f"Detected common objects: {objects}")
        
        # Test color analysis
        avg_color = np.mean(cat_image, axis=(0, 1))
        r, g, b = avg_color
        print(f"\n🎨 Average color: R={r:.1f}, G={g:.1f}, B={b:.1f}")
        
        colors = vlm._analyze_colors(r, g, b)
        print(f"Analyzed colors: {colors}")
        
        # Test cat color detection
        has_cat_colors = vlm._has_cat_like_colors(r, g, b)
        print(f"Has cat-like colors: {has_cat_colors}")
        
        # Test texture detection
        try:
            import cv2
            gray = cv2.cvtColor(cat_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (cat_image.shape[0] * cat_image.shape[1])
            print(f"Edge density (fur texture): {edge_density:.3f}")
            
            # Test circle detection for eyes
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=5, maxRadius=50)
            has_circles = circles is not None
            print(f"Has circular features (eyes): {has_circles}")
            if has_circles:
                print(f"Number of circles detected: {len(circles[0])}")
                
        except Exception as e:
            print(f"OpenCV analysis failed: {e}")
        
        # Test multiple animal detection
        multiple_animals = vlm._detect_multiple_animals(cat_image)
        print(f"Multiple animals detected: {multiple_animals}")
        
        # Test round shape detection
        is_round = vlm._detect_round_shape(cat_image)
        print(f"Is round shape: {is_round}")
        
        # Final test
        print("\n🧪 Testing full _mock_analyze_image...")
        result = vlm._mock_analyze_image(cat_image, "what do you see")
        print(f"Final result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run debug"""
    print("🔍 Detection System Debug")
    print("=" * 40)
    
    debug_detection()

if __name__ == "__main__":
    main()
