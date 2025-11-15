"""
CLIP-Only Image Comparison
Uses only the CLIP model for maximum accuracy with dancing/similar images
Fast and highly accurate - perfect for your use case
"""

import torch
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("❌ ERROR: CLIP not installed!")
    print("Install with:")
    print("  pip install ftfy regex")
    print("  pip install git+https://github.com/openai/CLIP.git")
    raise ImportError("CLIP is required for this script")


class CLIPComparator:
    """
    High-accuracy image comparison using ONLY CLIP model
    Best for distinguishing between very similar images
    """
    
    def __init__(self, image_a_stream, option1_path, option2_path):
        """
        Initialize with image paths
        
        Args:
            image_a_path: Reference image
            option1_path: First option
            option2_path: Second option
        """
        self.img_a_stream = image_a_stream
        self.opt1_path = option1_path
        self.opt2_path = option2_path
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        
        # Load CLIP model
        print("📦 Loading CLIP model (ViT-B/32)...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        print("✅ CLIP model loaded successfully!\n")
    
    def extract_features(self, image_path):
        """
        Extract CLIP visual features from an image
        
        Args:
            image_path: Path to image
            
        Returns:
            numpy array of normalized features
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            # Normalize features (important for cosine similarity)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().flatten()
    
    def extract_features_stream(self, image):
        """
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            # Normalize features (important for cosine similarity)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().flatten()
    
    def compare(self, amplification_factor=20):
        """
        Compare images and return probabilities
        
        Args:
            amplification_factor: Higher values make probabilities more extreme (default: 20)
                                 Use 10-15 for more conservative probabilities
                                 Use 20-30 for more confident probabilities
        
        Returns:
            dict with probabilities and match information
        """
        print("🔍 Extracting CLIP features from images...")
        
        # Extract features for all images
        feat_a = self.extract_features_stream(self.image_a_stream)
        feat_opt1 = self.extract_features(self.opt1_path)
        feat_opt2 = self.extract_features(self.opt2_path)
        
        print("📊 Computing similarities...")
        
        # Calculate cosine similarity (1 - cosine_distance)
        # Values closer to 1.0 mean more similar
        similarity1 = 1 - cosine(feat_a, feat_opt1)
        similarity2 = 1 - cosine(feat_a, feat_opt2)
        
        # Convert similarities to probabilities using exponential
        # This amplifies small differences
        exp_sim1 = np.exp(similarity1 * amplification_factor)
        exp_sim2 = np.exp(similarity2 * amplification_factor)
        total = exp_sim1 + exp_sim2
        
        prob1 = exp_sim1 / total
        prob2 = exp_sim2 / total
        
        # Determine best match
        best_match = 1 if prob1 > prob2 else 2
        
        # Calculate confidence level
        prob_diff = abs(prob1 - prob2)
        if prob_diff > 0.4:
            confidence = "VERY HIGH"
        elif prob_diff > 0.25:
            confidence = "HIGH"
        elif prob_diff > 0.15:
            confidence = "MODERATE"
        else:
            confidence = "LOW"
        
        # Print results
        print("\n" + "=" * 70)
        print("🎯 CLIP COMPARISON RESULTS")
        print("=" * 70)
        print(f"Raw Similarities:")
        print(f"  Option 1: {similarity1:.4f}")
        print(f"  Option 2: {similarity2:.4f}")
        print(f"\nProbabilities:")
        print(f"  Option 1: {prob1:.2%}")
        print(f"  Option 2: {prob2:.2%}")
        print(f"\n✅ BEST MATCH: Option {best_match}")
        print(f"📊 Confidence: {confidence} ({max(prob1, prob2):.1%})")
        print("=" * 70)
        
        return {
            'option1_probability': prob1,
            'option2_probability': prob2,
            'best_match': best_match,
            'confidence': confidence,
            'similarity1': similarity1,
            'similarity2': similarity2
        }


def compare_with_clip(image_a, option1, option2, amplification=20):
    """
    Quick function to compare images using CLIP
    
    Args:
        image_a: Path to reference image
        option1: Path to first option
        option2: Path to second option
        amplification: How much to amplify differences (default: 20)
    
    Returns:
        dict with keys: option1_probability, option2_probability, best_match, confidence
    
    Example:
        result = compare_with_clip("dancer.jpg", "person_a.jpg", "person_b.jpg")
        print(f"Match: Option {result['best_match']}")
        print(f"Confidence: {result['confidence']}")
    """
    comparator = CLIPComparator(image_a, option1, option2)
    return comparator.compare(amplification_factor=amplification)


# Example usage
if __name__ == "__main__":
    print("\n🎭 CLIP-ONLY IMAGE COMPARISON")
    print("=" * 70)
    print("High accuracy comparison using state-of-the-art CLIP model")
    print("=" * 70)
    
    # Replace with your actual image paths
    image_a = "Riley2.jpg"
    option_1 = "Riley3.jpg"
    option_2 = "Vadin1.jpg"
    
    try:
        # Run comparison
        result = compare_with_clip(image_a, option_1, option_2)
        
        # Print quick summary
        print(f"\n🔥 QUICK ANSWER:")
        print(f"   Match: Option {result['best_match']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Probability: {max(result['option1_probability'], result['option2_probability']):.1%}")

        #team = "team1"
        #if result['best_match'] != option_1:
        #    team = "team2"

        #return team


        # Access detailed results
        #print(f"\n📈 DETAILED PROBABILITIES:")
        #print(f"   Option 1: {result['option1_probability']:.2%}")
        #print(f"   Option 2: {result['option2_probability']:.2%}")
        
    except FileNotFoundError:
        pass
        #print("\n⚠️  Example images not found!")
        #print("Replace the image paths above with your actual images:")
        #print(f"  - {image_a}")
        #print(f"  - {option_1}")
        #print(f"  - {option_2}")