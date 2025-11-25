"""
CLIP Image Comparison using Replicate API
Lightweight alternative for deployment on memory-constrained platforms
"""

import replicate
import numpy as np
from scipy.spatial.distance import cosine
import base64
import io


class CLIPComparator:
    """
    Image comparison using Replicate's hosted CLIP model.
    Same interface as the local version for drop-in replacement.
    """

    def __init__(self, option1_path, option2_path):
        """
        Initialize with image paths.
        Pre-computes embeddings for reference images at startup.
        """
        self.opt1_path = option1_path
        self.opt2_path = option2_path

        # Pre-compute embeddings for reference images (2 API calls at startup)
        print("Loading reference image embeddings via Replicate API...")
        self.feat_opt1 = self._get_embedding_from_file(option1_path)
        self.feat_opt2 = self._get_embedding_from_file(option2_path)
        print("Reference embeddings loaded successfully!")

    def _get_embedding_from_file(self, image_path):
        """Get CLIP embedding from a local file path"""
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
            ext = image_path.split('.')[-1].lower()
            mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
            data_uri = f"data:{mime};base64,{data}"

        output = replicate.run(
            "openai/clip",
            input={"image": data_uri}
        )
        return np.array(output["embedding"])

    def _get_embedding_from_pil(self, pil_image):
        """Get CLIP embedding from a PIL Image object"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        data = base64.b64encode(buffer.getvalue()).decode()
        data_uri = f"data:image/jpeg;base64,{data}"

        output = replicate.run(
            "openai/clip",
            input={"image": data_uri}
        )
        return np.array(output["embedding"])

    def compare(self, image_a_stream, amplification_factor=20):
        """
        Compare uploaded image against the two reference options.

        Args:
            image_a_stream: PIL Image object to compare
            amplification_factor: Higher values = more extreme probabilities

        Returns:
            dict with probabilities and match information
        """
        feat_a = self._get_embedding_from_pil(image_a_stream)

        # Calculate cosine similarity (1 - cosine_distance)
        similarity1 = 1 - cosine(feat_a, self.feat_opt1)
        similarity2 = 1 - cosine(feat_a, self.feat_opt2)

        # Convert to probabilities using exponential amplification
        exp_sim1 = np.exp(similarity1 * amplification_factor)
        exp_sim2 = np.exp(similarity2 * amplification_factor)
        total = exp_sim1 + exp_sim2

        prob1 = exp_sim1 / total
        prob2 = exp_sim2 / total

        best_match = 1 if prob1 > prob2 else 2

        prob_diff = abs(prob1 - prob2)
        if prob_diff > 0.4:
            confidence = "VERY HIGH"
        elif prob_diff > 0.25:
            confidence = "HIGH"
        elif prob_diff > 0.15:
            confidence = "MODERATE"
        else:
            confidence = "LOW"

        return {
            'option1_probability': prob1,
            'option2_probability': prob2,
            'best_match': best_match,
            'confidence': confidence,
            'similarity1': similarity1,
            'similarity2': similarity2
        }
