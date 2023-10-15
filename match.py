import numpy as np
import os
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
import torch

def process_images(
    image_path_0: str, 
    image_path_1: str, 
    feature_type: str = 'superpoint', 
    max_num_keypoints: int = 2048
) -> np.ndarray:
    """Process two images and return matched points as an array.
    
    Args:
        image_path_0 (str): Path to the first image.
        image_path_1 (str): Path to the second image.
        feature_type (str, optional): Type of feature to use. Defaults to 'superpoint'.
        max_num_keypoints (int, optional): Maximum number of keypoints to use. Defaults to 2048.
    
    Returns:
        np.ndarray: Array of matched points.
    """
    if not os.path.exists(image_path_0) or not os.path.exists(image_path_1):
        raise FileNotFoundError("One or both image paths are invalid.")

    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available on this system.")

    if feature_type not in ['superpoint', 'disk']:
        raise ValueError(f"Invalid feature_type: {feature_type}")

    if feature_type == 'superpoint':
        extractor = SuperPoint(max_num_keypoints).eval().cuda()
        matcher = LightGlue(features='superpoint').eval().cuda()
    elif feature_type == 'disk':
        extractor = DISK(max_num_keypoints).eval().cuda()
        matcher = LightGlue(features='disk').eval().cuda()

    image0 = load_image(image_path_0).cuda()
    image1 = load_image(image_path_1).cuda()

    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    matches = matches01['matches']
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    return np.array([points0, points1])

if __name__ == "__main__":
    # Test the function locally
    image_path_0 = "test_image_0.jpg"
    image_path_1 = "test_image_1.jpg"
    feature_type = "superpoint"
    max_num_keypoints = 2048
    result = process_images(image_path_0, image_path_1, feature_type, max_num_keypoints)
    print(result)