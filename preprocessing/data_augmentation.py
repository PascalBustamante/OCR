import cv2
import numpy as np

"""
To DO:
    - Rotation: Rotate the image by a certain angle (e.g., -10° to +10°) to simulate different orientations of text.

    -Scaling: Resize the image to a slightly larger or smaller size to simulate variations in text size and aspect ratios.

    -Translation: Shift the image horizontally and/or vertically to simulate different placements of text within the image.

    -Brightness and Contrast Adjustment: Randomly adjust the brightness and contrast of the image to simulate variations in lighting conditions.

    -Noise Addition: Add random noise (e.g., Gaussian noise) to the image to simulate imperfections in scanning or image acquisition.

    -Blur: Apply blur filters (e.g., Gaussian blur) to simulate out-of-focus or blurry text.

    -Shearing: Apply a shearing transformation to the image to simulate text distortion caused by perspective.

    -Elastic Distortion: Apply elastic deformation to the image to simulate warping and distortion.

    -Background Removal/Replacement: Remove or replace the background of the image to simulate different backgrounds in real-world scenarios.

    -Color Space Conversion: Convert the image to different color spaces (e.g., grayscale, HSV) to introduce variations in color representation.

    -Text Synthesis: Add synthetic text to the image to increase the diversity of text patterns and fonts.

    -Flip: Flip the image horizontally or vertically to simulate different orientations of text.
"""
"""
The input to these methods is assumed to have passed through preprocessing
"""
def rotation(src):
    pass