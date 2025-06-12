import cv2
import numpy as np

def get_lab_warmth_score_normalized(graffiti_segment_rgb, chroma_threshold=15):
    """
    Calculates a numerical warmth score for a graffiti segment using the L*a*b* color space,
    normalized to a 0-1 range.

    The 'a*' component of L*a*b* represents the green-red axis.
    After normalization:
    - A score closer to 0 indicates cooler (more green) colors.
    - A score closer to 0.5 indicates neutral colors.
    - A score closer to 1 indicates warmer (more red) colors.

    Args:
        graffiti_segment_rgb (numpy.ndarray): An RGB image (or mask) of the graffiti.
                                             Assumes non-graffiti pixels are black (0,0,0) or transparent.
        chroma_threshold (int): Minimum chroma (sqrt(a*^2 + b*^2)) a pixel must have to be
                                included in the warmth score calculation. This helps to
                                exclude near-neutral (gray, black, white) pixels which
                                might have 'a*' close to zero and skew the average.
                                A value like 10-20 is often reasonable.

    Returns:
        float: The normalized average 'a*' value (0-1) of chromatically significant pixels
               in the graffiti segment. Returns 0.5 if no significant pixels are found
               (representing a neutral absence of color).
    """
    if graffiti_segment_rgb is None or graffiti_segment_rgb.size == 0:
        print("Warning: Empty graffiti segment provided for L*a*b* warmth score.")
        return 0.5 # Return neutral if no pixels are found

    # Convert RGB to L*a*b*
    graffiti_segment_bgr = cv2.cvtColor(graffiti_segment_rgb, cv2.COLOR_RGB2BGR)
    graffiti_lab = cv2.cvtColor(graffiti_segment_bgr, cv2.COLOR_BGR2LAB)

    # Extract the a* and b* components.
    a_star = graffiti_lab[:, :, 1].astype(np.float32)
    b_star = graffiti_lab[:, :, 2].astype(np.float32)

    # Calculate chroma (measure of color intensity/saturation)
    # Using centered values for chroma calculation for accuracy.
    centered_a_star = a_star - 128
    centered_b_star = b_star - 128
    chroma = np.sqrt(centered_a_star**2 + centered_b_star**2)

    # Create a mask for pixels that are not black (masked-out background)
    # And also above the chroma threshold
    l_channel = graffiti_lab[:, :, 0] # L channel (0-255)

    relevant_pixels_mask = (l_channel > 5) & (chroma > chroma_threshold)

    significant_a_stars = a_star[relevant_pixels_mask]

    if significant_a_stars.size == 0:
        return 0.5 # Return 0.5 (neutral) if no significant colored pixels found

    # Calculate the mean of the significant a* values
    mean_a_star = np.mean(significant_a_stars)

    # Normalize the mean a* value to the 0-1 range
    normalized_warmth_score = mean_a_star / 255.0

    return normalized_warmth_score

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy image representing a graffiti segment
    dummy_graffiti_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Add some warm colors (red, yellow)
    dummy_graffiti_image[10:30, 10:30] = [255, 0, 0] # Red
    dummy_graffiti_image[10:30, 40:60] = [255, 255, 0] # Yellow

    # Add some cool colors (blue, green)
    dummy_graffiti_image[50:70, 10:30] = [0, 0, 255] # Blue
    dummy_graffiti_image[50:70, 40:60] = [0, 255, 0] # Green

    # Add some neutral colors (gray, black, white)
    dummy_graffiti_image[80:90, 10:20] = [128, 128, 128] # Gray
    dummy_graffiti_image[80:90, 30:40] = [0, 0, 0]       # Black (will be effectively ignored by L channel > 5)
    dummy_graffiti_image[80:90, 50:60] = [255, 255, 255] # White (will likely have low chroma and be ignored)

    print("\nAnalyzing dummy graffiti L*a*b* warmth score (normalized 0-1):")
    warmth_score_normalized = get_lab_warmth_score_normalized(dummy_graffiti_image)
    print(f"L*a*b* Warmth Score: {warmth_score_normalized:.4f}")
    # Expected: Slightly above 0.5 because of the red/yellow patches


    # --- Corrected Example for a mostly cool image ---
    mostly_cool_image = np.zeros((50, 50, 3), dtype=np.uint8)
    mostly_cool_image[:, :] = [0, 200, 0] # A strong green color
    mostly_cool_image[10:20, 10:20] = [255,255,255] # some white
    print("\nAnalyzing mostly cool (green) image L*a*b* warmth score (normalized 0-1):")
    warmth_score_cool_normalized = get_lab_warmth_score_normalized(mostly_cool_image)
    print(f"L*a*b* Warmth Score: {warmth_score_cool_normalized:.4f}")
    # Expected: Significantly below 0.5


    # Example with a neutral image
    neutral_image = np.full((50, 50, 3), 128, dtype=np.uint8) # Mid-gray
    print("\nAnalyzing neutral image L*a*b* warmth score (normalized 0-1):")
    warmth_score_neutral_normalized = get_lab_warmth_score_normalized(neutral_image)
    print(f"L*a*b* Warmth Score: {warmth_score_neutral_normalized:.4f}")
    # Expected: Close to 0.5