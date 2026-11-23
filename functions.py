import gradio as gr
import numpy as np


EQUALIZATION = "Histogram equalization"
MATCHING = "Histogram matching"


def function_hw5(input_image, mode=EQUALIZATION, reference_image=None):
    """Apply a manual histogram transform to grayscale levels or HLS lightness."""
    def get_lightness(image, label):
        image = np.asarray(image)
        if image.size == 0 or image.dtype != np.uint8:
            raise gr.Error(f"{label} must be a nonempty 8-bit image.")
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            # HLS lightness is the mean of the minimum and maximum RGB channels.
            channel_sum = image.max(axis=2).astype(np.uint16) + image.min(axis=2)
            return ((channel_sum + 1) // 2).astype(np.uint8)
        raise gr.Error(f"{label} must be a grayscale or RGB image.")

    if input_image is None:
        raise gr.Error("Upload an input image.")
    source = np.asarray(input_image)
    lightness = get_lightness(source, "The input image")

    if mode == EQUALIZATION:
        counts = np.bincount(lightness.ravel(), minlength=256)
        cumulative = counts.cumsum()
        first_count = counts[counts > 0][0]
        if first_count == lightness.size:
            return source.copy()
        # Map the darkest occupied level to 0 and the brightest to 255.
        lookup = np.rint((cumulative - first_count) * 255.0 / (lightness.size - first_count))
        enhanced = lookup.clip(0, 255).astype(np.uint8)[lightness]
    elif mode == MATCHING:
        if reference_image is None:
            raise gr.Error("Upload a reference image for histogram matching.")
        reference = get_lightness(reference_image, "The reference image")
        source_counts = np.bincount(lightness.ravel(), minlength=256)
        reference_counts = np.bincount(reference.ravel(), minlength=256)
        source_cdf = source_counts.cumsum() / lightness.size
        reference_levels = np.flatnonzero(reference_counts)
        reference_cdf = reference_counts[reference_levels].cumsum() / reference.size

        # Find the closest cumulative probability among occupied reference levels.
        right = np.searchsorted(reference_cdf, source_cdf)
        left = np.maximum(right - 1, 0)
        right = np.minimum(right, reference_cdf.size - 1)
        use_left = source_cdf - reference_cdf[left] <= reference_cdf[right] - source_cdf
        lookup = reference_levels[np.where(use_left, left, right)].astype(np.uint8)
        enhanced = lookup[lightness]
    else:
        raise gr.Error("Select a valid processing method.")

    if source.ndim == 2:
        return enhanced

    import cv2

    # Change lightness while keeping the input hue and saturation in HLS space.
    hls = cv2.cvtColor(source.astype(np.float32) / 255.0, cv2.COLOR_RGB2HLS)
    changed = enhanced != lightness
    hls[..., 1] = np.where(changed, enhanced.astype(np.float32) / 255.0, hls[..., 1])
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
    return np.rint(rgb.clip(0, 1) * 255.0).astype(np.uint8)
