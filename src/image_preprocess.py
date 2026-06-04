from PIL import Image


def resize_by_long_edge(image: Image.Image, max_long_edge: int) -> Image.Image:
    """
    Resize image if long edge exceeds max_long_edge, keeping aspect ratio.
    """
    max_long_edge = max(256, int(max_long_edge))
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
