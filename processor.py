import numpy as np
from PIL import Image
import base64
from io import BytesIO

def pre_process(input):

    image_data = base64.b64decode(input)
    # Open the image using PIL
    original_image = Image.open(BytesIO(image_data)).convert("L")
    resized_image = original_image.resize((8, 8))
    image_array = np.array(resized_image)
    normalized_image = image_array / 255.0 * 16
    feature_vector = normalized_image.flatten()
    return feature_vector

# Post-process function: Converts model predictions to human-readable format
def post_process(input):
    predictions=input
    label_map = {i: str(i) for i in range(10)}  # Map digits to string labels
    readable_label = label_map[predictions[0]]
    return f"Predicted Digit: {readable_label}"