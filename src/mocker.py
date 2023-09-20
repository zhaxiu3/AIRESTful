from utils.uploader import download_file
from api_shap_e import ShapE

# settings
input_dir = "./data/inputs"

model_shape = ShapE()

def predict(request_id, prompt, image_url):
    input_image = f"{input_dir}/{request_id}_input.png"
    download_file(image_url, input_image)
    return model_shape.predict(request_id, input_image, prompt)