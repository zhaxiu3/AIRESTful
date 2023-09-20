from utils.uploader import download_file
from api_shap_e import ShapE
from settings import input_dir

model_shape = ShapE()

def predict(request_id: str, prompt: str, image_url: str):
    input_image = f"{input_dir}/{request_id}_input.png"
    download_file(image_url, input_image)
    return model_shape.predict(request_id, input_image, prompt)