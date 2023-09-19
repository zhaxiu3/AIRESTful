from utils.uploader import download_file
from api_shap_e import ShapE

model_shape = ShapE()

def predict(request_id, prompt, image_url):
    input_image = f"/home/ubuntu/inputs/{request_id}_input.png"
    download_file(image_url, input_image)
    return model_shape.predict(request_id, input_image, prompt)