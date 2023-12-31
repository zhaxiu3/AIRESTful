import json
import subprocess
import sys
import shlex

from utils.uploader import download_file

MODEL_SETTINGS = {
    "img2img": {
        "dir": "/home/ubuntu/stablediffusion",
        "template": "python scripts/img2img.py --request-id \"{request_id}\" --prompt \"{prompt}\" --init-img \"{image}\" --strength 0.8 --ckpt ./models/v2-1_768-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference-v.yaml"
    },
    "one2345": {
        "dir": "/content/AIRESTful/One-2-3-45",
        "template":"python run.py --request-id \"{request_id}\" --img_path \"{image}\" --half_precision --output_format \".obj\" "
    }
}

def invoke_model(model, request_id, prompt, image=None):
    setting = MODEL_SETTINGS[model]
    formated_template = setting["template"].format(prompt=prompt, image=image, request_id=request_id)
    command = shlex.split(formated_template)
    print(command)
    try:
        output = subprocess.run(command, cwd=setting["dir"],capture_output=True, text=True)
    except Exception as err:
        output = str(err)
    return output

if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) != 2:
        print("Usage: python mocker.py json")
        sys.exit(1)
    
    data = json.loads(sys.argv[1])
    
    request_id = data["id"]
    model = data["model"]
    prompt = data["prompt"]
    image_url = data["image"]
    input_image = f"{MODEL_SETTINGS[model]['dir']}/{request_id}_input.png"
    #如果image_url不是None,则下载图片
    if image_url:
        download_file(image_url, input_image)

    # 调用模型
    print(invoke_model(model, request_id, prompt, input_image))