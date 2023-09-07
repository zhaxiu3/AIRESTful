import json
import subprocess
import sys
import shlex

MODEL_SETTINGS = {
    "img2img": {
        "dir": "/home/ubuntu/stablediffusion",
        "template": "python scripts/img2img.py --prompt \"{prompt}\" --init-img \"{image}\" --strength 0.8 --ckpt ./models/v2-1_768-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference-v.yaml"
    }
}

def invoke_model(model, prompt, image=None):
    setting = MODEL_SETTINGS[model]
    formated_template = setting["template"].format(prompt=prompt, image=image)
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
    
    id = data["id"]
    model = data["model"]
    prompt = data["prompt"]
    image_url = data["image"]

    # 调用模型
    print(invoke_model(model, prompt, image_url))