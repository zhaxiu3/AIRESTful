from time import sleep
import requests
import json

UPLOAD = "https://api.bytescale.com/v2/accounts/kW15bfx/uploads/binary"
bearerToken = "public_kW15bfx4Xt3FH2TJC8Tpr3oRc1FP"

def upload(binaryData):
    try:
        headers = {
            "Authorization": "Bearer " + bearerToken,
            "Content-Type": "image/png"
        }
        files = {"file": binaryData}
        response = requests.post(UPLOAD, headers=headers, files=files)
        response.raise_for_status()
        response_json = json.loads(response.text)
        url = response_json.get("fileUrl")
        print(url)
        return url
    except requests.exceptions.RequestException as err:
        print(str(err))
        return None

def upload_file(imagefile):
    with open(imagefile, "rb") as file:
        imagebytes = file.read()

    return upload(imagebytes)

def download_file(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status() #检查响应状态
        with open(filename, "wb") as file:
            file.write(response.content)
        print("Downloaded " + filename)
    except requests.exceptions.RequestException as err:
        print(str(err))
