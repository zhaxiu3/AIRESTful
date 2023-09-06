import os
import signal
from flask import Flask, request
from flask_restful import Resource, Api
from queue import Queue
import threading
import uuid

app = Flask(__name__)
api = Api(app)
model_queue = Queue()
request_status = {}

def process_model_requests():
    while True:
        data = model_queue.get()
        id = data["id"]
        model = data["model"]
        prompt = data["prompt"]
        image_url = data["image"]
        request_status[id] = {"status":"processing"}
        # 处理模型请求的逻辑
        model_queue.task_done()
        request_status[id] = {"status":"completed"}

class ModelResource(Resource):
    def post(self):
        data = request.get_json()
        id = str(uuid.uuid4())
        data["id"] = id
        #将模型请求放入队列
        model_queue.put(data)
        #更新模型请求状态  
        request_status[id] = {"status":"pending"}
        
        return {"message": {"id": data["id"]}}

class RequestStatusResource(Resource):
    def get(self, request_id):
        if request_id in request_status:
            return request_status[request_id]
        else:
            return {"status": "not found"}

api.add_resource(ModelResource, '/model')
api.add_resource(RequestStatusResource, '/request/<string:request_id>')

if __name__ == '__main__':
    # 启动线程
    model_thread = threading.Thread(target=process_model_requests)
    model_thread.start()

    app.run(debug=True, host='0.0.0.0')

    if model_thread and model_thread.is_alive():
        os.kill(model_thread.ident, signal.SIGTERM)