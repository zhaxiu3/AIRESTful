## 部署过程
### 1. Shap-E 部署
```bash
# 安装anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
bash ./Anaconda3-2023.07-2-Linux-x86_64.sh -sbf
source ~/anaconda3/bin/activate
conda init
conda config --set auto_activate_base True

# 安装pytorch3d
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub

## Demos and examples
conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python

## Tests/Linting
pip install black usort flake8 flake8-bugbear flake8-comprehensions

## Anaconda Cloud
conda install pytorch3d -c pytorch3d


# clone shap-e
git clone https://github.com/openai/shap-e.git
cd shap-e
pip install -e .

## 开启 jupyter notebook
jupyter notebook --ip=0.0.0.0 --port=8888

# 可以使用如下指令查看当前可用的运行环境:
# jupyter notebook list 
```

### 2. install.sh
```bash
source ./install.sh
```

### 3. 开启server
```bash
source ./run.sh
```
```bash
cd data
python -m http.server 9222
```