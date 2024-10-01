echo off

python -m pip install --upgrade pip
python -m pip install --upgrade setuptools pip wheel
python -m pip install nvidia-pyindex
python -m pip install nvidia-cuda-runtime-cu12

pip install -r requirements.txt --no-cache-dir
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning -f https://data.pyg.org/whl/torch-2.3.0+cu121.html --no-cache-dir
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu121.html --no-cache-dir
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html --no-cache-dir
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html --no-cache-dir
pip install dgl -f https://data.pyg.org/whl/torch-2.3.0+cu121.html --no-cache-dir

goto(){
# Linux code here
uname -o
}

goto $@
exit

:(){
rem Windows script here
echo %OS%

python -m pip install --config-settings="--global-option=build_ext" --config-settings="--global-option=-IC:\Program Files\Graphviz\include" --config-settings="--global-option=-LC:\Program Files\Graphviz\lib"  pygraphviz
exit
