#!bin/bash 

apt update && apt upgrade  -y && apt install vim git curl gcc g++ curl wget 


cd outfit_viton

pip install -r requirements.txt

git clone --recursive https://github.com/open-mmlab/mmfashion.git

cd mmfashion/

python setup.py install

cd ..



