#!bin/bash 

apt update && apt upgrade && apt install vim git curl gcc g++ curl wget 

git clone https://github.com/trinanjan12/Image-Based-Virtual-Try-on-Network-from-Unpaired-Data

mv Image-Based-Virtual-Try-on-Network-from-Unpaired-Data outfit_viton

cd outfit_viton

pip install -r requirements.txt

Image-Based-Virtual-Try-on-Network-from-Unpaired-Data

cd mmfashion/

python setup.py install

cd ..



