conda create -n geojepa python=3.10
conda activate geojepa

pip install torch==2.2.2 # intel mac compatible, otherwise 2.4 should be alright
pip install -r requirements.txt
