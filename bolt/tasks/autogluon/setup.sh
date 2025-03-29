chmod +x setup-aws-cli.sh && \
./setup-aws-cli.sh && \
apt-get -y install zip unzip ffmpeg libsm6 libxext6 && \
#conda activate iris && \
#pip install -r requirements.txt && \
conda activate iris && \
pip install -U pip && \
pip install -U setuptools wheel && \
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
git clone https://github.com/autogluon/autogluon && \
cd autogluon && ./full_install.sh