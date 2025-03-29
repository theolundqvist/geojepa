chmod +x setup-aws-cli.sh && \
chmod +x tile.sh && \
chmod +x merge_tiles && \
apt-get install bc zip parallel && \
conda activate iris && \
conda install -y -c conda-forge gdal && \
pip install -r requirements.txt
