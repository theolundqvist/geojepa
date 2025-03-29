
# =============================================================================
# Automatically generated. Do not modify directly.
# Use the corresponding Python script (submit_experiment.py) to regenerate if necessary.
# =============================================================================
chmod +x setup-aws-cli.sh && \
./setup-aws-cli.sh && \
apt-get -y install zstd tar tree zip && \
conda activate iris && \
pip install -r requirements.txt && \
pip install torch_geometric && \
pip install pyg_lib torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
