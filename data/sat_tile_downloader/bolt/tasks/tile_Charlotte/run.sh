conda activate iris

# images are 500MB each (3s per image)
time python download.py --non-interactive \
  --min_lat 34.8 \
  --max_lat 35.6 \
  --min_lon -81.45 \
  --max_lon -80.3 \
  --start_date 2018-01-01 \
  --max_downloads 300 \
  --max_concurrent_downloads 48 \
  --download_dir naip

# tile into zoom 14  (21s per image on 12 cores)
time ./tile.sh 24 naip all-tiles

# merge tiles (???)
time ./merge_tiles all-tiles merged-tiles

time zip -r $CLOUD_ARTIFACT_DIR/charlotte.zip merged-tiles
