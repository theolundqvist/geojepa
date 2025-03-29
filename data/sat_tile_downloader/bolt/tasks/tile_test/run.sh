conda activate iris

# images are 500MB each (3s per image)
time python download.py --non-interactive \
  --min_lat 33.2 \
  --max_lat 34.6 \
  --min_lon -85 \
  --max_lon -83.2 \
  --start_date 2018-01-01 \
  --max_downloads 1 \
  --max_concurrent_downloads 1 \
  --download_dir naip

# tile into zoom 14  (21s per image on 12 cores)
time ./tile.sh 1 naip all-tiles

# merge tiles (???)
time ./merge_tiles all-tiles merged-tiles

time zip -r $CLOUD_ARTIFACT_DIR/atlanta.zip merged-tiles
