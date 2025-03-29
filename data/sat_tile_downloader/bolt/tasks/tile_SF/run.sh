conda activate iris

# images are 500MB each (3s per image)
time python download.py --non-interactive \
  --min_lat 37.224 \
  --max_lat 38.276 \
  --min_lon -122.60 \
  --max_lon -121.125 \
  --start_date 2018-01-01 \
  --max_downloads 500 \
  --max_concurrent_downloads 48 \
  --download_dir naip


# tile into zoom 14  (21s per image on 12 cores)
time ./tile.sh 24 naip all-tiles

# merge tiles (???)
time ./merge_tiles all-tiles merged-tiles

time zip -r $CLOUD_ARTIFACT_DIR/sanfrancisco.zip merged-tiles
