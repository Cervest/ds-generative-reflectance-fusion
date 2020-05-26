dvc run -v -f repro/toy-data/download_ts.dvc \
-d repro/toy-data/ts.py \
-o data/ts/ \
python repro/toy-data/ts.py
