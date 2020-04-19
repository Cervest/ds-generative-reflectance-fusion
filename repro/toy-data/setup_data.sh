dvc run -f repro/toy-data/download_data.dvc \
-o data/mnist/ \
-o data/time-series/ \
"python repro/toy-data/mnist.py; python repro/toy-data/ts.py"