dvc run -v -f repro/toy-data/download_mnist.dvc \
-d repro/toy-data/mnist.py \
-o data/mnist/ \
python repro/toy-data/mnist.py
