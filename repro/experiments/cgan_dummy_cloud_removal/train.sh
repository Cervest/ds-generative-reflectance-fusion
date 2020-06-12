dvc run -v -f repro/downloads/cgan_dummy_cloud_removal/train.dvc \
-d src/rsgan/experiments/cgan_cloud_removal.py \
-d src/rsgan/config/cgan_dummy_cloud_removal.yaml \
-o data/experiments_outputs/ \
python run_training.py --cfg=src/rsgan/config/cgan_dummy_cloud_removal.yaml \
--o=data/experiments_outputs/cgan_dummy_cloud_removal --device=0 --name=dvc_run
