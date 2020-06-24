dvc run -v -f repro/experiments/cgan_dummy_cloud_removal/train.dvc \
-d src/rsgan/experiments/cgan_cloud_removal.py \
-d src/rsgan/config/cgan_dummy_cloud_removal.yaml \
-o data/experiments_outputs/cgan_dummy_cloud_removal/dvc_run/checkpoints/ \
"python run_training.py --cfg=src/rsgan/config/cgan_dummy_cloud_removal.yaml --o=data/experiments_outputs/cgan_dummy_cloud_removal --device=0 --experiment_name=dvc_run"
