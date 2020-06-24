dvc run -v -f repro/experiments/cgan_dummy_cloud_removal/evaluate.dvc \
-d src/rsgan/experiments/cgan_cloud_removal.py \
-d src/rsgan/config/cgan_dummy_cloud_removal.yaml \
-o data/experiments_outputs/cgan_dummy_cloud_removal/dvc_eval/ \
"python make_baseline_classifier.py --cfg=src/rsgan/config/cgan_dummy_cloud_removal.yaml --o=data/experiments_outputs/cgan_dummy_cloud_removal/baseline_classifier/rf.pickle --njobs=32 \
&& python run_testing.py --cfg=src/rsgan/config/cgan_dummy_cloud_removal.yaml --o=data/experiments_outputs/cgan_dummy_cloud_removal --device=0"
