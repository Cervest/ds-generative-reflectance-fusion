dvc run -v -f repro/toy-data/dummy_cloud_removal/generate_dummy_cloud_removal_dataset.dvc \
-d run_toy_generation.py \
-d src/toygeneration/config/dummy_cloud_removal/optical/generation_latent_optical.yaml \
-d src/toygeneration/config/dummy_cloud_removal/optical/derivation_clean_optical.yaml \
-d src/toygeneration/config/dummy_cloud_removal/optical/derivation_clouded_optical.yaml \
-d src/toygeneration/config/dummy_cloud_removal/sar/generation_latent_sar.yaml \
-d src/toygeneration/config/dummy_cloud_removal/sar/derivation_sar.yaml \
-d data/ts/Multivariate_ts/Cricket/Cricket_TRAIN.ts \
-d data/ts/Multivariate_ts/Handwriting/Handwriting_TRAIN.ts \
-o data/toy/dummy_cloud_removal/ \
"source repro/toy-data/dummy_cloud_removal/dummy_cloud_removal.sh"
