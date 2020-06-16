dvc run -v -f repro/toy-data/dummy_cloud_removal/generate_dummy_sar_to_optical_dataset.dvc \
-d run_toy_generation.py \
-d src/toygeneration/config/dummy_sar_to_optical/optical/generation_latent_optical.yaml \
-d src/toygeneration/config/dummy_sar_to_optical/optical/derivation_optical.yaml \
-d src/toygeneration/config/dummy_sar_to_optical/sar/generation_latent_sar.yaml \
-d src/toygeneration/config/dummy_sar_to_optical/sar/derivation_sar.yaml \
-d data/ts/Multivariate_ts/Cricket/Cricket_TRAIN.ts \
-d data/ts/Multivariate_ts/Handwriting/Handwriting_TRAIN.ts \
-o data/toy/dummy_sar_to_optical/ \
"source repro/toy-data/dummy_sar_to_optical/dummy_sar_to_optical.sh"
