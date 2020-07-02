dvc run -v -f repro/toy-data/sar_to_optical/run_generation.dvc \
-d run_toy_generation.py \
-d run_toy_derivation.py \
-d src/toygeneration/config/sar_to_optical/optical/generation_latent_optical.yaml \
-d src/toygeneration/config/sar_to_optical/optical/derivation_optical.yaml \
-d src/toygeneration/config/sar_to_optical/sar/generation_latent_sar.yaml \
-d src/toygeneration/config/sar_to_optical/sar/derivation_sar.yaml \
-d data/ts/Multivariate_ts/FingerMovements/FingerMovements_TRAIN.ts \
-d data/ts/Multivariate_ts/Handwriting/Handwriting_TEST.ts \
-o data/toy/sar_to_optical/ \
"source repro/toy-data/sar_to_optical/generate_toy_sar_to_optical_dataset.sh"
