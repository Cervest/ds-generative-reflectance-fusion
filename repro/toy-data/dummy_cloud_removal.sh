config_root="src/toygeneration/config/dummy_cloud_removal/"
data_root="data/toy/dummy_cloud_removal/"
for seed in {1..50}
do
  printf "\n GENERATING FOR SEED = "$seed"\n"
  # Setup dump directories paths
  clean_optical_path=$data_root$seed"/clean_optical"
  clean_sar_path=$data_root$seed"/clean_sar"
  clouded_optical_path=$data_root$seed"/clouded_optical"
  derived_sar_path=$data_root$seed"/sar"
  # Run toy product generation and derivation
  python run_toy_generation.py --cfg=$config_root"optical/generation_clean_optical.yaml" \
  --o=$clean_optical_path --seed=$seed
  python run_toy_derivation.py --cfg=$config_root"optical/derivation_clouded_optical.yaml" \
  --o=$clouded_optical_path --product=$clean_optical_path
  python run_toy_generation.py --cfg=$config_root"sar/generation_clean_sar.yaml" \
  --o=$clean_sar_path --seed=$seed
  python run_toy_derivation.py --cfg=$config_root"sar/derivation_sar.yaml" \
  --o=$derived_sar_path --product=$clean_sar_path
  rm -rf $clean_sar_path
done
