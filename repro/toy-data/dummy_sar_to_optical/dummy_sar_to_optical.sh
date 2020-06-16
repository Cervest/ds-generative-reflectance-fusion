config_root="src/toygeneration/config/dummy_sar_to_optical/"
data_root="data/toy/dummy_sar_to_optical/"

for seed in {1..500}
do
  printf "\n GENERATING FOR SEED = "$seed"\n"

  # Setup dump directories paths
  latent_optical_path=$data_root$seed"/latent_optical"
  optical_path=$data_root$seed"/optical"
  latent_sar_path=$data_root$seed"/latent_sar"
  sar_path=$data_root$seed"/sar"

  # Run toy optical product generation
  python run_toy_generation.py --cfg=$config_root"optical/generation_latent_optical.yaml" \
  --o=$latent_optical_path --seed=$seed
  python run_toy_derivation.py --cfg=$config_root"optical/derivation_optical.yaml" \
  --o=$optical_path --product=$latent_optical_path
  rm -rf $latent_optical_path

  # Run toy SAR product generation
  python run_toy_generation.py --cfg=$config_root"sar/generation_latent_sar.yaml" \
  --o=$latent_sar_path --seed=$seed
  python run_toy_derivation.py --cfg=$config_root"sar/derivation_sar.yaml" \
  --o=$derived_sar_path --product=$latent_sar_path
  rm -rf $latent_sar_path
done
