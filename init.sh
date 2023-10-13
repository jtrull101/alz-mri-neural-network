echo Running init script to move pre-fetched-data and best weights
mkdir /tmp/alz_mri_cnn/
cp -r 'pre_fetched_data/Combined Dataset/' /tmp/alz_mri_cnn/

mkdir /tmp/alz_mri_cnn/models/
cp -r src/alz_mri_cnn/static/optimal_weights_*.keras /tmp/alz_mri_cnn/models/