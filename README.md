# Environment
conda create -n develop_cancer_detection python=3.10
conda activate develop_cancer_detection

conda install pytorch torchvision cpuonly -c pytorch
conda install matplotlib
conda install jupyterlab 

# Inception V3
Inception V3 is used as a basis for neural network

# How to run
python ./train_nn.py \
--dataset_path $DIR_PATH/data \
--dataset_cache_path $DIR_PATH/data \
--model_file $DIR_PATH/model.py \
--model_weights $DIR_PATH/weights/$NAME_OF_FILE \
--learning_rate 0.001 \
--batch_size 64 \
--save_every_epochs 5 \
--evaluate_epochs_period 1 \
--epochs_count 5 \
--load_existing_model True
