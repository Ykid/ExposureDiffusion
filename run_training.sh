# The initial version
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

# My favorite from the comments. Thanks @richarddewit & others!
set -a && source .env && set +a

# Training code
# python train_syn.py \
#         --name sid_Pg_naf2 \
#         --include 4 \
#         --noise P+g \
#         --model eld_iter_model \
#         --with_photon \
#         --adaptive_res_and_x0 \
#         --iter_num 2 \
#         --epoch 300 \
#         --auxloss \
#         --continuous_noise \
#         --adaptive_loss \
#         --netG naf2 --batchSize 2 --nThreads 4
# training with v2
python -m pdb train_syn_v2.py \
        --name sid_Pg_naf2 \
        --include 4 \
        --noise P+g \
        --model eld_iter_v2_model \
        --with_photon \
        --adaptive_res_and_x0 \
        --iter_num 2 \
        --epoch 600 \
        --auxloss \
        --continuous_noise \
        --adaptive_loss \
        --netG naf2 \
        --batchSize 1 \
        --nThreads 4 \
        --resume \
        --model_path /home/david.weijiecai/computational_imaging/sid_Pg_naf2.pt \
        --checkpoints_dir /home/david.weijiecai/computational_imaging/ExposureDiffusion/checkpoints/

# # The initial version
# if [ ! -f .env_debug ]
# then
#   export $(cat .env_debug | xargs)
# fi

# # My favorite from the comments. Thanks @richarddewit & others!
# set -a && source .env_debug && set +a
# overfitting
# python -m pdb train_syn.py \
#         --name sid_Pg_naf2 \
#         --include 4 \
#         --noise P+g \
#         --model eld_iter_model \
#         --with_photon \
#         --adaptive_res_and_x0 \
#         --iter_num 2 \
#         --epoch 300 \
#         --max_dataset_size 1 \
#         --auxloss \
#         --continuous_noise \
#         --adaptive_loss \
#         --netG naf2 \
#         --batchSize 1 \
#         --nThreads 0

# just for debugging
# python -m pdb train_syn_v2.py \
#         --name sid_Pg_naf2 \
#         --include 4 \
#         --noise P+g \
#         --model eld_iter_v2_model \
#         --with_photon \
#         --adaptive_res_and_x0 \
#         --iter_num 2 \
#         --epoch 600 \
#         --auxloss \
#         --continuous_noise \
#         --adaptive_loss \
#         --netG naf2 \
#         --batchSize 1 \
#         --nThreads 4 \
#         --resume \
#         --model_path /home/david.weijiecai/computational_imaging/sid_Pg_naf2.pt \
#         --checkpoints_dir /home/david.weijiecai/computational_imaging/ExposureDiffusion/checkpoints_debug/