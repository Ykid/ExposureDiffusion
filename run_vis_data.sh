# The initial version
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

# My favorite from the comments. Thanks @richarddewit & others!
set -a && source .env && set +a

echo $ED_SAVE_EPOCH_FREQ
python -m pdb vis_data_v2.py \
        --name sid_Pg_naf2 \
        --include 4 \
        --noise P+g \
        --model eld_iter_model \
        --with_photon \
        --adaptive_res_and_x0 \
        --iter_num 2 \
        --epoch 300 \
        --max_dataset_size 1 \
        --auxloss \
        --continuous_noise \
        --adaptive_loss \
        --netG naf2 \
        --seed 2025
        # --batchSize 2 \
        # --nThreads 4