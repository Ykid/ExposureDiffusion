# python3 test_SID.py \
#     --model eld_iter_model \
#     --model_path "/home/david.weijiecai/computational_imaging/sid_PGru.pt" \
#     --include 4 --with_photon \
#     --adaptive_res_and_x0 \
#     -r --iter_num 2 \
#     --concat_origin

# iteration 2 is the best for naf2
# python3 test_SID.py --model eld_iter_model \
#     --model_path "/home/david.weijiecai/computational_imaging/sid_Pg_naf2.pt" \
#     --include 4 --with_photon \
#     --adaptive_res_and_x0 -r \
#     --iter_num 2 --netG naf2

# OUTPUT_PATH_ROOT="/home/david.weijiecai/computational_imaging/ExposureDiffusion/output" \
python3 test_full_SID.py --model eld_iter_model \
    --model_path "/home/david.weijiecai/computational_imaging/sid_Pg_naf2.pt" \
    --include 4 --with_photon \
    --adaptive_res_and_x0 -r \
    --iter_num 2 --netG naf2