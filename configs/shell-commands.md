# use this to run the latest model
# GuestFeat - this is the top performer

## Inference
/Users/matthewatkinson/Apps/shorts-model_v4/data/raw/aaron-friedberg.txt 

python -u shorts_model/inference/infer_minilm_v1.py \
    --transcript data/raw/aaron-friedberg.txt  \
    --top_k 5 \
    --target_tokens 220 \
    --overlap 0.20 \
    --sim_threshold 0.85 \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --regressor_path runs/v5/ridge_regressor_v5_top5rand.pkl \
    --guest "Aaron Friedberg"
  


# Train model - this will overwrite current run BTW, so create a copy beforehand



python shorts_model/modeling/train.py \
    --csv data/processed/training-data_v4.3_with-pseudo.csv \
    --outdir runs/v6 \
    --name v6 #", default="guestfeat_v1", help="Name stem to use in output filenames (e.g., run1)")

# Old train model
python train_guest_regressor.py --csv data/training-data_v5.csv




# GuestNorm
•  For guestnorm model (ridge_regressor_guestnorm_v1.pkl), also pass --guest:
•  /Users/matthewatkinson/Desktop/032425/venv/bin/python -u runs/infer_minilm_v1/infer_minilm_v1.py \
    --transcript data/transcript_anne-applebaum.txt \
    --top_k 5 \
    --target_tokens 220 \
    --overlap 0.20 \
    --sim_threshold 0.85 \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --regressor_path runs/train_regressor_minilm_guestnorm_v1/ridge_regressor_guestnorm_v1.pkl \
    --guest "Anne Applebaum"
•  If you omit --guest or the guest is unknown, the code falls back to the global mean stored in the model pickle.





/Users/matthewatkinson/Desktop/032425/venv/bin/python runs/infer_minilm_v1/infer_minilm_v1.py \
  --transcript data/transcript_anne-applebaum.txt \
  --top_k 5 \
  --target_tokens 220 \
  --overlap 0.20 \
  --sim_threshold 0.85 \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --regressor_path runs/train_regressor_minilm_v1/ridge_regressor_v1.pkl



   /Users/matthewatkinson/Desktop/032425/venv/bin/
   python runs/infer_minilm_v1/infer_minilm_v1.py \
   --transcript data/transcript_anne-applebaum.txt \
   --top_k 5 \
   --target_tokens 220 \
   --overlap 0.20 \
   --sim_threshold 0.85 \
   --model_name sentence-transformers/all-MiniLM-L6-v2 \
   --regressor_path runs/train_regressor_minilm_guestfeat_v1/ridge_regressor_guestfeat_v1.pkl


   # Tip: if you want to see progress immediately, add -u after python for unbuffered output:







