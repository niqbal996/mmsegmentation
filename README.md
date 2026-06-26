# Installation 
For MMSeg installation, please follow the official mmseg installation [instructions](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).  

For dataset preparation and config paths, please refer to [datasets](DATASET.md).

[PhenoBench](https://www.phenobench.org/) and [SugarbeetSynthetic2026](https://bit.ly/40adJsL) can be downloaded from the given links. 
# Training
For any experiment, the relevant config needs to be provided to the `tools/train.py` script. All the configs are under `configs/wacv` folder. The dataset root paths need to be changed for all the datasets in `configs/wacv/datasets`.
```bash
python3 tools/train.py --config-file configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_syn_v6.py 
```
For training all the models in one go via slurm using `sbatch_train.sh`
```bash
bash ./schedule_training.sh
```

# Batch evaluation
```bash
bash ./eval_sweep_models.sh
```

# Generate metrics 
Once `eval_sweep_models.sh` is executed for each experiment, the metrics `json` file for each seed experiment set, can be generated using:
```bash
python print_metrics_summary.py \ 
    phenobench_metrics_seed_1.json \ 
    phenobench_metrics_seed_2.json \ 
    phenobench_metrics_seed_3.json \ 
    --out table.tex --preview
```

# Tiny instance labelling tool
To find and label tiny false positive weeds, use the `tools/label_exg_fps.py` which generates `exg_labels.csv` for each labeller:
```bash
python tools/label_exg_fps.py \ 
            configs/wacv/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_phenobench_ohem_loss.py \ 
            /mnt/e/trainers/WACV_results/Deeplabv3Plus_r50_phenobench_ohem_loss/best_IoU_weed_iter_24000.pth \ 
            --output-dir /tmp/exg_fps \ 
            --n-samples 60 \ 
            --max-area 200 \ 
            --crop-size256 \ 
            --devicecuda:0
```
and to find ExG threshold 
```bash
python3 tools/analyze_exg_labels.py \ 
            tmp/exg_fps_labeller_1/exg_labels.csv \ 
            tmp/exg_fps_labeller_2/exg_labels.csv \ 
            tmp/exg_fps_labeller_3/exg_labels.csv \ 
            --names labeller_1 labeller_2 labeller_3 \ 
            --output ./tmp/exg_analysis
```
# Model checkpoints 

For anonymity reasons, all the model checkpoint files are uploaded to the [Zenodo repository](https://zenodo.org/records/18961940?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc3MzI0MzM3OCwiZXhwIjoxNzg4MTM0Mzk5fQ.eyJpZCI6IjFiYWY4ZDc3LTY2ZmUtNGFjNy1hZDAwLWQ5ZGI5NmE5NTI1MiIsImRhdGEiOnt9LCJyYW5kb20iOiJhODYxNjZiNjNmM2E2YWRjNDEyYzZiODdiNTAxMjNhMyJ9.Swd1g8Q_NqQKD9HeL4BUqglU49365dBNPYaw8pA-IvTCXRkCWS_KpElj37vRXj-Uw-Ozxna04yC0pxwDrWtJJA). Once the anonymous phase is over, the weights will be added
as hyperlink in the respective position in the above table. The hyperlink might not work directly from the anonymous repository. Please copy the anonymous Zenodo link from the source code. 

# TODO 
- [ ] Add model checkpoints for the respective configs. Refer to the model checkpoints above. 