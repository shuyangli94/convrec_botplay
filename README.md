# convrec_botplay
Conversation generation for conversational recommendation.

This is our PyTorch implementation for the paper:

[Self-Supervised Bot Play for Transcript-Free Conversational Critiquing with Rationales, ACM TORS 2024](LINK)

which is an extended version of [Self-Supervised Bot Play for Transcript-Free Conversational Recommendation with Rationales, RecSys 2022](https://doi.org/10.1145/3523227.3546783)

The code is tested on a Linux workstation (with NVIDIA GeForce RTX 4090).

## Citation

If you find this repository useful for your research, please cite the following papers:

```
@article{DBLP:journals/tors/LiSSBPTFCCR24,
  author       = {Shuyang Li and
                  Bodhisattwa Prasad Majumder and
                  Julian J. McAuley},
  title        = {Self-Supervised Bot Play for Transcript-Free
                  Conversational Critiquing with Rationales},
  journal      = {Trans. Recomm. Syst.},
  volume       = {X},
  number       = {X},
  pages        = {XXX},
  year         = {2024},
  url          = {XXX},
  doi          = {XXX},
}

@inproceedings{DBLP:conf/recsys/LiMM22,
  author       = {Shuyang Li and
                  Bodhisattwa Prasad Majumder and
                  Julian J. McAuley},
  title        = {Self-Supervised Bot Play for Transcript-Free Conversational Recommendation
                  with Rationales},
  booktitle    = {RecSys '22: Sixteenth {ACM} Conference on Recommender Systems, Seattle,
                  WA, USA, September 18 - 23, 2022},
  pages        = {327--337},
  publisher    = {{ACM}},
  year         = {2022},
  url          = {https://doi.org/10.1145/3523227.3546783},
  doi          = {10.1145/3523227.3546783},
}
```

## Setup

To set up the environment, follow instructions in `README-ENV.md`

## Get Data

Please find the data on [Google Drive](https://drive.google.com/file/d/1e28d-vqaL0uWuAX5JHoJHunxUrL2HtBp/view?usp=sharing)

## How to Run

We provide sample code to train a recommender systems model, fine-tune it with bot-play, and run critiquing.

### Environment Variables

```
export HOME_DIR=/home/$USER/

export VENV_LOC=${HOME_DIR}/cr-env
export CODE_DIR=${HOME_DIR}/convrec_botplay
export DATA_DIR=${HOME_DIR}/BeerAdvocateDC

export MODEL_DIR=${HOME_DIR}/models-debug
mkdir -p ${MODEL_DIR}

export LOG_DIR=${HOME_DIR}/logs
mkdir -p ${LOG_DIR}

export CRITIQUES_DIR=${HOME_DIR}/critiques-debug
mkdir -p ${CRITIQUES_DIR}
```

### Training Base Models

```

# Train BPR
source ${VENV_LOC}/bin/activate
cd ${CODE_DIR}

nohup python3 -u -m train_rec params=bpr-kp-u-beer-debug \
	params.name=DEBUG-BPR-BEERDC-UKP \
	params.splits_loc=${DATA_DIR}/beeradvocate_DC_splits.pkl \
	params.kp_loc=${DATA_DIR}/beeradvocate_DC_ix.pkl \
	params.ckpt_path=${MODEL_DIR}/ > ${LOG_DIR}/BPR-BEERDCNB-U-DEBUG.log &

tail -f ${LOG_DIR}/BPR-BEERDCNB-U-DEBUG.log


# Train PLRec

source ${VENV_LOC}/bin/activate
cd ${CODE_DIR}


nohup python3 -u -m train_plrec \
	--split-file ${DATA_DIR}/beeradvocate_DC_splits.pkl \
	--kp-file ${DATA_DIR}/beeradvocate_DC_ix.pkl \
	--ds beer \
	--model-dir ${MODEL_DIR} > ${LOG_DIR}/PLREC-BEER-DEBUG.log &

tail -f ${LOG_DIR}/PLREC-BEER-DEBUG.log
```

### Bot-play Fine-Tuning

```

# Fine-tune BPR
nohup python3 -u -m train_ft_proj params=ft-r-bpr-beer-debug \
	params.name=FT-R-BEERDC-10 \
	params.splits_loc=${DATA_DIR}/beeradvocate_DC_splits.pkl \
	params.kp_loc=${DATA_DIR}/beeradvocate_DC_ix.pkl \
	params.pretrained=${MODEL_DIR}/DEBUG-BPR-BEERDC-UKP_k10_UE_EL0_PL0_BS4096_KW0.5_lr0.01_LL0.01_TI \
	params.ckpt_path=${MODEL_DIR}/> ${LOG_DIR}/FT-BPR-BEER-DEBUG.log &

tail -f ${LOG_DIR}/FT-BPR-BEER-DEBUG.log

# Fine-tune PLRec
nohup python3 -u -m train_ft_plrec_proj params=ft-r-proj-plrec-beer-debug \
	params.name=FT-BEER-PLREC-DEBUG \
	params.splits_loc=${DATA_DIR}/beeradvocate_DC_splits.pkl \
	params.kp_loc=${DATA_DIR}/beeradvocate_DC_ix.pkl \
	params.pretrained=${MODEL_DIR}/PLRecKP-beer_I10_L80_d50 \
	params.ckpt_path=${MODEL_DIR}/> ${LOG_DIR}/FT-PLREC-BEER-DEBUG.log &

tail -f ${LOG_DIR}/FT-PLREC-BEER-DEBUG.logs
```

### Evaluating critiquing

```
# BPR
source ${VENV_LOC}/bin/activate
cd ${CODE_DIR}

nohup python3 -u -m critique_proj \
    --model-dir ${MODEL_DIR}/FT-R-BEERDC-10_k10_UE_EL0_PL0_BS512_D0.9_F_MT5_lr0.001_LL0.0_TI_AL0.5 \
    --out-dir ${CRITIQUES_DIR} \
    --aspect-filter False \
    --allow-repeat False \
    --user-strat coop,random,diff \
    --feedback-type N \
    --target item \
    --kp-gen bernoulli \
    --gm-scale 1.0 \
    --window 1 \
    --n-feedback 1 \
    --criterion first \
    --overwrite \
    --name BPR-Beer-Debug \
    --sample 500 --shuffle --split test > ${LOG_DIR}/debug_bpr_critique.log &

tail -f ${LOG_DIR}/debug_bpr_critique.log 


# PLRec
source ${VENV_LOC}/bin/activate
cd ${CODE_DIR}

nohup python3 -u -m critique_plrec_proj \
    --model-dir ${MODEL_DIR}/FT-BEER-PLREC-DEBUG_BS4096_D0.9_F_MT3_lr0.01_LL0.001_TI_AL0.5 \
    --out-dir ${CRITIQUES_DIR} \
    --aspect-filter False \
    --allow-repeat False \
    --user-strat coop,random,diff \
    --feedback-type N \
    --target item \
    --kp-gen bernoulli \
    --gm-scale 1.0 \
    --window 1 \
    --n-feedback 1 \
    --criterion first \
    --overwrite \
    --name PLRec-Beer-Debug \
    --sample 500 --shuffle --split test > ${LOG_DIR}/debug_plrec_critique.log &

tail -f ${LOG_DIR}/debug_plrec_critique.log 
```

## Correspondence

For all correspondence, please email shuyangli94@gmail.com