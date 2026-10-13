#!/usr/bin/env bash
# Train a released model on one node. Every architecture and optimisation flag
# defaults to the released recipe (relsgg/config.py), so this sets only the
# data, the backbone and the artifacts.
#
#   BACKBONE=facebook/dinov3-vits16-pretrain-lvd1689m ./train.sh
#   BACKBONE=facebook/dinov3-convnext-tiny-pretrain-lvd1689m BACKBONE_LR=1e-5 ./train.sh
#   MIX=zeroshot ./train.sh                                      # evaluation control, no HICO-DET
#   NPROC=1 EPOCHS=1 ./train.sh --samples_per_epoch 2048         # smoke run; extra flags pass through
#
# Data roots are PACKED datasets (training/pack_megasg.py), not raw COCO; see
# docs/installation.md for where to download them, and docs/training.md for
# what the recipe does and why.
#
# The backbone is a gated Hugging Face repository: run `huggingface-cli login`
# once, or point BACKBONE at a local conversion (training/convert_dinov3_local.py,
# training/convert_dinov3_convnext.py). On an offline node, warm the cache first
# and export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1.
set -euo pipefail

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

BACKBONE="${BACKBONE:-facebook/dinov3-vits16plus-pretrain-lvd1689m}"
# 5e-5 for the ViTs; the ConvNeXt proxy ladder (1e-4 / 3e-5 / 1e-5 / 3e-6) peaks at 1e-5.
BACKBONE_LR="${BACKBONE_LR:-5e-5}"
# ConvNeXt-B and -L need activation checkpointing to fit 32 images per GPU at the
# 672 px rung on a 48 GB card; it leaves the gradients unchanged.
case "$BACKBONE" in
    *convnext-base*|*convnext-large*|*convnext_base*|*convnext_large*) GRAD_CKPT=(--backbone_grad_ckpt) ;;
    *) GRAD_CKPT=() ;;
esac
NPROC="${NPROC:-4}"
EPOCHS="${EPOCHS:-12}"
MIX="${MIX:-released}"
TS="${TS:-runs/packed/datamix_v22/text_space}"
RUN="${RUN:-runs/train/relsgg_$(basename "$BACKBONE")}"

case "$MIX" in
    # 72.7 / 6.3 / 21.0 % of images = 80.6 / 14.4 / 5.0 % of relations
    released) DATA=(--data_roots runs/packed/megasg_clean runs/packed/vg_raw runs/packed/hicodet
                    --mix_fractions 0.7274 0.063 0.2096
                    --restrict_neg_sources hicodet) ;;
    # 91.9 / 8.1 % of images = 84.6 / 15.4 % of relations
    zeroshot) DATA=(--data_roots runs/packed/megasg_clean runs/packed/vg_raw
                    --mix_fractions 0.919 0.081) ;;
    *) echo "MIX must be 'released' or 'zeroshot', got '$MIX'" >&2; exit 1 ;;
esac

torchrun --nproc_per_node "$NPROC" --master_port "${MASTER_PORT:-29500}" train.py \
    "${DATA[@]}" \
    --exclude_ids runs/datamix/indoorvg_holdout.json \
    --val_root runs/packed/megasg \
    --dev_root runs/packed/psg --dev_metric mR@50 \
    --backbone_model "$BACKBONE" --backbone_lr "$BACKBONE_LR" "${GRAD_CKPT[@]}" \
    --pred_embeds "$TS/pred_embeds_studentv2_512_photo.npz" \
    --ontology_meta "$TS/union_meta.json" \
    --soft_supervision "$TS/soft_supervision.npz" \
    --neg_rate_table runs/packed/datamix_v22/pair_opportunity.npz \
    --text_student runs/packed/text_student_v2_512/student.pt \
    --epochs "$EPOCHS" --batch_size 32 --num_workers 16 \
    --output_dir "$RUN" "$@"
