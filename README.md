# GLAT
Implementation for the ACL2021 paper "Glancing Transformer for Non-Autoregressive Neural Machine Translation"

### Requirements
* Python >= 3.7
* Pytorch >= 1.5.0
* Fairseq 1.0.0a0

### Preparation
Train an autoregressive Transformer according to the instructions in [Fairseq](https://github.com/pytorch/fairseq).

Use the trained autoregressive Transformer to generate target sentences for the training set.

Binarize the distilled training data.

```
input_dir=path_to_raw_text_data
data_dir=path_to_binarized_output
src=source_language
tgt=target_language
python3 fairseq_cli/preprocess.py --source-lang ${src} --target-lang ${tgt} --trainpref ${input_dir}/train \
    --validpref ${input_dir}/valid --testpref ${input_dir}/test --destdir ${data_dir}/ \
    --workers 32 --src-dict ${input_dir}/dict.${src}.txt --tgt-dict {input_dir}/dict.${tgt}.txt
```

### Train
* For training GLAT
```
save_path=path_for_saving_models
python3 train.py ${data_dir} --arch glat --noise full_mask --share-all-embeddings \
    --criterion glat_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 --task translation_lev_modified --max-tokens 8192 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 1000 --max-target-positions 1000 --max-update 300000 --seed 0 --clip-norm 5\
    --save-dir ${save_path} --src-embedding-copy --length-loss-factor 0.05 --log-interval 1000 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir glat_plugins \
```

* For training GLAT+CTC
```
save_path=path_for_saving_models
python3 train.py ${data_dir} --arch glat_ctc --noise full_mask --share-all-embeddings \
    --criterion ctc_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 --task translation_lev_modified --max-tokens 8192 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 1000 --max-target-positions 1000 --max-update 300000 --seed 0 --clip-norm 2\
    --save-dir ${save_path} --length-loss-factor 0 --log-interval 1000 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir glat_plugins \
```

### Inference
```
checkpoint_path=path_to_your_checkpoint
python3 fairseq_cli/generate.py ${data_dir} --path ${checkpoint_path} --user-dir glat_plugins \
    --task translation_lev_modified --remove-bpe --max-sentences 20 --source-lang ${src} --target-lang ${tgt} \
    --quiet --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset test
```
The script for averaging checkpoints is scripts/average_checkpoints.py

Thanks [dugu9sword](https://github.com/dugu9sword) for contributing part of the code.
