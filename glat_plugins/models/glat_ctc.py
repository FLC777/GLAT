# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import torch
import torch.nn.functional as F
from fairseq.models.nat.nonautoregressive_transformer import NATransformerEncoder, NATransformerDecoder, \
    NATransformerModel
import logging
import random
from contextlib import contextmanager
from torch_imputer.imputer import best_alignment

logger = logging.getLogger(__name__)


@contextmanager
def torch_seed(seed):
    state = torch.random.get_rng_state()
    state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        torch.cuda.random.set_rng_state(state_cuda)


@register_model("glat_ctc")
class GlatCTC(FairseqNATModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.mask_idx = self.unk

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        encoder = NATransformerEncoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward_encoder(self, encoder_inputs):
        return self.encoder(*encoder_inputs)

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        dec_out = self.decoder(
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            prev_output_probs=output_tokens,
            num_updates=step,
        )
        _scores, _tokens = dec_out.max(-1)
        _tokens = _tokens.masked_fill(~output_masks, self.pad)
        new_output_tokens = torch.zeros_like(_tokens).fill_(self.pad)
        batch_size, seq_len = _tokens.size()
        for bidx in range(batch_size):
            cidx = 0
            for sidx in range(0, seq_len - 1):
                if _tokens[bidx, sidx] == _tokens[bidx, sidx + 1] or _tokens[bidx, sidx] == self.unk:
                    continue
                else:
                    new_output_tokens[bidx, cidx] = _tokens[bidx, sidx]
                    cidx += 1
            new_output_tokens[bidx, cidx] = _tokens[bidx, -1]
        # output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=new_output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def initialize_ctc_input(self, src_tokens):
        length_tgt = src_tokens.ne(self.pad).sum(-1) * 2
        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        return initial_output_tokens

    def initialize_output_tokens(self, encoder_out, src_tokens):
        initial_output_tokens = self.initialize_ctc_input(src_tokens)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
                length_tgt[:, None]
                + utils.new_arange(length_tgt, 1, beam_size)
                - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )

    def set_num_updates(self, num_updates):
        self._num_updates = num_updates

    @property
    def get_num_updates(self):
        return self._num_updates

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat=None, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        decoder_padding_mask = prev_output_tokens.eq(self.pad)

        target = tgt_tokens
        bsz, target_len = tgt_tokens.size()
        nonpad_positions = tgt_tokens.ne(self.pad)
        target_lens = (nonpad_positions).sum(1)
        bsz, seq_len = prev_output_tokens.size()
        seq_lens = (prev_output_tokens.ne(self.pad)).sum(1)
        rand_seed = random.randint(0, 19260817)
        # glancing sampling
        glat_info = None
        if glat and tgt_tokens is not None:
            with torch.no_grad():
                with torch_seed(rand_seed):
                    word_ins_out = self.decoder(
                        normalize=False,
                        prev_output_tokens=prev_output_tokens,
                        encoder_out=encoder_out,
                    )
                pred_tokens = word_ins_out.argmax(-1)
                out_lprobs = F.log_softmax(word_ins_out, dim=-1, dtype=torch.float32)
                best_aligns = best_alignment(out_lprobs.transpose(0, 1), target, seq_lens, target_lens, self.mask_idx,
                                             zero_infinity=True)
                best_aligns_pad = torch.tensor([a + [0] * (word_ins_out.shape[1] - len(a)) for a in best_aligns],
                                               device=word_ins_out.device, dtype=target.dtype)
                oracle_pos = (best_aligns_pad // 2).clip(max=target.shape[1] - 1)

                oracle = target.gather(-1, oracle_pos)
                oracle_empty = oracle.masked_fill(best_aligns_pad % 2 == 0, self.mask_idx)

                same_num = ((pred_tokens == oracle_empty) & (~decoder_padding_mask)).sum(1)
                keep_prob = ((seq_lens - same_num) / seq_lens * glat['context_p']).unsqueeze(-1)
                # keep: True, drop: False
                keep_word_mask = (torch.rand(prev_output_tokens.shape, device=word_ins_out.device) < keep_prob).bool()

                glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask, 0) + oracle.masked_fill(
                    ~keep_word_mask, 0)

                prev_output_tokens = glat_prev_output_tokens

                glat_info = {
                    "glat_accu": (same_num.sum() / seq_lens.sum()).item(),
                    "glat_context_p": glat['context_p'],
                }

        with torch_seed(rand_seed):
            word_ins_out = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
            )
        ret = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "ctc_loss": True,
            },
        }
        if glat_info is not None:
            ret.update(glat_info)
        return ret


@register_model_architecture(
    "glat_ctc", "glat_ctc_6e6d512"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "glat_ctc", "glat_ctc"
)
def glat_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", args.encoder_embed_dim * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", args.encoder_embed_dim // 64)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.decoder_embed_dim * 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", args.decoder_embed_dim // 64)
    base_architecture(args)


@register_model_architecture(
    "glat_ctc", "glat_ctc_base"
)
def base_architecture2(args):
    base_architecture(args)
