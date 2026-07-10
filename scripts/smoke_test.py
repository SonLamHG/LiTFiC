"""
Part B smoke-test for LiTFiC.

Purpose: prove the CORE pipeline (MMProjector + LanguageDecoder prompt assembly,
label masking, CE loss, and auto-regressive generation) actually runs -- WITHOUT
the full lightning/hydra/LMDB stack, real BOBSL data, or Llama-3-8B.

It swaps Llama-3-8B for a tiny Llama-architecture model (SmolLM-135M) so it fits on
CPU / a 4GB GPU, forces eager attention + fp32, and feeds a synthetic batch shaped
exactly like `collate_fn_padd_t`'s output.

Run:  python scripts/smoke_test.py
"""
import os, sys, types, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# `src/utils/__init__.py` eagerly imports hydra/lightning/omegaconf (not installed
# here and not needed for the core pipeline). Inject an empty stub package so that
# `from src.utils.data_utils import ...` still resolves the submodule by path,
# without executing the heavy __init__.
_utils = types.ModuleType("src.utils")
_utils.__path__ = [os.path.join(ROOT, "src", "utils")]
sys.modules["src.utils"] = _utils

from src.models.components.vgg_slt import VggSLTNet

# A tiny, ungated, Llama-architecture model (has .model.embed_tokens, q_proj/v_proj).
LLM = os.environ.get("SMOKE_LLM", "HuggingFaceTB/SmolLM-135M")


class AttrDict(dict):
    """dict that also supports attribute access (LanguageDecoder does
    `decoder_config.attn_implementation = None` then `**decoder_config`)."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def build_net(llm_hidden):
    mm_projector_config = dict(
        mm_hidden_size=768,          # Video-Swin feature dim (paper)
        hidden_size=llm_hidden,      # must match the LLM embedding dim
        projector_type="mlp2x_gelu",
        dropout=0.0,
    )
    llm_config = dict(
        pretrained_llm=LLM,
        gradient_checkpointing_enable=False,
        freeze_decoder=False,
        lora=True,
        # full set of cues (the Vid+PG+Prev+BG experiment)
        use_pl_w_feats=True,
        mix_in_pls_prob=0.5,
        use_rec_prev=True,
        mix_in_prev_prob=0.5,
        bg_desc=True,
        use_bg_words=True,
        drop_bg_sw=True,
        mix_in_bg_prob=0.5,
        dropout=0.0,
        lora_config=dict(
            target_modules=["q_proj", "v_proj"],
            lora_alpha=16, lora_dropout=0.05, r=4, bias="none",
        ),
        tokenizer_config=dict(
            padding_side="left", trust_remote_code=True,
            use_fast=True, add_eos_token=True,
        ),
        decoder_config=AttrDict(trust_remote_code=True,
                                attn_implementation="eager"),
    )
    return VggSLTNet(mm_projector_config, llm_config,
                     load_features=True, precision="32-true")


def synthetic_batch(feat_dim=768, seqs=(7, 5)):
    B, T = len(seqs), max(seqs)
    features = torch.zeros(B, T, feat_dim)
    attn = torch.zeros(B, T)
    for i, t in enumerate(seqs):
        features[i, :t] = torch.randn(t, feat_dim)
        attn[i, :t] = 1.0
    instr = ("You are an AI assistant designed to interpret a video of a sign "
             "language signing sequence and translate it into English.")
    return {
        "features": features,
        "attn_masks": attn,
        "subtitles": ["as pagans the romans worshipped many gods and spirits.",
                      "the wind carried the scent of flowers."],
        "questions": [instr, instr],
        "previous_contexts": ["the temple stood on the hill.",
                              "it was a bright morning."],
        "pls": [["roman", "many", "god", "worship"], ["wind", "flower"]],
        "rec_prev": [],
        "bg_description": ["a stone temple with columns and statues.",
                           "a field of colourful flowers under the sky."],
        "spottings": [[], []],
    }


def main():
    from transformers import AutoConfig
    llm_hidden = AutoConfig.from_pretrained(LLM).hidden_size
    print(f"[smoke] loading tiny LLM: {LLM} (hidden dim = {llm_hidden})")
    net = build_net(llm_hidden)

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"[smoke] trainable params = {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)  <- MLP projector + LoRA only")

    batch = synthetic_batch()

    # ---- training forward: loss on subtitle tokens only ----
    net.train()
    outputs, labels, gen = net(batch)
    logits = outputs["logits"]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1))  # ignore_index=-100
    n_supervised = int((labels != -100).sum())
    print(f"[smoke] train loss = {loss.item():.4f} over {n_supervised} "
          f"subtitle tokens (rest masked to -100)")
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in net.parameters()
                if p.requires_grad and p.grad is not None)
    print(f"[smoke] backward OK, summed grad norm = {gnorm:.4f}")

    # ---- inference: auto-regressive generation ----
    net.eval()
    with torch.no_grad():
        _, _, gen = net(batch)
    tok = net.language_decoder.tokenizer
    print("[smoke] generated (projector/LoRA untrained; fluent because base LLM "
          "is pretrained, but not grounded in the video tokens):")
    for i, seq in enumerate(gen):
        print(f"   [{i}] {tok.decode(seq, skip_special_tokens=True)!r}")

    print("\n[smoke] PASS: forward+loss+backward+generate all executed.")


if __name__ == "__main__":
    main()
