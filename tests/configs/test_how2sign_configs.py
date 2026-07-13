import os

from hydra import compose, initialize_config_dir

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))


def _cfg(exp):
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        return compose(config_name="train", overrides=[f"experiment={exp}"])


def test_vid_config_t4_patches():
    cfg = _cfg("how2sign-vid")
    assert cfg.trainer.precision == "16-mixed"
    assert cfg.model.net.llm_config.decoder_config.attn_implementation == "sdpa"
    assert cfg.model.net.mm_projector_config.mm_hidden_size == 1024
    assert cfg.model.net.mm_projector_config.hidden_size == 2048
    assert cfg.data.dataset == "how2sign"
    assert cfg.model.net.llm_config.use_rec_prev is False


def test_vidprev_enables_prev():
    cfg = _cfg("how2sign-vid+prev")
    assert cfg.model.net.llm_config.use_rec_prev is True
    assert cfg.model.net.llm_config.use_gt_prev is False
    assert cfg.model.net.llm_config.mix_in_prev_prob == 1.0
    assert cfg.data.dataset_config.use_prev is True
