"""
Generate a tiny SYNTHETIC dataset in the exact BOBSL formats the real pipeline
expects, so `src/train.py` / `src/eval.py` (hydra + Lightning + datamodule + LMDB)
run end-to-end with NO downloads, NO license, NO 262 GB.

Outputs (into --out-dir): feats_lmdb/, pl_lmdb/, subtitles.pkl, info.pkl, vocab.pkl,
synonyms.pkl, blip.pkl, subset2episode.json, train_cap.json,
val_start_indices.json, test_start_indices.json, and paths_tiny.yaml (ready to use
as `paths=` config).

Formats reverse-engineered from src/data/components/{lmdb_loader,subtitles,sentence}.py.
Caption quality is meaningless (random features) — this proves the FLOW, not numbers.

Usage:
  python scripts/make_tiny_bobsl.py --out-dir ./tiny_bobsl
  python scripts/make_tiny_bobsl.py --out-dir ./tiny_bobsl --verify   # round-trip check
"""
import os, sys, json, pickle, argparse
import numpy as np

FPS = 25
WIN = 16            # lmdb_window_size
STRIDE = 2          # lmdb_stride
FEAT_DIM = 768
BEGIN = WIN // 2 - 1        # 7  (frame_idx_to_feature_idx offset)
VOCAB = ["hello", "world", "sign", "language", "translate", "background",
         "person", "table", "flower", "wind", "roman", "worship", "god",
         "man", "standing", "front", "water", "city", "history", "story"]


def _n_features(duration_s):
    """How many feature keys cover a clip of `duration_s` seconds."""
    last_frame = int(duration_s * FPS)
    return (last_frame - BEGIN) // STRIDE + 2  # small margin


def write_lmdbs(out_dir, episodes):
    import lmdb
    feats = lmdb.open(os.path.join(out_dir, "feats_lmdb"), map_size=256 * 1024**2, subdir=True)
    pls = lmdb.open(os.path.join(out_dir, "pl_lmdb"), map_size=256 * 1024**2, subdir=True)
    rng = np.random.default_rng(0)
    with feats.begin(write=True) as ft, pls.begin(write=True) as pt:
        for ep, dur in episodes.items():
            n = _n_features(dur)
            for i in range(n):
                end = f"{i+1:07d}.np"
                ft.put(f"{ep}/{end}".encode(),
                       rng.standard_normal(FEAT_DIM).astype(np.float16).tobytes())
                lab = rng.integers(0, len(VOCAB), size=5).astype(np.int64)
                prob = rng.random(5).astype(np.float16)
                pt.put(f"{ep}_label/{end}".encode(), lab.tobytes())
                pt.put(f"{ep}_prob/{end}".encode(), prob.tobytes())
    feats.close(); pls.close()
    print(f"[tiny] wrote LMDBs for {len(episodes)} episodes")


def write_metadata(out_dir, episodes, sents_per_ep=6):
    rng = np.random.default_rng(1)
    ep_names, starts, ends, subs, durs, ids = [], [], [], [], [], []
    gid = 0
    for ep, dur in episodes.items():
        t = 0.5
        for _ in range(sents_per_ep):
            length = float(rng.uniform(2.0, 4.0))
            if t + length > dur - 0.5:
                break
            words = rng.choice(VOCAB, size=int(rng.integers(4, 8)), replace=True)
            ep_names.append(ep); starts.append(round(t, 2)); ends.append(round(t + length, 2))
            subs.append(" ".join(words) + "."); durs.append(round(length, 2)); ids.append(gid)
            gid += 1; t += length + 0.3
    subtitles = {"episode_name": ep_names, "start": starts, "end": ends,
                 "subtitle": subs, "duration": durs, "id": ids}
    pickle.dump(subtitles, open(os.path.join(out_dir, "subtitles.pkl"), "wb"))

    # info.pkl -> ["videos"]["videos"]["T"] (frame counts) + ["videos"]["name"]
    info = {"videos": {"videos": {"T": np.array([int(d * FPS) for d in episodes.values()])},
                       "name": [f"{ep}.mp4" for ep in episodes]}}
    pickle.dump(info, open(os.path.join(out_dir, "info.pkl"), "wb"))

    # vocab (flat word->id, no 'words_to_id' key) + synonyms (self-synonym)
    vocab = {w: i for i, w in enumerate(VOCAB)}
    pickle.dump(vocab, open(os.path.join(out_dir, "vocab.pkl"), "wb"))
    syns = {w: [w] for w in VOCAB}
    pickle.dump(syns, open(os.path.join(out_dir, "synonyms.pkl"), "wb"))

    # blip captions: {"video":[ep...], "captions":[[per-second strings]...]}
    blip = {"video": list(episodes.keys()),
            "captions": [[" ".join(rng.choice(VOCAB, 3)) for _ in range(int(d) + 2)]
                          for d in episodes.values()]}
    pickle.dump(blip, open(os.path.join(out_dir, "blip.pkl"), "wb"))

    eps = list(episodes.keys())
    train = eps[:max(1, len(eps) - 1)]
    heldout = eps[-1:]
    json.dump({"train": train, "val": heldout, "public_test": heldout, "test": heldout},
              open(os.path.join(out_dir, "subset2episode.json"), "w"), indent=2)
    json.dump({}, open(os.path.join(out_dir, "train_cap.json"), "w"))
    # val idx[-1] huge -> datamodule falls back to a full-set loader (no exact sharding needed)
    json.dump({"idx": [0, 10_000_000]}, open(os.path.join(out_dir, "val_start_indices.json"), "w"))
    json.dump({"idx": [0]}, open(os.path.join(out_dir, "test_start_indices.json"), "w"))
    print(f"[tiny] wrote metadata: {len(subs)} sentences across {len(episodes)} episodes")


def write_paths_yaml(out_dir):
    ad = os.path.abspath(out_dir).replace("\\", "/")
    y = f"""# Auto-generated by make_tiny_bobsl.py -- use with: paths=tiny_paths (copy to configs/paths/)
root_dir: ${{oc.env:PROJECT_ROOT}}
data_dir: ${{paths.root_dir}}/data/
log_dir: ${{paths.root_dir}}/logs/
output_dir: ${{hydra:runtime.output_dir}}
work_dir: ${{hydra:runtime.cwd}}

subset2episode: {ad}/subset2episode.json
vocab_pkl: {ad}/vocab.pkl
info_pkl: {ad}/info.pkl
annotations_pkl: {ad}/pl_lmdb
vid_features_lmdb: {ad}/feats_lmdb
subtitles_path: {ad}/subtitles.pkl
aligned_subtitles_path: {ad}/subtitles.pkl
synonyms_pkl: {ad}/synonyms.pkl
llm_root: meta-llama/Llama-3.2-3B   # <-- set to your downloaded model path
bleurt_path: lucadiliello/BLEURT-20
blip_cap_path: {ad}/blip.pkl
val_episode_ind_path: {ad}/val_start_indices.json
test_episode_ind_path: {ad}/test_start_indices.json
train_cap_path: {ad}/train_cap.json
spottings_path: null
"""
    open(os.path.join(out_dir, "paths_tiny.yaml"), "w").write(y)
    print(f"[tiny] wrote paths_tiny.yaml (edit llm_root before running)")


def generate(out_dir, episodes):
    os.makedirs(out_dir, exist_ok=True)
    write_lmdbs(out_dir, episodes)
    write_metadata(out_dir, episodes)
    write_paths_yaml(out_dir)
    print(f"\n[tiny] DONE -> {os.path.abspath(out_dir)}")
    print("Run the real pipeline (on Kaggle/where torchvision is installed):")
    print("  cp <out>/paths_tiny.yaml configs/paths/tiny.yaml")
    print("  python src/train.py experiment=vid+pg+prev+bg paths=tiny trainer=gpu "
          "trainer.devices=[0] trainer.precision=16-mixed trainer.max_epochs=2 "
          "data.batch_size=1 model.net.mm_projector_config.hidden_size=3072 "
          "model.net.llm_config.decoder_config.attn_implementation=eager")


def verify(out_dir):
    """Round-trip the LMDBs and instantiate the real Subtitles dataset."""
    import lmdb
    env = lmdb.open(os.path.join(out_dir, "feats_lmdb"), readonly=True, subdir=True)
    with env.begin() as txn:
        k, v = next(iter(txn.cursor()))
        arr = np.frombuffer(v, dtype=np.float16)
    env.close()
    assert arr.shape == (FEAT_DIM,), arr.shape
    print(f"[verify] feats key {k.decode()} -> float16 dim {arr.shape[0]}  OK")

    env = lmdb.open(os.path.join(out_dir, "pl_lmdb"), readonly=True, subdir=True)
    with env.begin() as txn:
        lab = np.frombuffer(txn.get(next(k for k, _ in txn.cursor() if b"_label/" in k)), dtype=np.int64)
    env.close()
    assert lab.shape == (5,) and lab.max() < len(VOCAB), lab
    print(f"[verify] pl label -> int64 x{lab.shape[0]}, ids in vocab  OK")

    # exercise the real metadata parser (no torchvision needed for Subtitles)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    import types
    u = types.ModuleType("src.utils"); u.__path__ = [os.path.join(root, "src", "utils")]
    sys.modules["src.utils"] = u
    from src.data.components.subtitles import Subtitles
    for setname in ("train", "val"):
        ds = Subtitles(
            subset2episode=os.path.join(out_dir, "subset2episode.json"), setname=setname,
            subtitles_path=os.path.join(out_dir, "subtitles.pkl"),
            subtitles_temporal_shift=0.0, subtitles_max_duration=20.0,
            subtitles_min_duration=1.0, temporal_pad=1.0,
            info_pkl=os.path.join(out_dir, "info.pkl"), max_previous_sentences=1,
            subtitles_random_offset=0.0, fps=FPS,
            blip_cap_path=os.path.join(out_dir, "blip.pkl"), aug_prev=True,
            synonyms_pkl=os.path.join(out_dir, "synonyms.pkl"),
            train_cap_path=os.path.join(out_dir, "train_cap.json"), train_cap_prob=0.0,
            aligned_subtitles_path=os.path.join(out_dir, "subtitles.pkl"))
        item = ds[0]
        print(f"[verify] Subtitles('{setname}') len={len(ds)}  item0: "
              f"sub={item['subtitle']!r} prev={item['previous_context']!r} "
              f"bg={item['bg_description'][:40]!r}")
    print("\n[verify] PASS: LMDB + metadata formats accepted by the real loaders.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="./tiny_bobsl")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    EPISODES = {"ep001": 40.0, "ep002": 40.0, "ep003": 40.0}  # name -> duration seconds
    if not a.verify:
        generate(a.out_dir, EPISODES)
    verify(a.out_dir)
