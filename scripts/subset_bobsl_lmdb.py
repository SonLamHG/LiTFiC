"""
Carve a small, REAL subset out of the full BOBSL LMDBs so it fits on Kaggle.

The full BOBSL feature LMDB is ~262 GB (see docs/litfic-kaggle-runbook.md). This
script copies only the keys belonging to a chosen handful of episodes into small
output LMDBs, and writes a matching `subset2episode.json`. Run it ONCE on a machine
with the full LMDBs on local disk (e.g. a cloud VM that pulled them from your Drive),
then upload the tiny outputs to Kaggle.

Key formats (must match src/data/components/lmdb_loader.py):
  features LMDB:       "<episode>/<frame+1:07d>.np"
  pseudo-label LMDB:   "<episode>_label/<frame+1:07d>.np"
                       "<episode>_prob/<frame+1:07d>.np"
where "<episode>" = Path(episode_name.split('.')[0]).stem.

Usage (real):
  python scripts/subset_bobsl_lmdb.py \
    --feats-src   /data/bobsl/lmdb-feats_vswin_t-bs256_float16 \
    --pl-src      /data/bobsl/lmdb-pl_vswin_t-bs256_float16 \
    --out-dir     /data/bobsl_subset \
    --episodes    <ep1> <ep2> <ep3> \
    --train <ep1> <ep2> --val <ep3> --test <ep3>

Self-test (no real data; builds a fake source LMDB and verifies the copy):
  python scripts/subset_bobsl_lmdb.py --selftest
"""
import os, json, argparse
import lmdb


def _episode_prefixes(episodes, kind):
    """Byte prefixes (before the first '/') that belong to `episodes`."""
    prefixes = set()
    for ep in episodes:
        stem = os.path.splitext(ep)[0]
        if kind == "feats":
            prefixes.add(stem.encode("ascii"))
        else:  # pseudo-labels: two prefixes per episode
            prefixes.add(f"{stem}_label".encode("ascii"))
            prefixes.add(f"{stem}_prob".encode("ascii"))
    return prefixes


def copy_subset(src_path, dst_path, episodes, kind, map_size=None):
    """Copy every key whose prefix (before b'/') is in the chosen episodes."""
    wanted = _episode_prefixes(episodes, kind)
    os.makedirs(dst_path, exist_ok=True)
    src = lmdb.open(src_path, readonly=True, lock=False, max_readers=512,
                    subdir=os.path.isdir(src_path))
    # generous default map size (4 GB) -- a few episodes are far smaller
    dst = lmdb.open(dst_path, map_size=map_size or 4 * 1024**3, subdir=True)
    n_copied = 0
    with src.begin() as rtxn, dst.begin(write=True) as wtxn:
        cur = rtxn.cursor()
        for key, val in cur:
            prefix = key.split(b"/", 1)[0]
            if prefix in wanted:
                wtxn.put(key, val)
                n_copied += 1
    src.close(); dst.close()
    print(f"[subset] {kind}: copied {n_copied} keys -> {dst_path}")
    return n_copied


def write_subset2episode(out_dir, train, val, test):
    stems = lambda eps: [os.path.splitext(e)[0] for e in eps]
    mapping = {"train": stems(train), "val": stems(val),
               "public_test": stems(test), "test": stems(test)}
    path = os.path.join(out_dir, "subset2episode.json")
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[subset] wrote {path}: "
          f"train={len(train)} val={len(val)} test={len(test)}")


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    eps = args.episodes or sorted(set(args.train + args.val + args.test))
    copy_subset(args.feats_src, os.path.join(args.out_dir, "feats_lmdb"),
                eps, "feats")
    copy_subset(args.pl_src, os.path.join(args.out_dir, "pl_lmdb"),
                eps, "pseudo-labels")
    write_subset2episode(args.out_dir, args.train or eps,
                         args.val or eps[-1:], args.test or eps[-1:])
    print("\n[subset] DONE. Next: regenerate val/test start-indices on Kaggle "
          "(see docs/litfic-kaggle-runbook.md), then upload out-dir as a Kaggle Dataset.")


# --------------------------------------------------------------------------- #
def selftest():
    """Build a tiny fake source in the exact key format and verify the copy."""
    import tempfile, struct
    tmp = tempfile.mkdtemp(prefix="bobsl_selftest_")
    feats_src = os.path.join(tmp, "feats_src")
    pl_src = os.path.join(tmp, "pl_src")
    episodes = ["epA", "epB", "epC"]  # copy A and C, drop B

    def mk(path, kind):
        env = lmdb.open(path, map_size=64 * 1024**2, subdir=True)
        with env.begin(write=True) as txn:
            for ep in episodes:
                for fr in range(3):
                    end = f"{fr+1:07d}.np"
                    if kind == "feats":
                        txn.put(f"{ep}/{end}".encode(), b"\x00" * 4)
                    else:
                        txn.put(f"{ep}_label/{end}".encode(), b"\x00" * 4)
                        txn.put(f"{ep}_prob/{end}".encode(), b"\x00" * 4)
        env.close()
    mk(feats_src, "feats"); mk(pl_src, "pl")

    out = os.path.join(tmp, "out")
    chosen = ["epA", "epC"]
    n_f = copy_subset(feats_src, os.path.join(out, "feats_lmdb"), chosen, "feats")
    n_p = copy_subset(pl_src, os.path.join(out, "pl_lmdb"), chosen, "pseudo-labels")
    write_subset2episode(out, ["epA"], ["epC"], ["epC"])

    # verify: epB must be absent, epA/epC present with right counts
    env = lmdb.open(os.path.join(out, "feats_lmdb"), readonly=True, subdir=True)
    with env.begin() as txn:
        keys = [k.decode() for k, _ in txn.cursor()]
    env.close()
    assert n_f == 6 and n_p == 12, (n_f, n_p)          # 2 eps x 3 frames
    assert not any(k.startswith("epB") for k in keys)   # dropped episode gone
    assert sum(k.startswith("epA") for k in keys) == 3
    print("\n[selftest] PASS: subset copies only chosen episodes, key format intact.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--feats-src"); p.add_argument("--pl-src")
    p.add_argument("--out-dir", default="./bobsl_subset")
    p.add_argument("--episodes", nargs="*", default=None)
    p.add_argument("--train", nargs="*", default=[])
    p.add_argument("--val", nargs="*", default=[])
    p.add_argument("--test", nargs="*", default=[])
    a = p.parse_args()
    if a.selftest:
        selftest()
    else:
        assert a.feats_src and a.pl_src, "provide --feats-src and --pl-src (or --selftest)"
        run(a)
