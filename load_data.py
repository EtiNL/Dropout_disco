import os
from os.path import join as pjoin

import pandas as pd


def load_train_data(
    data_root: str = "./data",
    augmented_text: bool = False,
    n_desc_per_motion: int | None = None,
):
    """
    Args:
      data_root: dataset root containing motions/ and texts/ (and optionally augmented_texts/)
      augmented_text: if True, reads text files from {data_root}/augmented_texts/augmented_texts
                      otherwise reads from {data_root}/texts/texts
      n_desc_per_motion: number of descriptions to keep per motion in map_df.
                         Defaults to 6 if augmented_text=True, else 3.

    Returns:
      motion_ids   : list[int]
      motion_paths : list[str] (aligned with motion_ids)
      map_df       : DataFrame [motion_id, text_id_1, ..., text_id_{n_desc_per_motion}]
      text_df      : DataFrame [text_id, motion_id, description]
    """
    motion_dir = pjoin(data_root, "motions", "motions")
    text_dir = (
        pjoin(data_root, "augmented_texts", "augmented_texts")
        if augmented_text
        else pjoin(data_root, "texts", "texts")
    )
    train_list = pjoin(data_root, "train.txt")

    if n_desc_per_motion is None:
        n_desc_per_motion = 6 if augmented_text else 3
    if n_desc_per_motion <= 0:
        raise ValueError("n_desc_per_motion must be a positive integer.")

    with open(train_list, "r", encoding="utf-8") as f:
        motion_ids = [int(x) for x in f.read().splitlines() if x.strip()]

    motion_paths = [pjoin(motion_dir, f"{mid}.npy") for mid in motion_ids]

    text_rows, map_rows = [], []
    for mid in motion_ids:
        txt_path = pjoin(text_dir, f"{mid}.txt")
        lines = []
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as fr:
                lines = [ln.strip() for ln in fr.read().splitlines() if ln.strip()]

        tids = []
        for k, desc in enumerate(lines, start=1):
            tid = f"{mid}_{k}"
            tids.append(tid)
            text_rows.append({"text_id": tid, "motion_id": mid, "description": desc})

        chosen = (tids + [pd.NA] * n_desc_per_motion)[:n_desc_per_motion]
        row = {"motion_id": mid}
        for j in range(n_desc_per_motion):
            row[f"text_id_{j+1}"] = chosen[j]
        map_rows.append(row)

    text_df = pd.DataFrame(text_rows, columns=["text_id", "motion_id", "description"])

    map_cols = ["motion_id"] + [f"text_id_{j+1}" for j in range(n_desc_per_motion)]
    map_df = pd.DataFrame(map_rows, columns=map_cols)

    return motion_ids, motion_paths, map_df, text_df
