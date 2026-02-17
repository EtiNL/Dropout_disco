import os
from os.path import join as pjoin

import pandas as pd


def load_train_data(data_root: str = "./"):
    """
    Memory-friendly version:
      - does NOT build a giant padded motion_tensor
      - returns motion_paths instead

    Returns:
      motion_ids   : list[int]
      motion_paths : list[str] (aligned with motion_ids)
      map_df       : DataFrame [motion_id, text_id_1, text_id_2, text_id_3]
      text_df      : DataFrame [text_id, motion_id, description]
    """
    motion_dir = pjoin(data_root, "motions", "motions")
    text_dir = pjoin(data_root, "texts", "texts")
    train_list = pjoin(data_root, "train.txt")

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

        chosen = (tids + [pd.NA, pd.NA, pd.NA])[:3]
        map_rows.append({"motion_id": mid, "text_id_1": chosen[0], "text_id_2": chosen[1], "text_id_3": chosen[2]})

    text_df = pd.DataFrame(text_rows, columns=["text_id", "motion_id", "description"])
    map_df = pd.DataFrame(map_rows, columns=["motion_id", "text_id_1", "text_id_2", "text_id_3"])

    return motion_ids, motion_paths, map_df, text_df

