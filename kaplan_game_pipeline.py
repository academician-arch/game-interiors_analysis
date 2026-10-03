import argparse
import io
import json
import math
import os
import pickle
import sys
import time
import zipfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests
from scipy import stats, optimize
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it=None, **kw):
        return it if it is not None else _NullBar()
    class _NullBar:
        def update(self, *a, **k): pass
        def set_postfix(self, *a, **k): pass
        def close(self): pass

GENRE_TAGS = {
    "Horror": "Horror",
    "Life Sim": "Life Sim",
    "Exploration": "Walking Simulator",
    "Dungeon Crawler": "Dungeon Crawler",
}
DIMS = ["Coherence", "Complexity", "Legibility", "Mystery"]

CFG = dict(
    N_GAMES_PER_GENRE=100,
    MAX_CANDIDATES_PER_GENRE=1000,
    CORE_TAG_TOP_K=5,
    STRICT_TAG_TOP_K=3,
    MAX_SHOTS_SCAN=10,
    N_INDOOR_PER_GAME=4,
    INDOOR_PROB_THRESHOLD=0.5,
    CLIP_INDOOR_THRESHOLD=0.5,
    CLIP_NAME="openai/clip-vit-base-patch32",
    SPY_DELAY_SEC=1.05,
    STEAM_DELAY_SEC=1.5,
    N_PERMUTATIONS=2000,
    N_BOOTSTRAP=5000,
    RNG_SEED=42,
    AUDIT_N=48,
    HAND_CODE_N=30,
    COOLDOWN_SEC=300,
)

OFFICIAL_STEAM_GENRES = [
    "Action", "Adventure", "Casual", "Early Access", "Free to Play", "Indie",
    "Massively Multiplayer", "Racing", "RPG", "Simulation", "Sports", "Strategy",
]

P = SimpleNamespace()


def init_paths(root):
    root = Path(root).resolve()
    P.root = root
    P.img = root / "images"
    P.fig = root / "figures"
    P.res = root / "results"
    P.cache = root / "cache"
    P.reject = root / "rejected_thumbnails"
    for d in (P.img, P.fig, P.res, P.cache, P.reject):
        d.mkdir(parents=True, exist_ok=True)
    return P


def log(*a):
    print(*a, flush=True)


def save_table(df, name, index=True):
    p = P.res / name
    df.to_csv(p, sep=";", decimal=",", index=index, encoding="utf-8-sig")
    return p


def savefig(fig, name):
    fig.savefig(P.fig / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def rel(path):
    return Path(path).resolve().relative_to(P.root).as_posix()


def absp(relpath):
    return P.root / relpath


def robust_get(url, params=None, max_retries=6, base_sleep=2.0, max_sleep=120.0, timeout=30):
    host = url.split("/")[2] if "//" in url else url
    for attempt in range(max_retries):
        wait = min(base_sleep * (2 ** attempt), max_sleep)
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 403, 500, 502, 503, 504):
                if wait >= 8:
                    log(f"  [waiting] {host} HTTP {r.status_code} (possible rate limit); retrying in {wait:.0f} s "
                        f"({attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            return None
        except Exception as e:
            if wait >= 8:
                log(f"  [waiting] {host} connection error ({type(e).__name__}); retrying in {wait:.0f} s "
                    f"({attempt + 1}/{max_retries})")
            time.sleep(wait)
    return None


def fetch_bytes(url):
    if not url:
        return None
    r = robust_get(url, max_retries=3, base_sleep=1.0, max_sleep=10.0)
    return r.content if r is not None else None


_spy = SimpleNamespace(cache={}, new_calls=0)


def load_spy_cache():
    path = P.cache / "steamspy_appdetails_cache.json"
    if path.exists():
        try:
            _spy.cache = json.loads(path.read_text(encoding="utf-8"))
            log(f"SteamSpy cache loaded: {len(_spy.cache)} games")
        except Exception:
            _spy.cache = {}


def save_spy_cache():
    (P.cache / "steamspy_appdetails_cache.json").write_text(json.dumps(_spy.cache), encoding="utf-8")


def get_candidates_by_tag(tag, max_n):
    r = robust_get("https://steamspy.com/api.php", {"request": "tag", "tag": tag})
    if r is None:
        log(f"[WARNING] Could not retrieve the SteamSpy tag list for '{tag}'.")
        return [], 0
    try:
        data = r.json()
    except Exception:
        return [], 0
    ids = list(data.keys())
    return ids[:max_n], len(ids)


def steamspy_tags(appid):
    key = str(appid)
    if key in _spy.cache:
        return _spy.cache[key]
    for attempt in range(2):
        time.sleep(CFG["SPY_DELAY_SEC"])
        r = robust_get("https://steamspy.com/api.php", {"request": "appdetails", "appid": appid}, max_retries=4)
        if r is None:
            continue
        try:
            d = r.json()
        except Exception:
            time.sleep(5 * (attempt + 1))
            continue
        if not isinstance(d, dict):
            continue
        tags = d.get("tags")
        tags = tags if isinstance(tags, dict) else {}
        _spy.cache[key] = {"name": d.get("name"), "tags": tags}
        _spy.new_calls += 1
        if _spy.new_calls % 25 == 0:
            save_spy_cache()
        return _spy.cache[key]
    return None


def tag_ranks(tags_dict):
    ranked = sorted(tags_dict.items(), key=lambda kv: kv[1], reverse=True)
    names = [t.lower() for t, _ in ranked]
    return {g: (names.index(tag.lower()) + 1 if tag.lower() in names else np.nan) for g, tag in GENRE_TAGS.items()}


def primary_genre(ranks):
    valid = {g: r for g, r in ranks.items() if not np.isnan(r)}
    if not valid:
        return None, np.nan
    g = min(valid, key=valid.get)
    return g, valid[g]


def get_steam_details(appid):
    for attempt in range(3):
        time.sleep(CFG["STEAM_DELAY_SEC"])
        r = robust_get("https://store.steampowered.com/api/appdetails",
                       {"appids": appid, "l": "english"}, max_retries=4)
        if r is None:
            continue
        try:
            payload = r.json()
        except Exception:
            payload = None
        if payload is None:
            time.sleep(10 * (attempt + 1))
            continue
        entry = payload.get(str(appid), {})
        if entry.get("success"):
            return entry["data"], "ok"
        return None, "store_unavailable"
    return None, "steam_failed"


M = SimpleNamespace(loaded_gate=False, loaded_midas=False)

INDOOR_PROMPTS = [
    "a video game screenshot of the inside of a room in a building",
    "a video game screenshot of a corridor or hallway inside a building",
    "a video game screenshot of the inside of a dungeon, cave or crypt",
    "a video game screenshot of the inside of a spaceship, space station or vehicle",
    "a video game screenshot of a home interior with furniture",
]
OUTDOOR_PROMPTS = [
    "a video game screenshot of an outdoor landscape with sky and terrain",
    "a video game screenshot of a city street or town seen outdoors",
    "a video game screenshot of outer space or the outside of a spaceship or space station",
    "a video game screenshot of a forest, field or mountains in an open world",
    "a video game screenshot of a building seen from the outside",
]
NONSCENE_PROMPTS = [
    "a promotional collage with review scores, award laurels and logos",
    "a game title screen or logo on a dark background",
    "a menu, inventory or character stats screen made mostly of text",
    "a completely black or blank screen",
    "key art or an illustration of a character",
    "a close-up portrait of a character's face",
    "a game map or diagram",
]
ALL_PROMPTS = INDOOR_PROMPTS + OUTDOOR_PROMPTS + NONSCENE_PROMPTS
N_IN, N_OUT = len(INDOOR_PROMPTS), len(OUTDOOR_PROMPTS)


def download_file(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    log(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gate_models():
    if M.loaded_gate:
        return
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T
    M.torch = torch
    M.device = get_device()
    log("Cihaz:", M.device, "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (will be slow)"))

    wfile = download_file("http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
                          P.cache / "resnet18_places365.pth.tar")
    iofile = download_file("https://raw.githubusercontent.com/CSAILVision/places365/master/IO_places365.txt",
                           P.cache / "IO_places365.txt")
    model = tvm.resnet18(num_classes=365)
    try:
        ckpt = torch.load(wfile, map_location="cpu")
    except Exception:
        ckpt = torch.load(wfile, map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state)
    M.places = model.eval().to(M.device)
    labels = []
    for line in iofile.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            labels.append(int(parts[-1]))
    M.io_labels = np.array(labels)
    M.places_tf = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    log(f"Places365 loaded ({(M.io_labels == 1).sum()} indoor / {(M.io_labels == 2).sum()} outdoor categories).")

    from transformers import CLIPModel, CLIPProcessor
    log("Loading CLIP:", CFG["CLIP_NAME"])
    M.clip = CLIPModel.from_pretrained(CFG["CLIP_NAME"]).to(M.device).eval()
    M.clip_proc = CLIPProcessor.from_pretrained(CFG["CLIP_NAME"])
    M.loaded_gate = True


def load_midas():
    if M.loaded_midas:
        return
    import torch
    M.torch = torch
    M.device = get_device()
    log("Loading MiDaS_small (torch.hub downloads it on first use)...")
    M.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(M.device).eval()
    M.midas_tf = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    M.loaded_midas = True


def places_indoor_prob_pil(img):
    x = M.places_tf(img).unsqueeze(0).to(M.device)
    with M.torch.no_grad():
        probs = M.torch.nn.functional.softmax(M.places(x), dim=1).cpu().numpy()[0]
    return float(probs[M.io_labels == 1].sum())


def clip_class_probs(img):
    inputs = M.clip_proc(text=ALL_PROMPTS, images=img, return_tensors="pt", padding=True).to(M.device)
    with M.torch.no_grad():
        logits = M.clip(**inputs).logits_per_image[0]
    p = logits.softmax(dim=-1).cpu().numpy()
    return {"clip_indoor": float(p[:N_IN].sum()),
            "clip_outdoor": float(p[N_IN:N_IN + N_OUT].sum()),
            "clip_nonscene": float(p[N_IN + N_OUT:].sum())}


def scene_gate_pil(img):
    p_places = places_indoor_prob_pil(img)
    c = clip_class_probs(img)
    accept = (p_places >= CFG["INDOOR_PROB_THRESHOLD"]) and (c["clip_indoor"] >= CFG["CLIP_INDOOR_THRESHOLD"])
    luma = float(np.asarray(img.convert("L"), dtype=np.float32).mean() / 255.0)
    return {"places_indoor": p_places, **c, "mean_luma": luma, "accept": bool(accept)}


def scene_gate_from_bytes(content):
    from PIL import Image
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return scene_gate_pil(img)


TRANSIENT_STATUSES = ("steamspy_failed", "steam_failed", "full_download_failed")


def run_sampling(force=False):
    state_path = P.cache / "sampling_state.pkl"
    final_csv = P.res / "00_corpus_index.csv"
    state = dict(corpus_records=[], game_records=[], retrieval_log=[], shot_log=[], pool_sizes={}, done=[])
    if state_path.exists() and not force:
        state = pickle.loads(state_path.read_bytes())
        log(f"Resuming: completed genres = {state['done']}, accepted games = {len(state['game_records'])}")
    n_drop = sum(1 for r in state["retrieval_log"] if r["status"] in TRANSIENT_STATUSES)
    if n_drop:
        state["retrieval_log"] = [r for r in state["retrieval_log"] if r["status"] not in TRANSIENT_STATUSES]
        for g in list(state["done"]):
            n_acc = sum(1 for r in state["game_records"] if r["genre"] == g)
            if n_acc < CFG["N_GAMES_PER_GENRE"]:
                state["done"].remove(g)
        log(f"{n_drop} candidates with transient errors (network/API limits) will be retried.")
    if len(state["done"]) == len(GENRE_TAGS) and final_csv.exists() and not force:
        log("Sampling already completed; skipping (use --force-sample to redo it).")
        return load_corpus()

    load_spy_cache()
    load_gate_models()

    def save_state():
        state_path.write_bytes(pickle.dumps(state))

    for genre_label, tag in GENRE_TAGS.items():
        if genre_label in state["done"]:
            continue
        gdir = P.img / genre_label
        gdir.mkdir(parents=True, exist_ok=True)
        cand_ids, pool_total = get_candidates_by_tag(tag, CFG["MAX_CANDIDATES_PER_GENRE"])
        state["pool_sizes"][genre_label] = pool_total
        seen = {r["game_id"] for r in state["retrieval_log"] if r["genre"] == genre_label}
        n_accepted = sum(1 for r in state["game_records"] if r["genre"] == genre_label)
        log(f"{genre_label}: SteamSpy pool = {pool_total} games; scanning at most {len(cand_ids)} candidates "
            f"(target {CFG['N_GAMES_PER_GENRE']}, already {n_accepted})")
        pbar = tqdm(cand_ids, desc=genre_label)
        examined = 0
        consec_fail = 0
        for appid_str in pbar:
            if n_accepted >= CFG["N_GAMES_PER_GENRE"]:
                break
            appid = int(appid_str)
            if appid in seen:
                continue
            log_row = {"genre": genre_label, "game_id": appid, "status": None, "tag_rank": np.nan,
                       "n_scanned": 0, "n_indoor_selected": 0, "n_saved": 0}
            status = _process_candidate(genre_label, tag, appid, gdir, log_row, state)
            log_row["status"] = status
            state["retrieval_log"].append(log_row)
            if status in TRANSIENT_STATUSES:
                consec_fail += 1
                if consec_fail >= 5:
                    log(f"  [warning] {consec_fail} consecutive transient errors (possible API rate limit); "
                        f"cooling down for {CFG['COOLDOWN_SEC']} s...")
                    time.sleep(CFG["COOLDOWN_SEC"])
                    consec_fail = 0
            else:
                consec_fail = 0
            if status == "accepted":
                n_accepted += 1
                try:
                    pbar.set_postfix(accepted=n_accepted)
                except Exception:
                    pass
            examined += 1
            if examined % 10 == 0:
                save_state()
        state["done"].append(genre_label)
        save_spy_cache()
        save_state()
        log(f"  -> {genre_label}: {n_accepted} games accepted.\n")

    return finalize_corpus(state)


def _process_candidate(genre_label, tag, appid, gdir, log_row, state):
    spy = steamspy_tags(appid)
    if spy is None:
        return "steamspy_failed"
    ranks = tag_ranks(spy["tags"])
    primary, prank = primary_genre(ranks)
    log_row["tag_rank"] = ranks.get(genre_label, np.nan)
    if primary is None:
        return "target_tag_not_in_top20"
    if primary != genre_label:
        return "other_primary_genre"
    if prank > CFG["CORE_TAG_TOP_K"]:
        return "tag_below_top_k"

    details, st = get_steam_details(appid)
    if details is None:
        return st
    shots = details.get("screenshots", [])[:CFG["MAX_SHOTS_SCAN"]]
    if not shots:
        return "no_screenshots"

    thumb_urls = [s.get("path_thumbnail") or s.get("path_full") for s in shots]
    with ThreadPoolExecutor(max_workers=6) as ex:
        thumbs = list(ex.map(fetch_bytes, thumb_urls))
    selected, n_scanned = [], 0
    for idx, (shot, tb) in enumerate(zip(shots, thumbs)):
        if tb is None:
            continue
        try:
            gate = scene_gate_from_bytes(tb)
        except Exception:
            continue
        n_scanned += 1
        state["shot_log"].append({"genre": genre_label, "game_id": appid, "shot_index": idx, **gate})
        if gate["accept"]:
            selected.append((idx, shot, gate))
            if len(selected) >= CFG["N_INDOOR_PER_GAME"]:
                break
        else:
            (P.reject / f"{appid}_{idx}.jpg").write_bytes(tb)
    log_row["n_scanned"] = n_scanned
    log_row["n_indoor_selected"] = len(selected)
    if not selected:
        return "no_interior_scene_passed_gate"

    with ThreadPoolExecutor(max_workers=4) as ex:
        full_bytes = list(ex.map(fetch_bytes, [s[1].get("path_full") for s in selected]))
    steam_genres = ", ".join(g.get("description", "") for g in details.get("genres", []))
    game_name = details.get("name", str(appid))
    saved = 0
    for (idx, shot, gate), content in zip(selected, full_bytes):
        if content is None:
            continue
        fpath = gdir / f"{appid}_{idx}.jpg"
        fpath.write_bytes(content)
        state["corpus_records"].append({
            "genre": genre_label, "appid": appid, "game_id": appid, "game_name": game_name,
            "steam_genres": steam_genres, "image_path": rel(fpath), "shot_index": idx,
            "indoor_probability": gate["places_indoor"], "clip_indoor": gate["clip_indoor"],
            "clip_outdoor": gate["clip_outdoor"], "clip_nonscene": gate["clip_nonscene"], "tag_rank": prank})
        saved += 1
    log_row["n_saved"] = saved
    if saved == 0:
        return "full_download_failed"
    rec = {"game_id": appid, "genre": genre_label, "game_name": game_name, "target_tag": tag,
           "tag_rank": prank, "n_tags_total": len(spy["tags"]), "is_core_genre": True}
    for g in GENRE_TAGS:
        rec[f"rank_{g.replace(' ', '_')}"] = ranks[g]
    state["game_records"].append(rec)
    return "accepted"


def finalize_corpus(state):
    corpus_df = pd.DataFrame(state["corpus_records"])
    corpus_df["is_indoor"] = True
    corpus_df["is_core_genre"] = True
    tag_rank_df = pd.DataFrame(state["game_records"])
    retrieval_df = pd.DataFrame(state["retrieval_log"])
    shot_df = pd.DataFrame(state["shot_log"])
    save_table(corpus_df, "00_corpus_index.csv", index=False)
    save_table(tag_rank_df, "00e_tag_prominence_accepted_games.csv", index=False)
    if len(tag_rank_df):
        hc = pd.concat([d.sample(min(CFG["HAND_CODE_N"], len(d)), random_state=CFG["RNG_SEED"])
                        for _, d in tag_rank_df.groupby("genre")], ignore_index=True)[["genre", "game_id", "game_name", "tag_rank"]].copy()
        hc["primary_genre_manual_yes_no"] = ""
        save_table(hc, "00f_handcoding_sample.csv", index=False)
    save_table(retrieval_df, "00h_retrieval_log.csv", index=False)
    funnel = retrieval_df.groupby(["genre", "status"]).size().unstack(fill_value=0)
    funnel["candidates_examined"] = funnel.sum(axis=1)
    funnel["steamspy_pool_total"] = pd.Series(state["pool_sizes"])
    save_table(funnel, "00g_sampling_funnel.csv")
    if len(shot_df):
        save_table(shot_df, "00j_shot_gate_log_raw.csv", index=False)
    log(f"Total images: {len(corpus_df)} | unique games: {corpus_df['game_id'].nunique()}")
    log(corpus_df.groupby("genre").agg(images=("image_path", "count"), games=("game_id", "nunique")).to_string())
    log(funnel.to_string())
    return corpus_df


def load_corpus():
    return pd.read_csv(P.res / "00_corpus_index.csv", sep=";", decimal=",", encoding="utf-8-sig")


def box_counting_dimension(binary_image, min_box=2):
    Z = np.asarray(binary_image, dtype=bool)
    H, W = Z.shape
    p = min(H, W)
    n = 2 ** int(np.floor(np.log2(p)))
    max_power = int(np.log2(n))
    sizes = 2 ** np.arange(int(np.log2(min_box)), max_power - 1, 1)
    counts = []
    for size in sizes:
        ph, pw = (-H) % size, (-W) % size
        Zp = np.pad(Z, ((0, ph), (0, pw)), constant_values=False) if (ph or pw) else Z
        blocks = Zp.reshape(Zp.shape[0] // size, size, Zp.shape[1] // size, size).any(axis=(1, 3))
        counts.append(int(blocks.sum()))
    counts = np.array(counts)
    valid = counts > 0
    if valid.sum() < 2:
        return np.nan
    return np.polyfit(np.log(1.0 / sizes[valid]), np.log(counts[valid]), 1)[0]


def compute_complexity(image_bgr):
    import cv2
    from skimage import feature, measure
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    glcm = feature.graycomatrix(gray, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                levels=256, symmetric=True, normed=True)
    glcm_contrast = float(feature.graycoprops(glcm, "contrast").mean())
    entropy = float(measure.shannon_entropy(gray))
    fractal_dim = float(box_counting_dimension(edges > 0))
    return {"edge_density": edge_density, "glcm_contrast": glcm_contrast,
            "entropy": entropy, "fractal_dimension": fractal_dim}


def compute_coherence(image_bgr):
    import cv2
    from skimage.metrics import structural_similarity as ssim
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    flipped = cv2.flip(gray, 1)
    symmetry_index = float(ssim(gray, flipped))
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue_rad = np.deg2rad(hsv[:, :, 0].astype(np.float32) * 2.0)
    R = np.sqrt(np.mean(np.sin(hue_rad)) ** 2 + np.mean(np.cos(hue_rad)) ** 2)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    orientations = np.arctan2(sobely, sobelx)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    mask = mag > np.percentile(mag, 75)
    if mask.sum() > 0:
        Rc = np.sqrt(np.mean(np.sin(2 * orientations[mask])) ** 2 + np.mean(np.cos(2 * orientations[mask])) ** 2)
        orientation_regularity = float(Rc)
    else:
        orientation_regularity = np.nan
    return {"symmetry_index": symmetry_index, "color_harmony_concentration": float(R),
            "orientation_regularity": orientation_regularity}


def compute_legibility(depth_norm):
    h, w = depth_norm.shape
    openness_ratio = float(np.mean(depth_norm > np.median(depth_norm)))
    gy, gx = np.gradient(depth_norm)
    depth_gradient_smoothness = float(1.0 / (1.0 + np.mean(np.sqrt(gx ** 2 + gy ** 2))))
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)
    ray_means = []
    for a in np.linspace(0, 2 * np.pi, 36, endpoint=False):
        rs = np.arange(5, max_r, 5)
        xs = (cx + rs * np.cos(a)).astype(int)
        ys = (cy + rs * np.sin(a)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if valid.sum() > 0:
            ray_means.append(depth_norm[ys[valid], xs[valid]].mean())
    isovist = float(np.mean(ray_means)) if ray_means else np.nan
    return {"openness_ratio": openness_ratio, "depth_gradient_smoothness": depth_gradient_smoothness,
            "isovist_openness_proxy": isovist}


def compute_mystery(depth_norm, image_bgr):
    import cv2
    h, w = depth_norm.shape
    depth_edges = cv2.Canny((depth_norm * 255).astype(np.uint8), 30, 90)
    occlusion_density = float(np.mean(depth_edges > 0))
    border = max(5, int(0.08 * min(h, w)))
    mb = np.zeros((h, w), dtype=bool)
    mb[:border, :] = True; mb[-border:, :] = True; mb[:, :border] = True; mb[:, -border:] = True
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200) > 0
    peripheral_opening_ratio = float(edges[mb].mean()) if mb.any() else np.nan
    pdvr = float(depth_norm[mb].var()) / (float(depth_norm[~mb].var()) + 1e-8)
    return {"occlusion_density": occlusion_density, "peripheral_opening_ratio": peripheral_opening_ratio,
            "peripheral_depth_variance_ratio": pdvr}


def estimate_depth(image_bgr):
    import cv2
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    batch = M.midas_tf(img_rgb).to(M.device)
    with M.torch.no_grad():
        pred = M.midas(batch)
        pred = M.torch.nn.functional.interpolate(pred.unsqueeze(1), size=img_rgb.shape[:2],
                                                 mode="bicubic", align_corners=False).squeeze()
    d = pred.cpu().numpy()
    return (d - d.min()) / (d.max() - d.min() + 1e-8)


def run_metrics(corpus_df):
    import cv2
    partial = P.cache / "metrics_partial.csv"
    done_paths = set()
    if partial.exists():
        done_paths = set(pd.read_csv(partial)["image_path"])
        log(f"Metric cache: {len(done_paths)} images already computed.")
    todo = corpus_df[~corpus_df["image_path"].isin(done_paths)]
    if len(todo):
        load_midas()
        buffer = []
        carry = ["genre", "appid", "game_id", "game_name", "image_path", "tag_rank", "indoor_probability",
                 "clip_indoor", "clip_outdoor", "clip_nonscene"]

        def flush():
            if buffer:
                pd.DataFrame(buffer).to_csv(partial, mode="a", header=not partial.exists(), index=False)
                buffer.clear()

        for _, row in tqdm(list(todo.iterrows()), desc="Kaplan metrics"):
            img = cv2.imread(str(absp(row["image_path"])))
            if img is None:
                continue
            try:
                depth = estimate_depth(img)
                rec = {k: row[k] for k in carry if k in row.index}
                rec.update({f"complexity_{k}": v for k, v in compute_complexity(img).items()})
                rec.update({f"coherence_{k}": v for k, v in compute_coherence(img).items()})
                rec.update({f"legibility_{k}": v for k, v in compute_legibility(depth).items()})
                rec.update({f"mystery_{k}": v for k, v in compute_mystery(depth, img).items()})
                buffer.append(rec)
            except Exception as e:
                log(f"Skipped ({row['image_path']}): {e}")
            if len(buffer) >= 25:
                flush()
        flush()
    m = pd.read_csv(partial)
    m = m[m["image_path"].isin(set(corpus_df["image_path"]))].reset_index(drop=True)
    return add_composites(m)


def add_composites(m):
    m = m.copy()
    for dim in DIMS:
        cols = [c for c in m.columns if c.startswith(dim.lower() + "_")]
        z = m[cols].apply(lambda s: stats.zscore(s, nan_policy="omit"))
        m[dim] = z.mean(axis=1)
    return m


def submeasure_cols(m):
    return {dim: [c for c in m.columns if c.startswith(dim.lower() + "_")] for dim in DIMS}


def extract_numeric_id(x):
    digits = "".join(ch for ch in str(x) if ch.isdigit())
    return int(digits) if digits else np.nan


def fetch_savoias():
    target = P.cache / "Savoias-Dataset"
    if target.exists() and any(target.iterdir()):
        return target
    if os.system(f'git clone --depth 1 https://github.com/esaraee/Savoias-Dataset.git "{target}"') == 0 and target.exists():
        return target
    for branch in ("master", "main"):
        try:
            zpath = download_file(f"https://github.com/esaraee/Savoias-Dataset/archive/refs/heads/{branch}.zip",
                                  P.cache / f"savoias_{branch}.zip")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(P.cache)
            ex = next(P.cache.glob("Savoias-Dataset-*"), None)
            if ex is not None:
                ex.rename(target)
                return target
        except Exception as e:
            log(f"Could not download the SAVOIAS {branch} zip: {e}")
    return None


def locate_savoias(root):
    img_dir, gt = None, None
    for d in root.rglob("*"):
        if d.is_dir() and "interior" in d.name.lower():
            n_img = sum(1 for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if n_img >= 20:
                img_dir = d
                break
    cands = [f for f in root.rglob("*") if f.is_file() and "interior" in f.name.lower()
             and f.suffix.lower() in (".xlsx", ".xls", ".csv")]
    cands.sort(key=lambda f: (0 if "global_ranking" in f.name.lower() else 1,
                              0 if f.suffix.lower() in (".xlsx", ".xls") else 1, f.name))
    if cands:
        gt = cands[0]
    return img_dir, gt


def load_savoias_gt(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        if df.shape[1] == 1:
            out = pd.DataFrame({"image_id": df.iloc[:, 0].apply(extract_numeric_id),
                                "human_score": np.arange(1, len(df) + 1)})
        else:
            low = [str(c).lower() for c in df.columns]
            id_col = next((df.columns[i] for i, c in enumerate(low) if any(k in c for k in ("image", "name", "file", "id"))),
                          df.columns[0])
            num = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
            score_col = num[0] if num else df.columns[-1]
            out = pd.DataFrame({"image_id": df[id_col].apply(extract_numeric_id),
                                "human_score": pd.to_numeric(df[score_col], errors="coerce")})
    else:
        df = pd.read_csv(path, header=None)
        out = pd.DataFrame({"image_id": df.iloc[:, 0].apply(extract_numeric_id),
                            "human_score": pd.to_numeric(df.iloc[:, 1], errors="coerce")})
    out = out.dropna()
    out["image_id"] = out["image_id"].astype(int)
    return out


def run_savoias(metrics_cols):
    import cv2
    out_csv = P.res / "02_savoias_validation.csv"
    if out_csv.exists():
        log("SAVOIAS validation already exists; skipping (delete the file to rerun it).")
        return pd.read_csv(out_csv, sep=";", decimal=",", encoding="utf-8-sig")
    root = fetch_savoias()
    if root is None:
        log("[WARNING] Could not download the SAVOIAS repository; validation skipped.")
        return None
    img_dir, gt_file = locate_savoias(root)
    log("SAVOIAS image folder:", img_dir, "| ground-truth file:", gt_file)
    if img_dir is None or gt_file is None:
        log("[WARNING] SAVOIAS folder/file not found; validation skipped.")
        return None
    gt = load_savoias_gt(gt_file)
    rows = []
    files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for f in tqdm(files, desc="SAVOIAS complexity"):
        img = cv2.imread(str(f))
        if img is None:
            continue
        try:
            row = {"filename": f.name, "image_id": extract_numeric_id(f.name)}
            row.update({f"complexity_{k}": v for k, v in compute_complexity(img).items()})
            rows.append(row)
        except Exception:
            continue
    sv = pd.DataFrame(rows)
    cx = metrics_cols["Complexity"]
    sv["complexity_composite"] = sv[cx].apply(lambda s: stats.zscore(s, nan_policy="omit")).mean(axis=1)
    merged = sv.merge(gt, on="image_id", how="inner")
    log(f"SAVOIAS: {len(sv)} images processed, {len(merged)} matched to the human ranking.")
    if len(merged) < 10:
        log("[WARNING] Fewer than 10 matched images; validation skipped.")
        return None
    rho, p = stats.spearmanr(merged["human_score"], merged["complexity_composite"])
    rng = np.random.default_rng(CFG["RNG_SEED"])
    boots = []
    for _ in range(CFG["N_BOOTSTRAP"]):
        idx = rng.integers(0, len(merged), len(merged))
        r, _ = stats.spearmanr(merged["human_score"].to_numpy()[idx], merged["complexity_composite"].to_numpy()[idx])
        if not np.isnan(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res = pd.DataFrame([{"n_matched_images": len(merged), "spearman_rho": rho, "p_value": p,
                         "ci_95_lower": lo, "ci_95_upper": hi, "n_bootstrap": CFG["N_BOOTSTRAP"]}])
    save_table(res, "02_savoias_validation.csv", index=False)
    save_table(merged, "02b_savoias_matched_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged["human_score"], merged["complexity_composite"], alpha=0.6)
    ax.set_xlabel("SAVOIAS human-derived complexity rank")
    ax.set_ylabel("Our computed Complexity composite (z-score)")
    ax.set_title(f"Convergent Validity: Complexity vs. SAVOIAS Human Ranking\n"
                 f"Spearman rho = {rho:.2f}, 95% CI [{lo:.2f}, {hi:.2f}] (n = {len(merged)})")
    fig.tight_layout()
    savefig(fig, "fig07_savoias_validation.png")
    return res


def anova_power(f, nobs, k, alpha=0.05):
    df1, df2 = k - 1, nobs - k
    crit = stats.f.ppf(1 - alpha, df1, df2)
    return float(stats.ncf.sf(crit, df1, df2, f * f * nobs))


def solve_total_n(f, k, alpha=0.05, power=0.80):
    return optimize.brentq(lambda n: anova_power(f, n, k, alpha) - power, k + 2, 1e6)


def manova_oneway(df, dvs, group):
    d = df[[group] + dvs].dropna()
    cat = pd.Categorical(d[group])
    codes, k = cat.codes, len(cat.categories)
    Y = d[dvs].to_numpy(float)
    N, p = Y.shape
    grand = Y.mean(0)
    H = np.zeros((p, p)); E = np.zeros((p, p))
    for c in range(k):
        Yg = Y[codes == c]
        mg = Yg.mean(0)
        dm = (mg - grand)[:, None]
        H += len(Yg) * (dm @ dm.T)
        R = Yg - mg
        E += R.T @ R
    q, nu = k - 1, N - k
    lam = np.clip(np.sort(np.real(np.linalg.eigvals(np.linalg.solve(E, H))))[::-1], 0, None)
    V = float(np.sum(lam / (1 + lam)))
    s = min(p, q); m = (abs(p - q) - 1) / 2; n = (nu - p - 1) / 2
    Fp = (2 * n + s + 1) / (2 * m + s + 1) * V / (s - V)
    d1p, d2p = s * (2 * m + s + 1), s * (2 * n + s + 1)
    L = float(np.prod(1.0 / (1 + lam)))
    t = math.sqrt((p * p * q * q - 4) / (p * p + q * q - 5)) if (p * p + q * q - 5) > 0 else 1.0
    w = nu - (p - q + 1) / 2
    d1w, d2w = p * q, w * t - (p * q - 2) / 2
    Lt = L ** (1.0 / t)
    Fw = (1 - Lt) / Lt * d2w / d1w
    return {"pillai": {"value": V, "num_df": d1p, "den_df": d2p, "F": Fp, "p": float(stats.f.sf(Fp, d1p, d2p))},
            "wilks": {"value": L, "num_df": d1w, "den_df": d2w, "F": Fw, "p": float(stats.f.sf(Fw, d1w, d2w))},
            "n": N, "k": k}


def manova_text(res, term="genre"):
    lines = ["Multivariate linear model (own implementation: Pillai's trace & Wilks' lambda)",
             f"{term}: N = {res['n']}, k = {res['k']}",
             f"{'':24s}{'Value':>10s}{'Num DF':>10s}{'Den DF':>12s}{'F Value':>10s}{'Pr > F':>12s}"]
    for key, name in (("wilks", "Wilks' lambda"), ("pillai", "Pillai's trace")):
        r = res[key]
        lines.append(f"{name:24s}{r['value']:10.4f}{r['num_df']:10.4f}{r['den_df']:12.4f}{r['F']:10.4f}{r['p']:12.3e}")
    return "\n".join(lines)


def pillai_perm_test(df, dvs, group, n_perm, seed):
    d = df[[group] + dvs].dropna()
    cat = pd.Categorical(d[group]); codes, k = cat.codes, len(cat.categories)
    Y = d[dvs].to_numpy(float); N, p = Y.shape
    Yc = Y - Y.mean(0)
    Tinv = np.linalg.inv(Yc.T @ Yc)

    def pillai(cd):
        H = np.zeros((p, p))
        for c in range(k):
            sel = cd == c
            ng = sel.sum()
            if ng:
                mg = Yc[sel].mean(0)[:, None]
                H += ng * (mg @ mg.T)
        return float(np.trace(H @ Tinv))

    obs = pillai(codes)
    rng = np.random.default_rng(seed)
    perms = np.array([pillai(rng.permutation(codes)) for _ in range(n_perm)])
    return obs, perms, float((perms >= obs).mean())


def welch_anova(groups):
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    mu = np.array([np.mean(g) for g in groups])
    var = np.array([np.var(g, ddof=1) for g in groups])
    w = n / var; W = w.sum(); mw = (w * mu).sum() / W
    A = (w * (mu - mw) ** 2).sum() / (k - 1)
    tmp = ((1 - w / W) ** 2 / (n - 1)).sum()
    B = 1 + 2 * (k - 2) / (k * k - 1) * tmp
    F = A / B
    df1, df2 = k - 1, (k * k - 1) / (3 * tmp)
    allv = np.concatenate(groups); gm = allv.mean()
    ssb = (n * (mu - gm) ** 2).sum(); ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    return {"ddof1": df1, "ddof2": df2, "F": F, "p_unc": float(stats.f.sf(F, df1, df2)), "np2": ssb / (ssb + ssw)}


def games_howell(groups, names):
    k = len(groups); rows = []
    n = [len(g) for g in groups]; mu = [np.mean(g) for g in groups]; var = [np.var(g, ddof=1) for g in groups]
    for i in range(k):
        for j in range(i + 1, k):
            diff = mu[i] - mu[j]
            vi, vj = var[i] / n[i], var[j] / n[j]
            se = math.sqrt(vi + vj)
            T = diff / se
            dof = (vi + vj) ** 2 / (vi ** 2 / (n[i] - 1) + vj ** 2 / (n[j] - 1))
            pval = float(stats.studentized_range.sf(abs(T) * math.sqrt(2), k, dof))
            rows.append({"A": names[i], "B": names[j], "mean(A)": mu[i], "mean(B)": mu[j], "diff": diff,
                         "se": se, "T": T, "df": dof, "p_value": pval})
    return pd.DataFrame(rows)


def tukey_table(df, dim, group="genre"):
    names = sorted(df[group].unique())
    groups = [df.loc[df[group] == g, dim].to_numpy() for g in names]
    res = stats.tukey_hsd(*groups)
    ci = res.confidence_interval(0.95)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rows.append({"group1": names[i], "group2": names[j],
                         "meandiff": groups[j].mean() - groups[i].mean(), "p-adj": res.pvalue[i, j],
                         "lower": -ci.high[i, j], "upper": -ci.low[i, j], "reject": bool(res.pvalue[i, j] < 0.05)})
    return pd.DataFrame(rows)


def lmm_ri_ml(y, X, g):
    y = np.asarray(y, float); X = np.asarray(X, float); g = np.asarray(g)
    N, G = len(y), int(g.max()) + 1
    ng = np.bincount(g, minlength=G).astype(float)
    sX = np.zeros((G, X.shape[1])); np.add.at(sX, g, X)
    sy = np.zeros(G); np.add.at(sy, g, y)
    XtX, Xty, yty = X.T @ X, X.T @ y, float(y @ y)

    def prof(loglam):
        lam = math.exp(loglam)
        c = lam / (1 + ng * lam)
        A = XtX - (sX.T * c) @ sX
        b = Xty - (sX.T * c) @ sy
        beta = np.linalg.solve(A, b)
        q = (yty - float(np.sum(c * sy ** 2))) - 2 * beta @ b + beta @ A @ beta
        s2e = q / N
        ll = -0.5 * (N * math.log(2 * math.pi * s2e) + N + float(np.sum(np.log1p(ng * lam))))
        return ll, beta, s2e, lam

    grid = np.arange(-12, 8.01, 0.5)
    lls = [prof(x)[0] for x in grid]
    i = int(np.argmax(lls))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = optimize.minimize_scalar(lambda x: -prof(x)[0], bounds=(lo, hi), method="bounded",
                                   options={"xatol": 1e-8})
    ll, beta, s2e, lam = prof(res.x if -res.fun >= lls[i] else grid[i])
    return {"llf": ll, "beta": beta, "s2_e": s2e, "s2_u": lam * s2e, "lambda": lam}


def corr_matrix(X):
    return np.corrcoef(np.asarray(X, float), rowvar=False)


def calculate_kmo(R):
    R = np.array(R, float)
    Rinv = np.linalg.inv(R)
    d = np.sqrt(np.outer(np.diag(Rinv), np.diag(Rinv)))
    Pc = -(Rinv / d)
    np.fill_diagonal(R, 0); np.fill_diagonal(Pc, 0)
    R2, P2 = R ** 2, Pc ** 2
    return R2.sum(0) / (R2.sum(0) + P2.sum(0)), float(R2.sum() / (R2.sum() + P2.sum()))


def bartlett_sphericity(R, n):
    p = R.shape[0]
    chi = -(n - 1 - (2 * p + 5) / 6) * math.log(np.linalg.det(R))
    dof = p * (p - 1) / 2
    return float(chi), int(dof), float(stats.chi2.sf(chi, dof))


def _varimax(L, gamma=1.0, max_iter=500, tol=1e-5):
    X = L.copy(); n, k = X.shape
    if k < 2:
        return X, np.eye(k)
    norm = np.sqrt((X ** 2).sum(1))
    X = (X.T / norm).T
    Rm = np.eye(k); d = 0.0
    for _ in range(max_iter):
        old = d
        basis = X @ Rm
        diag = np.diag(np.ones(n) @ (basis ** 2))
        transformed = X.T @ (basis ** 3 - gamma * basis @ diag / n)
        U, S, Vt = np.linalg.svd(transformed)
        Rm = U @ Vt
        d = S.sum()
        if d < old * (1 + tol):
            break
    X = (X @ Rm).T * norm
    return X.T, Rm


def fit_efa(Xdata, n_factors, rotation="varimax"):
    R = corr_matrix(Xdata); p = R.shape[0]
    smc = 1 - 1 / np.diag(np.linalg.inv(R))
    start = np.diag(R) - smc

    def obj(psi):
        Rr = R.copy(); np.fill_diagonal(Rr, 1 - psi)
        vals, vecs = np.linalg.eigh(Rr)
        vals = vals[::-1][:n_factors]; vecs = vecs[:, ::-1][:, :n_factors]
        load = vecs * np.sqrt(np.maximum(vals, 0))
        return float(((Rr - load @ load.T) ** 2).sum())

    res = optimize.minimize(obj, start, method="L-BFGS-B", bounds=[(0.005, 1)] * p, options={"maxiter": 1000})
    Rr = R.copy(); np.fill_diagonal(Rr, 1 - res.x)
    vals, vecs = np.linalg.eigh(Rr)
    vals = vals[::-1][:n_factors]; vecs = vecs[:, ::-1][:, :n_factors]
    L = vecs * np.sqrt(np.maximum(vals, 0))
    if rotation == "varimax" and n_factors > 1:
        L, _ = _varimax(L)
    signs = np.sign(L.sum(0)); signs[signs == 0] = 1
    L = L * signs
    order = np.argsort(-(L ** 2).sum(0))
    L = L[:, order]
    ss = (L ** 2).sum(0)
    return {"loadings": L, "ss_loadings": ss, "prop_var": ss / p, "cum_var": np.cumsum(ss / p),
            "eigenvalues": np.sort(np.linalg.eigvalsh(R))[::-1], "R": R}


import contextlib
import traceback


@contextlib.contextmanager
def guard(name):
    try:
        yield
    except Exception as e:
        log(f"[WARNING] Step '{name}' failed and was SKIPPED: {type(e).__name__}: {e}")
        _fl = [l.strip() for l in traceback.format_exc().splitlines() if l.strip().startswith("File")]
        log("        location: " + (_fl[-1] if _fl else "?"))


def savefig_ax(fig, name):
    fig.tight_layout()
    savefig(fig, name)


def run_analysis(metrics_df, corpus_df=None, savoias_res=None):
    sheets = {}
    rng_seed = CFG["RNG_SEED"]
    m = metrics_df.copy()
    sub = submeasure_cols(m)
    save_table(m, "01_full_metrics.csv", index=False)
    sheets["full_metrics"] = (m, False)
    if corpus_df is not None:
        sheets["corpus_index"] = (corpus_df, False)

    with guard("power analysis"):
        k = len(GENRE_TAGS)
        prow = []
        for label, f in {"small (f=0.10)": 0.10, "medium (f=0.25)": 0.25, "large (f=0.40)": 0.40}.items():
            tot = solve_total_n(f, k)
            prow.append({"effect_size_label": label, "cohens_f": f, "k_groups": k, "alpha": 0.05, "target_power": 0.80,
                         "required_total_n": tot, "required_n_per_group": tot / k})
        power_table = pd.DataFrame(prow)
        save_table(power_table, "00c_power_analysis_sample_size.csv", index=False)
        sheets["power_analysis"] = (power_table, False)
        fig, ax = plt.subplots(figsize=(8, 6))
        nr = np.arange(5, 101, 5)
        for label, f in {"small (f=0.10)": 0.10, "medium (f=0.25)": 0.25, "large (f=0.40)": 0.40}.items():
            ax.plot(nr, [anova_power(f, n * k, k) for n in nr], marker="o", markersize=3, label=label)
        ax.axhline(0.80, color="gray", linestyle="--", linewidth=1, label="Target power = 0.80")
        ax.axvline(CFG["N_GAMES_PER_GENRE"], color="red", linestyle=":", linewidth=1.5,
                   label=f"Target design (n={CFG['N_GAMES_PER_GENRE']}/genre)")
        ax.set_xlabel("Sample size per genre (n)"); ax.set_ylabel("Statistical power (1 - beta)")
        ax.set_title("A Priori Power Curves for the One-Way ANOVA (k = 4 genres, alpha = .05)")
        ax.legend()
        savefig_ax(fig, "fig00_power_analysis.png")

    with guard("official genre composition"):
        if corpus_df is not None and "steam_genres" in corpus_df.columns:
            cd = corpus_df.copy()
            for g in OFFICIAL_STEAM_GENRES:
                cd["official_" + g.replace(" ", "_")] = cd["steam_genres"].apply(lambda s: g.lower() in str(s).lower())
            cols = ["official_" + g.replace(" ", "_") for g in OFFICIAL_STEAM_GENRES]
            comp = cd.groupby("genre")[cols].mean() * 100
            comp.columns = OFFICIAL_STEAM_GENRES
            save_table(comp, "00d_official_genre_composition.csv")
            sheets["official_genre_comp"] = (comp, True)
            fig, ax = plt.subplots(figsize=(12, 6))
            comp.T.plot(kind="bar", ax=ax)
            ax.set_ylabel("% of images whose game carries this official Steam genre")
            ax.set_xlabel("Official Steam Genre (Cunha et al., 2024)")
            ax.set_title("Official Steam Genre Composition of Our Four Genre Groups")
            ax.legend(title="Our genre group", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.xticks(rotation=45, ha="right")
            savefig_ax(fig, "fig00b_official_genre_composition.png")

    with guard("sampling funnel / tag rank / gate summaries"):
        for fname, sheet, idx in (("00g_sampling_funnel.csv", "sampling_funnel", True),
                                  ("00h_retrieval_log.csv", "retrieval_log", False),
                                  ("00e_tag_prominence_accepted_games.csv", "tag_prominence", False)):
            fp = P.res / fname
            if fp.exists():
                sheets[sheet] = (pd.read_csv(fp, sep=";", decimal=",", encoding="utf-8-sig", index_col=0 if idx else None), idx)
        tagdf = sheets.get("tag_prominence", (None,))[0]
        if tagdf is not None and "tag_rank" in tagdf.columns:
            rank_dist = pd.crosstab(tagdf["genre"], tagdf["tag_rank"].astype(int))
            save_table(rank_dist, "00i_tag_rank_distribution.csv")
            sheets["tag_rank_distribution"] = (rank_dist, True)
            fig, ax = plt.subplots(figsize=(8, 5))
            rank_dist.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
            ax.set_xlabel("Genre group"); ax.set_ylabel("Number of games")
            ax.set_title("Rank of the Target Tag Among Each Accepted Game's Most-Voted Tags")
            ax.legend(title="Tag rank"); plt.xticks(rotation=0)
            savefig_ax(fig, "fig11_tag_rank_distribution.png")
        fun = sheets.get("sampling_funnel", (None,))[0]
        if fun is not None:
            sc = [c for c in fun.columns if c not in ("candidates_examined", "steamspy_pool_total")]
            fig, ax = plt.subplots(figsize=(10, 5))
            fun[sc].plot(kind="barh", stacked=True, ax=ax, colormap="tab20")
            ax.set_xlabel("Number of candidate games examined"); ax.set_ylabel("Genre group")
            ax.set_title("Sampling Funnel: Outcome of Each Candidate Game")
            ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            savefig_ax(fig, "fig14_sampling_funnel.png")
        shot_raw = P.res / "00j_shot_gate_log_raw.csv"
        if shot_raw.exists():
            shot_df = pd.read_csv(shot_raw, sep=";", decimal=",", encoding="utf-8-sig")

            def reason(r):
                if r["accept"]:
                    return "accepted"
                pf = r["places_indoor"] < CFG["INDOOR_PROB_THRESHOLD"]
                cf = r["clip_indoor"] < CFG["CLIP_INDOOR_THRESHOLD"]
                return "rejected_by_both" if (pf and cf) else ("rejected_by_places_only" if pf else "rejected_by_clip_only")
            shot_df["gate_result"] = shot_df.apply(reason, axis=1)
            save_table(shot_df, "00j_shot_gate_log.csv", index=False)
            gate_summary = shot_df.groupby(["genre", "gate_result"]).size().unstack(fill_value=0)
            gate_summary["shots_scanned"] = gate_summary.sum(axis=1)
            save_table(gate_summary, "00k_gate_summary.csv")
            luma = shot_df.pivot_table(index="genre", columns="gate_result", values="mean_luma", aggfunc="mean")
            save_table(luma, "00l_gate_mean_luma_by_genre.csv")
            sheets["shot_gate_log"] = (shot_df, False); sheets["gate_summary"] = (gate_summary, True)
            sheets["gate_luma"] = (luma, True)
            log("\nGate summary (genre x decision):\n" + gate_summary.to_string())
            log("\nMean luminance (genre x decision) - if rejected thumbnails are clearly DARKER, the gate may be discarding dark interiors:\n"
                + luma.round(3).to_string())
            _audit_sheets(corpus_df, shot_df)

    with guard("descriptive statistics"):
        desc = m.groupby("genre")[DIMS].agg(["mean", "std", "count"])
        save_table(desc, "02c_descriptive_statistics.csv"); sheets["descriptives"] = (desc, True)
        log("\nDescriptive statistics:\n" + desc.round(3).to_string())

    with guard("independence: ICC, design effect, mixed models"):
        avg_cluster = len(m) / m["game_id"].nunique()
        icc_rows = []
        for dim in DIMS:
            d = m[[dim, "game_id"]].dropna()
            g = pd.Categorical(d["game_id"]).codes
            r0 = lmm_ri_ml(d[dim].to_numpy(), np.ones((len(d), 1)), g)
            icc = r0["s2_u"] / (r0["s2_u"] + r0["s2_e"])
            de = 1 + (avg_cluster - 1) * icc
            icc_rows.append({"dimension": dim, "icc": icc, "between_var": r0["s2_u"], "within_var": r0["s2_e"],
                             "design_effect": de, "raw_n": len(d), "effective_n": len(d) / de})
        icc_table = pd.DataFrame(icc_rows)
        save_table(icc_table, "07_icc_design_effect.csv", index=False); sheets["icc_design_effect"] = (icc_table, False)
        log("\nICC / design effect:\n" + icc_table.round(3).to_string(index=False))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(icc_table["dimension"], icc_table["icc"], color="#4C72B0")
        for i, v in enumerate(icc_table["icc"]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        ax.set_ylabel("Intraclass Correlation (ICC)"); ax.set_title("Within-Game Clustering of Screenshots by Dimension")
        savefig_ax(fig, "fig08_icc_by_dimension.png")

    with guard("mixed-model LRT"):
        lrt_rows = []
        for dim in DIMS:
            d = m[[dim, "genre", "game_id"]].dropna()
            g = pd.Categorical(d["game_id"]).codes
            Xf = pd.get_dummies(d["genre"], drop_first=True, dtype=float)
            Xf.insert(0, "const", 1.0)
            r_null = lmm_ri_ml(d[dim].to_numpy(), np.ones((len(d), 1)), g)
            r_full = lmm_ri_ml(d[dim].to_numpy(), Xf.to_numpy(), g)
            lr = 2 * (r_full["llf"] - r_null["llf"]); dfl = Xf.shape[1] - 1
            lrt_rows.append({"dimension": dim, "llf_null": r_null["llf"], "llf_full": r_full["llf"],
                             "lr_chi2": lr, "df": dfl, "p_value": float(stats.chi2.sf(lr, dfl))})
        mixed = pd.DataFrame(lrt_rows)
        save_table(mixed, "07b_mixedlm_lrt_results.csv", index=False); sheets["mixedlm_results"] = (mixed, False)
        log("\nMixed model (random intercept for game) LRT:\n" + mixed.round(5).to_string(index=False))

    with guard("game-level analysis"):
        gl = m.groupby(["game_id", "genre"])[DIMS].mean().reset_index()
        save_table(gl, "07c_gamelevel_metrics.csv", index=False); sheets["gamelevel_metrics"] = (gl, False)
        gm = manova_oneway(gl, DIMS, "genre")
        (P.res / "07d_gamelevel_manova.txt").write_text(manova_text(gm), encoding="utf-8")
        garows = []
        for dim in DIMS:
            f_, p_ = stats.f_oneway(*[gl.loc[gl.genre == g, dim].to_numpy() for g in sorted(gl.genre.unique())])
            garows.append({"dimension": dim, "F_statistic": f_, "p_value": p_})
        game_anova = pd.DataFrame(garows)
        save_table(game_anova, "07e_gamelevel_anova.csv", index=False); sheets["gamelevel_anova"] = (game_anova, False)
        log("\nGame-level MANOVA:\n" + manova_text(gm) + "\n" + game_anova.round(5).to_string(index=False))

    with guard("image-level: MANOVA, permutation, Levene, ANOVA, Tukey, Welch, Games-Howell"):
        md = m[["genre"] + DIMS].dropna()
        mres = manova_oneway(md, DIMS, "genre")
        (P.res / "03_manova_results.txt").write_text(manova_text(mres), encoding="utf-8")
        manova_tab = pd.DataFrame([{"statistic": n, **mres[k_]} for k_, n in (("pillai", "Pillai's trace (primary)"), ("wilks", "Wilks' lambda"))])
        save_table(manova_tab, "03_manova_omnibus.csv", index=False); sheets["manova_omnibus"] = (manova_tab, False)
        log("\nMANOVA (image level):\n" + manova_text(mres))

    with guard("permutation MANOVA"):
        obs, perms, pperm = pillai_perm_test(md, DIMS, "genre", CFG["N_PERMUTATIONS"], rng_seed)
        p_plus1 = float((1 + (perms >= obs).sum()) / (1 + len(perms)))
        perm_sum = pd.DataFrame([{"observed_pillai": obs, "n_permutations": len(perms), "permutation_p_value": pperm,
                                  "permutation_p_plus1_corrected": p_plus1}])
        save_table(perm_sum, "09_permutation_manova.csv", index=False); sheets["permutation_manova"] = (perm_sum, False)
        log(f"Permutation MANOVA: Pillai = {obs:.4f}, p = {pperm:.5f}; +1-corrected p = {p_plus1:.5f} (n={len(perms)})")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(perms, bins=40, color="#555555", edgecolor="none")
        ax.axvline(obs, color="crimson", linewidth=2, label=f"Observed = {obs:.3f}")
        ax.set_xlabel("Pillai's trace under random genre permutation"); ax.set_ylabel("Frequency")
        ax.set_title(f"Permutation Test for the Genre Effect ({'p < .001' if p_plus1 < .001 else f'p = {p_plus1:.3f}'}, {len(perms):,} permutations)")
        ax.legend(); savefig_ax(fig, "fig09_permutation_manova.png")

    with guard("Levene / ANOVA / Tukey / Welch / Games-Howell"):
        names = sorted(md["genre"].unique())
        lev, anov, welch, gh_frames, tuk_frames = [], [], [], [], []
        for dim in DIMS:
            groups = [md.loc[md.genre == g, dim].to_numpy() for g in names]
            w_, p_ = stats.levene(*groups)
            lev.append({"dimension": dim, "levene_W": w_, "p_value": p_, "variances_equal_at_.05": p_ >= 0.05})
            f_, pf = stats.f_oneway(*groups)
            anov.append({"dimension": dim, "F_statistic": f_, "p_value": pf})
            wr = welch_anova(groups); welch.append({"dimension": dim, "Source": "genre", **wr})
            gh = games_howell(groups, names); gh.insert(0, "dimension", dim); gh_frames.append(gh)
            tk = tukey_table(md, dim); tk.insert(0, "dimension", dim); tuk_frames.append(tk)
            save_table(tk.drop(columns="dimension"), f"04_tukey_{dim.lower()}.csv", index=False)
        lev_t, anova_t, welch_t = pd.DataFrame(lev), pd.DataFrame(anov), pd.DataFrame(welch)
        gh_t, tuk_t = pd.concat(gh_frames, ignore_index=True), pd.concat(tuk_frames, ignore_index=True)
        for df_, fn, sh in ((lev_t, "08_levene_homogeneity.csv", "levene_homogeneity"),
                            (anova_t, "04_anova_summary.csv", "anova_univariate"),
                            (welch_t, "08b_welch_anova.csv", "welch_anova"),
                            (gh_t, "08c_games_howell_posthoc.csv", "games_howell"),
                            (tuk_t, "04_tukey_all_dimensions.csv", "tukey_hsd")):
            save_table(df_, fn, index=False); sheets[sh] = (df_, False)
        log("\nLevene:\n" + lev_t.round(5).to_string(index=False))
        log("\nANOVA:\n" + anova_t.round(5).to_string(index=False))

    with guard("EFA (composites), H3 bootstrap, 13-sub-measure EFA"):
        efa_X = m[DIMS].dropna().to_numpy(float)
        R4 = corr_matrix(efa_X)
        _, kmo_model = calculate_kmo(R4)
        chi, dfb, pb = bartlett_sphericity(R4, len(efa_X))
        fa = fit_efa(efa_X, 2)
        loadings = pd.DataFrame(fa["loadings"], index=DIMS, columns=["Factor1", "Factor2"])
        save_table(loadings, "05_efa_loadings_composite.csv"); sheets["efa_composite"] = (loadings, True)
        (P.res / "05_efa_summary.txt").write_text(
            f"KMO (overall): {kmo_model}\nBartlett chi-square: {chi}, df = {dfb}, p = {pb}\n"
            f"Variance explained (SS loadings, prop. var, cum. var):\n"
            f"({fa['ss_loadings']}, {fa['prop_var']}, {fa['cum_var']})\n", encoding="utf-8")
        log(f"\nEFA (composites): KMO = {kmo_model:.3f}; Bartlett chi2({dfb}) = {chi:.2f}, p = {pb:.3g}")
        log(loadings.round(3).to_string())

    with guard("H3 bootstrap"):
        cd_ = m[DIMS].dropna().reset_index(drop=True)
        r_cm = np.corrcoef(cd_["Complexity"], cd_["Mystery"])[0, 1]
        r_cl = np.corrcoef(cd_["Coherence"], cd_["Legibility"])[0, 1]
        rng = np.random.default_rng(rng_seed)
        A = cd_[["Complexity", "Mystery", "Coherence", "Legibility"]].to_numpy()
        diffs = np.empty(CFG["N_BOOTSTRAP"])
        for i in range(CFG["N_BOOTSTRAP"]):
            s = A[rng.integers(0, len(A), len(A))]
            diffs[i] = np.corrcoef(s[:, 0], s[:, 1])[0, 1] - np.corrcoef(s[:, 2], s[:, 3])[0, 1]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        bp = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
        h3 = pd.DataFrame([{"r_complexity_mystery": r_cm, "r_coherence_legibility": r_cl, "observed_difference": r_cm - r_cl,
                            "ci_95_lower": lo, "ci_95_upper": hi, "bootstrap_p_value": bp,
                            "bootstrap_p_smallest_reportable": 2.0 / (CFG["N_BOOTSTRAP"] + 1), "n_bootstrap": CFG["N_BOOTSTRAP"]}])
        save_table(h3, "10_bootstrap_h3_test.csv", index=False); sheets["bootstrap_h3"] = (h3, False)
        log(f"H3 bootstrap: r(Cx,My) = {r_cm:.3f}, r(Co,Lg) = {r_cl:.3f}, difference = {r_cm - r_cl:.3f}, CI [{lo:.3f}, {hi:.3f}], p = {bp:.5f}")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(diffs, bins=50, color="#4C72B0", edgecolor="white")
        ax.axvline(r_cm - r_cl, color="crimson", linewidth=2, label=f"Observed diff = {r_cm - r_cl:.3f}")
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Bootstrap difference: r(Complexity, Mystery) - r(Coherence, Legibility)"); ax.set_ylabel("Frequency")
        ax.set_title(f"Bootstrap Test of H3's Predicted Axis Pairing ({'p < .001' if bp < .001 else f'p = {bp:.3f}'})"); ax.legend()
        savefig_ax(fig, "fig10_bootstrap_h3.png")

    with guard("13-sub-measure EFA"):
        SUB = sub["Complexity"] + sub["Coherence"] + sub["Legibility"] + sub["Mystery"]
        const = [c for c in SUB if m[c].nunique() <= 1]
        if const:
            log(f"[WARNING] Constant (zero-variance) sub-measure(s) removed from the EFA: {const}")
        SUB = [c for c in SUB if c not in const]
        sd = m[SUB].dropna().to_numpy(float)
        msa, kmo_sub = calculate_kmo(corr_matrix(sd))
        per_item = pd.DataFrame({"submeasure": SUB, "item_MSA": msa})
        save_table(per_item, "11b_efa_submeasure_kmo.csv", index=False); sheets["efa_submeasure_kmo"] = (per_item, False)
        fa4 = fit_efa(sd, 4)
        sub_load = pd.DataFrame(fa4["loadings"], index=SUB, columns=["Factor1", "Factor2", "Factor3", "Factor4"])
        save_table(sub_load, "11_efa_submeasure_loadings.csv"); sheets["efa_submeasure"] = (sub_load, True)
        log(f"\n13-sub-measure EFA: overall KMO = {kmo_sub:.3f}")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(1, len(fa4["eigenvalues"]) + 1), fa4["eigenvalues"], marker="o")
        ax.axhline(1.0, color="gray", linestyle="--", label="Eigenvalue = 1")
        ax.set_xlabel("Factor number"); ax.set_ylabel("Eigenvalue")
        ax.set_title("Scree Plot: 13 Sub-Measures Underlying the Four Kaplan Dimensions"); ax.legend()
        savefig_ax(fig, "fig12_submeasure_scree.png")
        fig, ax = plt.subplots(figsize=(7, 8))
        im = ax.imshow(sub_load.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels(sub_load.columns)
        ax.set_yticks(range(len(sub_load))); ax.set_yticklabels(sub_load.index, fontsize=8)
        plt.colorbar(im, ax=ax, label="Loading"); ax.set_title("Sub-Measure Loadings (4-Factor Solution)")
        savefig_ax(fig, "fig13_submeasure_loadings.png")

    with guard("clustering"):
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        cdata = m[DIMS].dropna()
        scaled = StandardScaler().fit_transform(cdata)
        km = KMeans(n_clusters=4, random_state=rng_seed, n_init=10).fit(scaled)
        m.loc[cdata.index, "kmeans_cluster"] = km.labels_
        linked = linkage(scaled, method="ward")
        m.loc[cdata.index, "hierarchical_cluster"] = fcluster(linked, t=4, criterion="maxclust")
        save_table(m, "06_metrics_with_clusters.csv", index=False)
        prof = m.groupby("kmeans_cluster")[DIMS].mean()
        save_table(prof, "06_cluster_profiles.csv"); sheets["cluster_profiles"] = (prof, True)
        ct = pd.crosstab(m["genre"], m["kmeans_cluster"], normalize="index") * 100
        save_table(ct, "06b_cluster_by_genre.csv"); sheets["cluster_by_genre"] = (ct, True)
        log("\nCluster profiles:\n" + prof.round(3).to_string())
        pca = PCA(n_components=2); xy = pca.fit_transform(scaled)
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=km.labels_, cmap="tab10", alpha=0.75, s=40)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
        ax.set_title("K-Means Clusters in PCA-Reduced Space", fontsize=14)
        ax.add_artist(ax.legend(*sc.legend_elements(), title="Cluster"))
        savefig_ax(fig, "fig05_pca_clusters.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(linked, ax=ax, no_labels=True, color_threshold=0.7 * max(linked[:, 2]))
        ax.set_title("Hierarchical Clustering of Interior Spatial Profiles", fontsize=14)
        ax.set_xlabel("Images"); ax.set_ylabel("Ward Distance")
        savefig_ax(fig, "fig04_dendrogram.png")

    with guard("strict-threshold sensitivity analysis"):
        if "tag_rank" in m.columns and m["tag_rank"].notna().any():
            sd_ = m[m["tag_rank"] <= CFG["STRICT_TAG_TOP_K"]].copy()
            cnt = sd_.groupby("genre").agg(images=("image_path", "count"), games=("game_id", "nunique"))
            save_table(cnt, "12c_strict_sensitivity_counts.csv"); sheets["strict_counts"] = (cnt, True)
            log(f"\nStrict threshold (target tag within the top {CFG['STRICT_TAG_TOP_K']}):\n" + cnt.to_string())
            try:
                sm = manova_oneway(sd_, DIMS, "genre")
                st_t = pd.DataFrame([{"statistic": "Pillai's trace", **sm["pillai"]}, {"statistic": "Wilks' lambda", **sm["wilks"]}])
                save_table(st_t, "12_strict_sensitivity_manova.csv", index=False); sheets["strict_manova"] = (st_t, False)
                nm = sorted(sd_["genre"].unique()); rows = []
                for dim in DIMS:
                    w = welch_anova([sd_.loc[sd_.genre == g, dim].to_numpy() for g in nm])
                    rows.append({"dimension": dim, **w})
                sw = pd.DataFrame(rows)
                save_table(sw, "12b_strict_sensitivity_welch.csv", index=False); sheets["strict_welch"] = (sw, False)
                log(st_t.round(4).to_string(index=False)); log(sw.round(4).to_string(index=False))
            except Exception as e:
                log("[WARNING] Strict-threshold sensitivity analysis could not be run:", e)

    with guard("main figures"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, dim in zip(axes.flat, DIMS):
            m.boxplot(column=dim, by="genre", ax=ax, grid=False)
            ax.set_title(f"{dim} by Genre"); ax.set_xlabel("Genre"); ax.set_ylabel(f"{dim} (z-score composite)")
        plt.suptitle("")
        fig.suptitle("Kaplan Preference Matrix Dimensions by Game Genre", fontsize=14, y=1.02)
        savefig_ax(fig, "fig01_boxplots_by_genre.png")
        gmeans = m.groupby("genre")[DIMS].mean()
        ang = [n / len(DIMS) * 2 * math.pi for n in range(len(DIMS))]; ang += ang[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for gname in gmeans.index:
            vals = gmeans.loc[gname].tolist(); vals += vals[:1]
            ax.plot(ang, vals, linewidth=2, label=gname); ax.fill(ang, vals, alpha=0.1)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(DIMS)
        ax.set_title("Genre Profiles Across the Four Kaplan Dimensions", fontsize=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
        savefig(fig, "fig02_radar_genre_profiles.png")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(loadings.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(2)); ax.set_xticklabels(loadings.columns)
        ax.set_yticks(range(4)); ax.set_yticklabels(loadings.index)
        for i in range(4):
            for j in range(2):
                ax.text(j, i, f"{loadings.values[i, j]:.2f}", ha="center", va="center")
        ax.set_title("Exploratory Factor Analysis: Factor Loadings (Composite Dimensions)", fontsize=12)
        plt.colorbar(im, ax=ax, label="Loading")
        savefig_ax(fig, "fig03_efa_loadings_heatmap.png")
        corr = m[DIMS].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(DIMS, rotation=45, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels(DIMS)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
        ax.set_title("Correlation Among the Four Composite Dimensions", fontsize=13)
        plt.colorbar(im, ax=ax)
        savefig_ax(fig, "fig06_dimension_correlation_heatmap.png")

    if savoias_res is not None:
        sheets["savoias_validation"] = (savoias_res, False)
    write_excel(sheets)
    zip_images(m)
    return m


def _contact_sheet(paths, titles, name, ncols=8, thumb_w=260):
    import cv2
    n = len(paths)
    if n == 0:
        return
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 1.8))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, pth, ttl in zip(axes, paths, titles):
        img = cv2.imread(str(pth))
        if img is None:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (thumb_w, max(1, int(thumb_w * h / w))))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.set_title(ttl, fontsize=5)
    savefig_ax(fig, name)


def _audit_sheets(corpus_df, shot_df):
    if corpus_df is not None and len(corpus_df):
        a = corpus_df.sample(min(CFG["AUDIT_N"], len(corpus_df)), random_state=CFG["RNG_SEED"])
        _contact_sheet([absp(p) for p in a["image_path"]],
                       [f"{r.genre} | clip_in={r.clip_indoor:.2f}" for r in a.itertuples()],
                       "fig15_audit_accepted_sample.png")
    rej = shot_df[~shot_df["accept"]]
    if len(rej):
        r_ = rej.sample(min(CFG["AUDIT_N"], len(rej)), random_state=CFG["RNG_SEED"])
        _contact_sheet([P.reject / f"{int(r.game_id)}_{int(r.shot_index)}.jpg" for r in r_.itertuples()],
                       [f"{r.gate_result.replace('rejected_by_', '')} in={r.clip_indoor:.2f} out={r.clip_outdoor:.2f} "
                        f"ns={r.clip_nonscene:.2f}" for r in r_.itertuples()],
                       "fig16_audit_rejected_sample.png")


def write_excel(sheets):
    path = P.root / "kaplan_game_interiors_all_results_v3.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, (df, idx) in sheets.items():
            try:
                df.to_excel(w, sheet_name=name[:31], index=idx)
            except Exception as e:
                log(f"[WARNING] Could not write Excel sheet ({name}): {e}")
    log("\nExcel workbook saved:", path)


def zip_images(m):
    zpath = P.root / "analyzed_images.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for rp in m["image_path"].unique():
            fp = absp(rp)
            if fp.exists():
                z.write(fp, rp)
    log("Analyzed images zipped:", zpath)


def run_gate_test(folder):
    from PIL import Image
    folder = Path(folder)
    load_gate_models()
    rows = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            g = scene_gate_pil(Image.open(f).convert("RGB"))
            rows.append({"file": f.name, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()}})
    df = pd.DataFrame(rows)
    log(df.to_string(index=False))
    log("\nExpected: accept = False for collages, text screens, and open-world/space images.")
    save_table(df, "00m_gate_test.csv", index=False)
    return df


def run_crosscheck(m):
    rows = []

    def add(q, own, lib):
        own, lib = np.asarray(own, float), np.asarray(lib, float)
        rows.append({"quantity": q,
                     "own_value": float(own.ravel()[0]) if own.size == 1 else np.nan,
                     "library_value": float(lib.ravel()[0]) if lib.size == 1 else np.nan,
                     "max_abs_diff": float(np.nanmax(np.abs(own - lib)))})

    md = m[["genre"] + DIMS].dropna()
    with guard("cross-check: statsmodels (MANOVA, mixed model)"):
        from statsmodels.multivariate.manova import MANOVA
        import statsmodels.formula.api as smf
        st = MANOVA.from_formula("Coherence + Complexity + Legibility + Mystery ~ genre", data=md).mv_test().results["genre"]["stat"]
        mine = manova_oneway(md, DIMS, "genre")
        for key, nm in (("pillai", "Pillai's trace"), ("wilks", "Wilks' lambda")):
            add(f"MANOVA {nm} value", mine[key]["value"], st.loc[nm, "Value"])
            add(f"MANOVA {nm} F", mine[key]["F"], st.loc[nm, "F Value"])
            add(f"MANOVA {nm} den.df", mine[key]["den_df"], st.loc[nm, "Den DF"])
        for dim in DIMS:
            d = m[[dim, "genre", "game_id"]].dropna()
            g = pd.Categorical(d["game_id"]).codes
            r0 = lmm_ri_ml(d[dim].to_numpy(), np.ones((len(d), 1)), g)
            Xf = pd.get_dummies(d["genre"], drop_first=True, dtype=float); Xf.insert(0, "c", 1.0)
            r1 = lmm_ri_ml(d[dim].to_numpy(), Xf.to_numpy(), g)
            n0 = smf.mixedlm(f"{dim} ~ 1", d, groups=d["game_id"]).fit(reml=False)
            n1 = smf.mixedlm(f"{dim} ~ C(genre)", d, groups=d["game_id"]).fit(reml=False)
            add(f"ICC [{dim}]", r0["s2_u"] / (r0["s2_u"] + r0["s2_e"]),
                float(n0.cov_re.iloc[0, 0] / (n0.cov_re.iloc[0, 0] + n0.scale)))
            add(f"LRT chi2 [{dim}]", 2 * (r1["llf"] - r0["llf"]), 2 * (n1.llf - n0.llf))

    with guard("cross-check: pingouin (Welch, Games-Howell)"):
        import pingouin as pg
        names = sorted(md["genre"].unique())
        for dim in DIMS:
            groups = [md.loc[md.genre == k, dim].to_numpy() for k in names]
            w = welch_anova(groups)
            wl = pg.welch_anova(data=md, dv=dim, between="genre")
            add(f"Welch F [{dim}]", w["F"], wl["F"].iloc[0])
            add(f"Welch ddof2 [{dim}]", w["ddof2"], wl["ddof2"].iloc[0])
            gh = games_howell(groups, names)
            gl = pg.pairwise_gameshowell(data=md, dv=dim, between="genre")
            pcol = "pval" if "pval" in gl.columns else next(c for c in gl.columns if c.lower().startswith("p"))
            j = gh.merge(gl[["A", "B", pcol]], on=["A", "B"])
            add(f"Games-Howell p [{dim}] (all pairs)", j["p_value"].to_numpy(), j[pcol].to_numpy())

    with guard("cross-check: factor_analyzer (EFA, KMO, Bartlett)"):
        from factor_analyzer import FactorAnalyzer
        from factor_analyzer.factor_analyzer import calculate_kmo as fa_kmo, calculate_bartlett_sphericity as fa_bart
        Xd = m[DIMS].dropna()
        own = fit_efa(Xd.to_numpy(float), 2)
        fa = FactorAnalyzer(n_factors=2, rotation="varimax"); fa.fit(Xd)
        add("EFA composite |loadings|", np.abs(own["loadings"]), np.abs(fa.loadings_))
        add("KMO overall (composite)", calculate_kmo(corr_matrix(Xd))[1], fa_kmo(Xd)[1])
        chi, _, _ = bartlett_sphericity(corr_matrix(Xd), len(Xd))
        add("Bartlett chi2 (composite)", chi, fa_bart(Xd)[0])
        subc = submeasure_cols(m)
        SUB = [c for c in subc["Complexity"] + subc["Coherence"] + subc["Legibility"] + subc["Mystery"] if m[c].nunique() > 1]
        Xs = m[SUB].dropna()
        own4 = fit_efa(Xs.to_numpy(float), 4)
        fa4 = FactorAnalyzer(n_factors=4, rotation="varimax"); fa4.fit(Xs)
        add("EFA 13-item |loadings|", np.abs(own4["loadings"]), np.abs(fa4.loadings_))
        add("13-item MSA (per item)", calculate_kmo(corr_matrix(Xs))[0], fa_kmo(Xs)[0])

    out = pd.DataFrame(rows)
    if len(out):
        save_table(out, "13_crosscheck_vs_libraries.csv", index=False)
        log("\nLibrary cross-check (own = this file, library = statsmodels/pingouin/factor_analyzer):")
        log(out.to_string(index=False))
        bad = out[out["max_abs_diff"] > 1e-3]
        log("\nAll differences < 1e-3." if bad.empty else "\nDifferences larger than 1e-3:\n" + bad.to_string(index=False))
    else:
        log("None of the libraries (statsmodels / pingouin / factor_analyzer) is installed; cross-check skipped.")
    return out

def main(argv=None):
    ap = argparse.ArgumentParser(description="Kaplan Preference Matrix - game interiors pipeline")
    ap.add_argument("--stage", default="all", choices=["all", "sample", "metrics", "savoias", "analyze", "crosscheck"])
    ap.add_argument("--gate-test", metavar="FOLDER", help="run the images in a folder through the scene gate and print the decisions")
    ap.add_argument("--cross-check", action="store_true", help="compare with statsmodels/pingouin/factor_analyzer after the analysis")
    ap.add_argument("--quick", action="store_true", help="small test run (writes to the quick_test folder)")
    ap.add_argument("--force-sample", action="store_true", help="redo sampling from scratch")
    ap.add_argument("--skip-savoias", action="store_true")
    ap.add_argument("--n-games", type=int)
    ap.add_argument("--max-candidates", type=int)
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    root = Path(__file__).resolve().parent
    if args.quick:
        CFG.update(N_GAMES_PER_GENRE=4, MAX_CANDIDATES_PER_GENRE=60, N_PERMUTATIONS=300, N_BOOTSTRAP=500,
                   HAND_CODE_N=5, AUDIT_N=16)
        root = root / "quick_test"
    if args.n_games:
        CFG["N_GAMES_PER_GENRE"] = args.n_games
    if args.max_candidates:
        CFG["MAX_CANDIDATES_PER_GENRE"] = args.max_candidates
    init_paths(root)
    log(f"Working folder: {P.root}")
    t0 = time.time()
    if args.gate_test:
        run_gate_test(args.gate_test)
        return

    corpus_df = None
    if args.stage in ("all", "sample"):
        corpus_df = run_sampling(force=args.force_sample)
    if args.stage == "sample":
        return
    if corpus_df is None:
        corpus_df = load_corpus()

    metrics_df = None
    if args.stage in ("all", "metrics", "analyze", "savoias", "crosscheck"):
        if args.stage in ("all", "metrics"):
            metrics_df = run_metrics(corpus_df)
        else:
            metrics_df = add_composites(pd.read_csv(P.cache / "metrics_partial.csv"))
            metrics_df = metrics_df[metrics_df["image_path"].isin(set(corpus_df["image_path"]))].reset_index(drop=True)
    if args.stage == "metrics":
        return
    if args.stage == "crosscheck":
        run_crosscheck(metrics_df)
        return

    sub = submeasure_cols(metrics_df)
    sav = None
    if args.stage in ("all", "savoias") and not args.skip_savoias:
        sav = run_savoias(sub)
    elif (P.res / "02_savoias_validation.csv").exists():
        sav = pd.read_csv(P.res / "02_savoias_validation.csv", sep=";", decimal=",", encoding="utf-8-sig")
    if args.stage == "savoias":
        return

    res_m = run_analysis(metrics_df, corpus_df, sav)
    if args.cross_check:
        run_crosscheck(res_m)
    log(f"\nDone ({(time.time() - t0) / 60:.1f} min). Outputs: {P.root}")


if __name__ == "__main__":
    main()
