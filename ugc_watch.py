#!/usr/bin/env python3
"""
ugc_watch.py - watches newly published Roblox UGC for remakes / recolors /
copies of YOUR items and reports scored matches to a Discord webhook.

How it decides (cheap checks first, expensive ones last):
  1. Source   - Roblox's own catalog search API ("recently published" sort) for
                the accessory types you sell, plus keyword sweeps built from your
                item names so older uploads are covered too.
  2. Skip     - anything already checked, your own group, whitelisted creators,
                items older than your matching item.
  3. Look     - downloads the candidate's thumbnail and compares it with every
                one of your items: silhouette overlap + fur/shading structure,
                both normal and mirrored, ignoring colour (so recolours match).
  4. Read     - name similarity (colour words removed) and description
                similarity, including your credit lines being pasted.
  5. AI check - optional. If an API key is set, the two thumbnails are shown to a
                vision model (Claude or Gemini) for a second opinion before alerting.
  6. Verdict  - a name match alone is never called a copy.

Usage:
  python ugc_watch.py              run forever
  python ugc_watch.py --once       one cycle then exit
  python ugc_watch.py --check ID   score one catalog item against yours, no alert
  python ugc_watch.py --test       send a test message to the webhook
"""
import argparse, base64, calendar, difflib, gzip, hashlib, io, json, os, pathlib, re, sqlite3, struct, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
CATALOG = "https://catalog.roblox.com/v1/search/items/details"
THUMBS = "https://thumbnails.roblox.com/v1/assets"
# assetType id -> catalog Subcategory id. 46/47 confirmed live; the rest are
# checked at runtime (wrong ones fall back to the general accessories feed).
SUBCAT = {46: 25, 47: 26}
COLOURS = set("""black white grey gray red blue pink purple violet green yellow orange brown
light dark pastel neon gold golden silver cyan teal lavender beige cream blonde ginger
ombre rainbow and""".split())
STOP = set("the a an of for with to in on my your our is it its this that very".split())

DEFAULTS = {
    "discord_webhook_url": "",
    "my_creator_id": 0,              # your group id (or user id)
    "my_creator_type": "Group",      # "Group" or "User"
    "ignore_creator_ids": [],        # friends / alt groups never to flag
    "extra_asset_types": [],         # extra assetType ids to watch besides the ones you sell
    "poll_minutes": 5,
    "feed_pages": 6,                 # pages (30 items each) per type per poll
    "sweep_hours": 12,               # how often to redo the keyword sweep
    "sweep_pages": 15,               # pages per keyword in a sweep
    "refresh_my_items_hours": 6,
    "copy_threshold": 0.80,          # visual score: same model (copy / recolour)
    "remake_threshold": 0.63,        # visual score: similar shape on looks alone (remake)
    "combo_threshold": 0.50,         # visual score that counts when name/description match too
    "ai_provider": "auto",           # "auto" (every model whose key is set), "anthropic", "gemini", "off"
    "anthropic_api_key": "",         # paid, a small fraction of a cent per check
    "gemini_api_key": "",            # free tier
    "anthropic_workspace_id": "",    # wrkspc_... ; only needed when the Claude key is not tied to one workspace
    "anthropic_model": "claude-haiku-5-5",
    "gemini_model": "gemini-3.5-flash",
    "ai_min_visual": 0.45,           # also ask the AI about near-misses above this visual score
    "ai_max_calls_per_day": 300,
    "text_alerts": True,             # also report listings that copy your text only
    "max_alerts_per_cycle": 15,
    "status_updates": "every_run",   # "every_run", "alerts_only" or "off"
    "max_run_minutes": 12,           # a run stops starting new sweep keywords after this long
}

sess = requests.Session()
sess.headers["User-Agent"] = "Mozilla/5.0 (ugc-watch; personal IP monitoring)"
AI_DOWN = set()
STATS = {"checked": 0, "alerts": 0, "ai_ok": 0, "ai_failed": 0, "waited": 0.0, "note": "", "mesh": 0, "mesh_fail": 0}


QUIET = {"on": False}                            # public scanner runs: counts only, no item or creator names


def log(*a):
    if QUIET["on"] and a and str(a[0]).startswith("  "):
        return
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def http_json(url, params=None, tries=6):
    delay = 2.0
    for _ in range(tries):
        try:
            r = sess.get(url, params=params, timeout=25)
        except requests.RequestException as e:
            log("  network error:", e)
            time.sleep(delay); delay = min(delay * 2, 60); continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            hdr = r.headers.get("Retry-After") or r.headers.get("x-ratelimit-reset") or ""
            wait = min(float(hdr) + 0.5, 90) if hdr.replace(".", "", 1).isdigit() else delay   # wait exactly as long as Roblox asks
            STATS["waited"] += wait
            time.sleep(wait); delay = min(delay * 2, 60); continue
        log(f"  HTTP {r.status_code} from {url}: {r.text[:120]}")
        return None
    return None


# ----------------------------------------------------------------- images ---
N = 64


def _mask_rgb(im):
    a = np.asarray(im.convert("RGBA"), dtype=np.float32)
    rgb, alpha = a[..., :3], a[..., 3]
    if alpha.min() < 200:                       # real transparency
        return alpha > 60, rgb
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(edge, axis=0)                # flat background: key it out
    return np.abs(rgb - bg).sum(-1) > 40, rgb


def features(img_bytes):
    """Colour-independent shape/structure descriptor of a thumbnail (or None)."""
    im = Image.open(io.BytesIO(img_bytes))
    mask, rgb = _mask_rgb(im)
    if mask.sum() < 200:
        return None
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    side = max(h, w)
    grey = rgb[y0:y1, x0:x1].mean(-1)
    m = mask[y0:y1, x0:x1]
    mu, sd = grey[m].mean(), grey[m].std() + 1e-6
    g = np.where(m, (grey - mu) / sd, 0.0)      # brightness/contrast normalised
    G = np.zeros((side, side), np.float32); M = np.zeros((side, side), np.float32)
    oy, ox = (side - h) // 2, (side - w) // 2
    G[oy:oy + h, ox:ox + w] = g; M[oy:oy + h, ox:ox + w] = m
    G = np.asarray(Image.fromarray(G).resize((N, N), Image.BILINEAR))
    M = np.asarray(Image.fromarray(M).resize((N, N), Image.BILINEAR)) > 0.5
    return {"mask": M, "grey": G * M, "aspect": w / h,
            "colour": [float(c) for c in rgb[mask].mean(0)]}


def _hog(g, m):
    """Unsigned gradient-orientation histogram per 8x8 cell (ignores light/dark flips)."""
    gy, gx = np.gradient(g)
    inner = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    mag = np.hypot(gx, gy) * inner              # interior detail only, not the outline
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * 9
    b = np.minimum(ang.astype(int), 8)
    out = np.zeros((8, 8, 9), np.float32)
    cy, cx = np.mgrid[0:N, 0:N] // 8
    np.add.at(out, (cy, cx, b), mag)
    out /= np.linalg.norm(out, axis=2, keepdims=True) + 1e-6
    v = out.ravel()
    return v / (np.linalg.norm(v) + 1e-6)


def _sdf_like(m):
    """Blurred mask: tolerant silhouette comparison."""
    f = m.astype(np.float32)
    for _ in range(3):
        f = (f + np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 5
    return f


def finish(feat):
    feat["hog"] = _hog(feat["grey"], feat["mask"])
    feat["hog_m"] = _hog(feat["grey"][:, ::-1], feat["mask"][:, ::-1])
    feat["soft"] = _sdf_like(feat["mask"])
    return feat


def _variants(a):
    """The candidate in 8 orientations (4 quarter-turns, each also mirrored), built once and cached."""
    if "var" not in a:
        a["var"] = []
        for r in range(4):
            for flip in (False, True):
                g, m = np.rot90(a["grey"], r), np.rot90(a["mask"], r)
                if flip:
                    g, m = g[:, ::-1], m[:, ::-1]
                g, m = np.ascontiguousarray(g), np.ascontiguousarray(m)
                a["var"].append((_sdf_like(m), _hog(g, m), a["aspect"] if r % 2 == 0 else 1 / a["aspect"],
                                 1.0 if r == 0 else 0.92, flip or r > 0))
    return a["var"]


def visual_score(a, b):
    """0..1. ~1 = same model (colour ignored). The first item may be mirrored or turned in quarter steps."""
    best = (0.0, 0.0, 0.0, False)
    for soft, hog, aspect, weight, changed in _variants(a):
        shape = float(np.minimum(soft, b["soft"]).sum() / (np.maximum(soft, b["soft"]).sum() + 1e-6))
        detail = float(np.dot(hog, b["hog"]))
        shape *= 0.5 + 0.5 * min(aspect, b["aspect"]) / max(aspect, b["aspect"])
        # shape gates detail: matching fur texture on a different silhouette is not a copy
        score = weight * shape * (0.55 + 0.45 * max(detail, 0.0))
        if score > best[0]:
            best = (score, shape, detail, changed)
    return best


# ------------------------------------------------------------------- text ---
def tokens(s, drop_colours=True):
    t = re.findall(r"[a-z0-9]+", (s or "").lower())
    return [x for x in t if x not in STOP and not (drop_colours and x in COLOURS)]


def name_score(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    jac = len(set(ta) & set(tb)) / len(set(ta) | set(tb))
    return max(jac, difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio())


def desc_score(mine, theirs):
    """How much of MY description shows up inside THEIRS (handles pasted credit lines)."""
    a, b = " ".join(tokens(mine, False)), " ".join(tokens(theirs, False))
    if len(a) < 20 or len(b) < 20:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    covered = sum(blk.size for blk in sm.get_matching_blocks() if blk.size >= 12)
    return min(1.0, covered / len(a))


def handles(s):
    return set(re.findall(r"@\w{3,}", s or ""))


# --------------------------------------------------------------------- db ---
def db_open():
    db = sqlite3.connect(HERE / "ugc_watch.db")
    db.executescript("""
      create table if not exists seen(id integer primary key, at real);
      create table if not exists mine(id integer primary key, name text, descr text,
                                      created text, atype int, thumb text, feat blob);
      create table if not exists kv(k text primary key, v text);
      create table if not exists later(id integer primary key);
      create table if not exists mesh3d(id integer primary key, at real, name text, creator text, mine_id int,
        mine_name text, mesh_id int, tex_id int, my_mesh_id int, my_tex_id int, score real, render real,
        texsim real, info text);""")
    return db


def kv_get(db, k, d=None):
    r = db.execute("select v from kv where k=?", (k,)).fetchone()
    return json.loads(r[0]) if r else d


def kv_set(db, k, v):
    db.execute("insert or replace into kv values(?,?)", (k, json.dumps(v))); db.commit()


def pack(f):
    buf = io.BytesIO()
    np.savez_compressed(buf, mask=f["mask"], grey=f["grey"], aspect=f["aspect"], colour=f["colour"])
    return buf.getvalue()


def unpack(b):
    z = np.load(io.BytesIO(b))
    return finish({"mask": z["mask"], "grey": z["grey"], "aspect": float(z["aspect"]),
                   "colour": z["colour"].tolist()})


# ----------------------------------------------------------------- roblox ---
def search(pages, **params):
    """Yield catalog items page by page (a list per page)."""
    cursor = None
    for _ in range(pages):
        p = {"Category": 11, "Limit": 30, "SortType": 3, **params}
        if cursor:
            p["Cursor"] = cursor
        j = http_json(CATALOG, p)
        time.sleep(1.2)                          # stay well under the rate limit
        if not j or not j.get("data"):
            return
        yield j["data"]
        cursor = j.get("nextPageCursor")
        if not cursor:
            return


def thumb_urls(ids):
    out = {}
    ids = list(ids)
    for i in range(0, len(ids), 100):
        j = http_json(THUMBS, {"assetIds": ",".join(map(str, ids[i:i + 100])),
                               "size": "420x420", "format": "Png", "isCircular": "false"})
        for d in (j or {}).get("data", []):
            out[d["targetId"]] = d.get("imageUrl") if d.get("state") == "Completed" else d.get("state")
    return out


def fetch_feat(url):
    try:
        r = sess.get(url, timeout=25)
        if r.status_code != 200:
            return None, None
        f = features(r.content)
        return (finish(f) if f else None), r.content
    except Exception as e:
        log("  thumbnail failed:", e)
        return None, None


def refresh_mine(db, cfg):
    log("refreshing your items...")
    items = []
    ids = []                                     # whole store, off-sale items included, 120 per request
    for page in search_wide(CATALOG.replace("/details", ""), 40, Category=1, SortType=3,
                            CreatorTargetId=cfg["my_creator_id"], CreatorType=cfg["my_creator_type"],
                            IncludeNotForSale="true"):
        ids += [d["id"] for d in page if d.get("itemType") == "Asset"]
    listed = len(ids)
    items = [d for d in details_batch(ids) if d.get("assetType") in ACCESSORY
             and "DO NOT ONSALE" not in (d.get("name") or "").upper()]
    before = db.execute("select count(*) from mine").fetchone()[0]
    if not items or len(items) < before * 0.8:   # a partial answer must never shrink the list
        log(f"  could not list your whole store ({listed} listed, {len(items)} usable, {before} known); keeping the known ones")
        return
    have = {r[0] for r in db.execute("select id from mine where feat is not null")}
    urls = thumb_urls([d["id"] for d in items if d["id"] not in have])
    for d in items:
        feat = None
        if d["id"] not in have:
            u = urls.get(d["id"])
            f = fetch_feat(u)[0] if isinstance(u, str) and u.startswith("http") else None
            feat = pack(f) if f else None
            db.execute("insert or replace into mine values(?,?,?,?,?,?,?)",
                       (d["id"], d["name"], d.get("description") or "", d.get("itemCreatedUtc") or "",
                        d.get("assetType"), u if isinstance(u, str) else "", feat))
        else:
            db.execute("update mine set name=?, descr=? where id=?",
                       (d["name"], d.get("description") or "", d["id"]))
    db.execute(f"delete from mine where id not in ({','.join(str(d['id']) for d in items)})")
    db.commit(); kv_set(db, "mine_at", time.time())
    log(f"  {len(items)} of your items indexed ({listed} listed in your store)")
    if before and len(items) >= before + 5:      # many more references than before: compare everything again
        db.execute("delete from seen where id not in (select id from checked where verdict<>'')")
        db.commit()
        for k in ("sweep_at", "sweep_pos", "stores_at"):
            kv_set(db, k, 0)
        log("  your item list grew, so earlier items will be compared again")


def load_mine(db):
    rows = db.execute("select id,name,descr,created,atype,thumb,feat from mine").fetchall()
    return [{"id": r[0], "name": r[1], "descr": r[2], "created": r[3], "atype": r[4],
             "thumb": r[5], "feat": unpack(r[6]) if r[6] else None} for r in rows]


def keywords(mine):
    """Distinct search phrases from your item names, colours stripped."""
    out = {}
    for m in mine:
        t = tokens(m["name"])
        for k in {" ".join(t), " ".join(t[-2:])}:        # full name, and its last two words ("wolf tail")
            if len(k) >= 6 and (k not in out or m["created"] < out[k]):
                out[k] = m["created"]
    return out


# ---------------------------------------------------------------- scoring ---
def judge(cand, feat, mine, cfg):
    """Compare one candidate with all of your items. verdict is None when nothing is wrong."""
    older = [m for m in mine if not cand["created"] or not m["created"] or m["created"] <= cand["created"]]
    if not older:
        return None
    vis = (0.0, 0.0, 0.0, False); vm = None
    if feat:
        for m in older:
            if m["feat"] is None:
                continue
            s = visual_score(feat, m["feat"])
            if s[0] > vis[0]:
                vis, vm = s, m
    nm = max(older, key=lambda m: name_score(cand["name"], m["name"]))
    ns = name_score(cand["name"], nm["name"])
    dm = max(older, key=lambda m: desc_score(m["descr"], cand["descr"]))
    ds = desc_score(dm["descr"], cand["descr"])
    my_handles = set().union(*(handles(m["descr"]) for m in mine)) if mine else set()
    stolen_credit = sorted(handles(cand["descr"]) & my_handles)

    v = vis[0]
    if v >= cfg["copy_threshold"]:
        verdict, colour = "Likely COPY / recolour of your model", 0xE74C3C
    elif v >= cfg["remake_threshold"]:
        verdict, colour = "Possible REMAKE of your model", 0xE67E22
    elif v >= cfg["combo_threshold"] and (ns >= 0.8 or ds >= 0.3 or stolen_credit):
        verdict, colour = "Possible REMAKE: similar shape + matching name/description", 0xE67E22
    elif cfg["text_alerts"] and (stolen_credit or ds >= 0.5 or (ns >= 0.9 and ds >= 0.25)):
        verdict, colour = "Listing text copied (model looks different)", 0xF1C40F
    else:
        verdict, colour = None, 0x95A5A6
    text = 0.3 * ns + 0.4 * max(ds, 1.0 if stolen_credit else 0.0)
    conf = min(0.99, v + (1 - v) * text) if v >= cfg["combo_threshold"] \
        else min(0.6, 0.25 + 0.35 * ds + (0.2 if stolen_credit else 0))
    return {"verdict": verdict, "colour": colour, "confidence": conf, "visual": v,
            "shape": vis[1], "detail": vis[2], "mirrored": vis[3], "name": ns, "desc": ds,
            "credit": stolen_credit, "ai": None, "visual_match": vm,
            "match": vm if (vm and v >= cfg["combo_threshold"]) else (dm if ds >= 0.25 or stolen_credit else (vm or nm))}


# --------------------------------------------------------------------- ai ---
AI_PROMPT = """You are checking whether a newly published Roblox accessory copies an existing one.
Image 1 is the ORIGINAL ("{mine}"). Image 2 is the NEW item ("{theirs}").
Both are catalog thumbnails rendered from the same fixed camera. Ignore colour and texture colour.
Look at silhouette, pose/curl, proportions, and the shape, size and placement of individual
details (tufts, spikes, folds, parts).
Answer with JSON only:
{{"relation": "same_model" | "remake" | "similar_style" | "unrelated",
  "confidence": 0-100, "reason": "one short sentence naming the deciding details"}}
same_model = the same 3D mesh, possibly recoloured, retextured, mirrored or rescaled.
remake = a different mesh clearly modelled after the original (same pose and distinctive details).
similar_style = same kind of item or trend, but independently made.
unrelated = different kind of item."""


def _ai_png(img_bytes):
    im = Image.open(io.BytesIO(img_bytes)).convert("RGBA").resize((384, 384))
    bg = Image.new("RGBA", im.size, (120, 124, 130, 255))   # mid grey: black and white items both show
    bg.alpha_composite(im)
    out = io.BytesIO(); bg.convert("RGB").save(out, "PNG")
    return base64.b64encode(out.getvalue()).decode()


def ai_providers(cfg):
    p = cfg["ai_provider"]
    return [n for n, k in (("anthropic", "anthropic_api_key"), ("gemini", "gemini_api_key"))
            if p in ("auto", n) and cfg[k] and n not in AI_DOWN]


def ai_provider(cfg):
    return (ai_providers(cfg) or [None])[0]


def ai_configured(cfg):
    return cfg["ai_provider"] != "off" and bool(cfg["anthropic_api_key"] or cfg["gemini_api_key"])


def ai_allowance(cfg, db):
    """Today's AI budget: the daily amount plus whatever earlier days left unused (at most $0.30 extra)."""
    day, daily = time.strftime("%Y-%m-%d"), cfg.get("ai_daily_budget_usd", 0.06)
    a = kv_get(db, "ai_allow")
    if not a or a["day"] != day:
        used = kv_get(db, "ai_spend", {}).get(a["day"], 0) if a else 0
        carry = min(0.30, max(0.0, a["carry"] + daily - used)) if a else cfg.get("ai_catchup_usd", 0.20)
        a = {"day": day, "carry": carry}
        kv_set(db, "ai_allow", a)
    return daily + a["carry"]


def ai_ready(cfg, db):
    """True when a vision model can be asked right now (a provider is up and today's budget is not used)."""
    day = time.strftime("%Y-%m-%d")
    if not ai_provider(cfg) or kv_get(db, "ai_calls", {}).get(day, 0) >= cfg["ai_max_calls_per_day"]:
        return False
    if kv_get(db, "ai_spend", {}).get(day, 0) >= ai_allowance(cfg, db):
        STATS["note"] = "daily AI budget reached, the rest waits for tomorrow"
        return False
    return True


SEVERITY = ["unrelated", "similar_style", "remake", "same_model"]


def ai_compare(cfg, db, mine_png, theirs_png, mine_name, theirs_name):
    """Ask every configured vision model; if they disagree, keep the more cautious answer."""
    for p in ai_providers(cfg):                  # Claude first; Gemini only when Claude gives no answer
        o = ai_one(cfg, db, mine_png, theirs_png, mine_name, theirs_name, p)
        if o:
            return o
    return None
    got = []                                     # (older two-model comparison below is no longer reached)
    if len(got) < 2:
        return got[0] if got else None
    a, b = sorted(got, key=lambda o: SEVERITY.index(o["relation"]))
    agree = a["relation"] == b["relation"]
    return {"relation": a["relation"],
            "confidence": max(a["confidence"], b["confidence"]) if agree else min(a["confidence"], b["confidence"]),
            "reason": " | ".join(f"{o['provider']}: {o['relation'].replace('_', ' ')}, {o['reason']}" for o in got)[:600],
            "provider": "both agree" if agree else "they disagree"}


def ai_one(cfg, db, mine_png, theirs_png, mine_name, theirs_name, prov):
    """Ask one vision model. Returns {"relation","confidence","reason","provider"} or None."""
    if not prov or not mine_png or not theirs_png:
        return None
    day = time.strftime("%Y-%m-%d")
    used = kv_get(db, "ai_calls", {})
    if used.get(day, 0) >= cfg["ai_max_calls_per_day"]:
        return None
    spend = kv_get(db, "ai_spend", {})
    if spend.get(day, 0) >= ai_allowance(cfg, db):   # real cost, from the token counts Claude reports
        STATS["note"] = "daily AI budget reached, the rest waits for tomorrow"; return None
    prompt = AI_PROMPT.format(mine=mine_name[:80], theirs=theirs_name[:80])
    a, b = _ai_png(mine_png), _ai_png(theirs_png)
    try:
        if prov == "anthropic":
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=60,
                headers={"x-api-key": cfg["anthropic_api_key"], "anthropic-version": "2023-06-01",
                         "content-type": "application/json",
                         **({"anthropic-workspace-id": cfg["anthropic_workspace_id"]}
                            if cfg["anthropic_workspace_id"] else {})},
                json={"model": cfg["anthropic_model"], "max_tokens": 1500, "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": a}},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b}},
                    {"type": "text", "text": prompt}]}]})
        else:
            r = requests.post("https://generativelanguage.googleapis.com/v1beta/models/"
                              f"{cfg['gemini_model']}:generateContent", timeout=60,
                headers={"x-goog-api-key": cfg["gemini_api_key"], "content-type": "application/json"},
                json={"contents": [{"parts": [{"inline_data": {"mime_type": "image/png", "data": a}},
                                              {"inline_data": {"mime_type": "image/png", "data": b}},
                                              {"text": prompt}]}]})
        if r.status_code != 200:
            STATS["ai_failed"] += 1; STATS["note"] = f"{prov} returned HTTP {r.status_code}"
            if prov == "anthropic" and ("credit" in r.text.lower() or r.status_code in (401, 402, 403)) \
                    and kv_get(db, "ai_alert_day") != day:      # tell the owner once a day that Claude has stopped
                kv_set(db, "ai_alert_day", day)
                send(cfg, {"username": "UGC Watch", "embeds": [{"title": "Claude reviews have stopped", "color": 0xE74C3C,
                     "description": "Claude refused the last review, which usually means the API credit has run out "
                     "or the key was changed. Add credit in the Claude Console to bring AI reviews back. "
                     "Until then every item is still checked by the picture and text comparison."}]})
            if r.status_code == 429:             # out of quota: stop asking this model for the rest of the run
                AI_DOWN.add(prov)
            log(f"  AI check failed ({prov} HTTP {r.status_code}): {r.text[:300]}")
            return None
        body = r.json()
        parts = body.get("content", []) if prov == "anthropic" else body["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        STATS["ai_ok"] += 1
        u = body.get("usage", {}) if prov == "anthropic" else {}
        cost = (u.get("input_tokens", 0) * cfg.get("ai_price_in", 0.10) + u.get("output_tokens", 0) * cfg.get("ai_price_out", 0.50)) / 1e6
        kv_set(db, "ai_spend", {day: spend.get(day, 0) + cost, "total": spend.get("total", 0) + cost})
        kv_set(db, "ai_calls", {day: used.get(day, 0) + 1})
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            STATS["ai_failed"] += 1; STATS["note"] = f"{prov} gave no usable answer"
            log(f"  AI check: no answer from {prov}: {str(body)[:300]}")
            return None
        j = json.loads(m.group(0))
        if j.get("relation") not in ("same_model", "remake", "similar_style", "unrelated"):
            return None
        return {"relation": j["relation"], "confidence": float(j.get("confidence", 0)),
                "reason": str(j.get("reason", ""))[:300], "provider": prov}
    except Exception as e:
        log("  AI check error:", repr(e))
        return None


def apply_ai(res, ai, cfg):
    """Let the AI opinion confirm, promote or dismiss the automatic verdict."""
    res["ai"] = ai
    if not ai:
        return res
    strong_text = bool(res["credit"]) or res["desc"] >= 0.5
    rel, c = ai["relation"], ai["confidence"] / 100
    if rel == "same_model":
        res.update(verdict="Likely COPY / recolour of your model (AI agrees)", colour=0xE74C3C,
                   confidence=max(res["confidence"], min(0.99, 0.5 + 0.5 * c)),
                   match=res["visual_match"] or res["match"])
    elif rel == "remake":
        res.update(verdict="Possible REMAKE of your model (AI agrees)", colour=0xE67E22,
                   confidence=max(min(res["confidence"], 0.9), 0.4 + 0.4 * c),
                   match=res["visual_match"] or res["match"])
    elif strong_text and cfg["text_alerts"]:
        res.update(verdict="Listing text copied (AI: model is different)", colour=0xF1C40F,
                   confidence=min(res["confidence"], 0.6))
    elif res["visual"] >= cfg["copy_threshold"] and c < 0.8:
        pass                                     # near-identical render: keep the alert, show the disagreement
    elif rel == "similar_style" and res["verdict"]:
        res.update(verdict="Similar look and name (AI: same style, different model)", colour=0x3498DB,
                   confidence=min(res["confidence"], 0.5))
    else:
        res["verdict"] = None                    # AI says unrelated: no alert
    return res


# ---------------------------------------------------------------- discord ---
def send(cfg, payload, url=None):
    url = url or cfg["discord_webhook_url"]
    if not url:
        log("  (no webhook set) ", payload["embeds"][0]["title"]); return True
    for _ in range(5):
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.3); continue
        return r.status_code < 300
    return False


def when(iso):
    """ISO date -> Discord timestamp, shown in the reader's own time zone."""
    try:
        return f"<t:{calendar.timegm(time.strptime(iso[:19], '%Y-%m-%dT%H:%M:%S'))}:f>"
    except Exception:
        return iso[:10] or "unknown"


def pct(v):
    """Display scale. Any two solid shapes overlap about 40%, so that baseline is removed."""
    return max(0.0, (v - 0.4) / 0.6)


def label(v):
    for a, b in (("Likely COPY", "Likely copy or recolour"), ("Possible REMAKE", "Possible remake"),
                 ("Listing text", "Listing text copied, model looks different"), ("Similar look", "Similar look and name")):
        if v.startswith(a):
            return b
    return v


def takedown(cand, res, m):
    """(level, text): how strong a takedown case this is, and a draft description for the request."""
    ai = res["ai"]["relation"] if res["ai"] else ""
    copy = res["verdict"].startswith("Likely COPY")
    if copy and (ai == "same_model" or (not ai and res["visual"] >= 0.9) or (res.get("mesh") or 0) >= 0.985):
        level = "Strong: same model as yours"
    elif copy:
        level = "Possible: compare both in the 3D viewer first"
    else:
        return "Not on this evidence", ""
    d0, d1 = m["created"][:10], cand["created"][:10]
    try:
        gap = (calendar.timegm(time.strptime(d1, "%Y-%m-%d")) - calendar.timegm(time.strptime(d0, "%Y-%m-%d"))) // 86400
    except Exception:
        gap = -1
    verb = "reproduces" if level.startswith("Strong") else "appears to be copied from"
    t = [f"Original work: {m['name']}", f"https://www.roblox.com/catalog/{m['id']}", f"Published {d0} (UTC) by my group.", "",
         f"Infringing item: {cand['name']}", f"https://www.roblox.com/catalog/{cand['id']}",
         f"Published {d1} (UTC) by {cand['creator']}.", "",
         f"The infringing item {verb} the 3D model of my original accessory. Rendered from the same catalog camera, "
         "the two items share the same silhouette, proportions and placement of individual details"
         + (" (the copy is mirrored or rotated)" if res["mirrored"] else "")
         + f"; an automated image comparison scored them {pct(res['visual']):.0%} alike."]
    if res["ai"]:
        t.append("Side-by-side review: " + res["ai"]["reason"])
    if gap >= 0:
        t.append(f"My item was published {gap} day{'s' if gap != 1 else ''} before the infringing item.")
    if res["credit"] or res["desc"] >= 0.5:
        t.append("Their listing description also reuses text from my listing"
                 + (", including my modeller credit " + ", ".join(res["credit"]) if res["credit"] else "") + ".")
    t.append("I am the rights holder of the original item and did not authorise this use.")
    return level, "\n".join(t)


PENDING, TIERS = {}, {}
RANK = [("Likely COPY", "copy"), ("Possible REMAKE", "remake"), ("Listing text", "text"), ("Similar look", "similar")]


def tier(v):
    return next((k for k, (p, _) in enumerate(RANK) if v.startswith(p)), 9)


KIND = {8: "Hat", 41: "Hair Accessory", 42: "Face Accessory", 43: "Neck Accessory", 44: "Shoulder Accessory",
        45: "Front Accessory", 46: "Back Accessory", 47: "Waist Accessory"}


ACCESSORY = {8, 41, 42, 43, 44, 45, 46, 47}


def stamp():
    """Footer date and time for a card, shown in the reader's own time zone."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def who(c):
    return f"[{c['creator']}](https://www.roblox.com/{'communities' if c.get('ctype') == 'Group' else 'users'}/{c.get('cid')})"


def cost(c):
    return "Off sale" if c.get("price") is None else str(c["price"])


def fld(n, v):
    return {"name": n, "value": str(v), "inline": True}


def alert(cfg, cand, res):
    """Queue the alert; flush_alerts sends one card per item at the end of the run."""
    PENDING.setdefault("all", []).append((cand, res))
    return True


def flush_alerts(cfg):
    day = when                                   # full date and time
    for cand, res in sorted(PENDING.get("all", []), key=lambda x: (tier(x[1]["verdict"]), -x[1]["visual"])):
        m = res["match"]
        level, text = takedown(cand, res, m)
        key = "strong" if level.startswith("Strong") else "possible" if text else RANK[min(tier(res["verdict"]), 3)][1]
        TIERS[key] = TIERS.get(key, 0) + 1
        link = f"https://www.roblox.com/catalog/{cand['id']}"
        why = []
        if res.get("mesh"):
            if res["mesh"] >= 0.5:               # a low number in an alert only confuses
                why.append("3D model is the same as yours" if res["mesh"] >= 1.0 else f"3D shape {res['mesh']:.0%} the same as yours")
        m3 = res.get("m3d") or {}
        if m3.get("counts") and not m3.get("same_mesh"):
            why.append("Same vertex and face count as your mesh")
        if m3.get("inner"):
            why.append(f"Their file is still named \"{m3['inner'][0][:60]}\", same as yours")
        if AI_MESH.search((m3.get("theirs") or {}).get("mesh_name") or ""):
            why.append("Their mesh name looks like an AI 3D-generator export")
        if res.get("render") is not None:
            why.append(f"Bare mesh without texture looks {pct(res['render']):.0%} like yours")
        if res.get("tex"):
            why.append("Uses the same texture image as yours")
        elif (res.get("texsim") or 0) >= 0.6:
            why.append(f"Texture image {res['texsim']:.0%} the same as yours")
        if res.get("mesh_info"):
            why.append("Their mesh: " + res["mesh_info"])
        if res["credit"]:
            why.append("Uses your credit " + ", ".join(res["credit"]))
        elif res["desc"] >= 0.5:
            why.append(f"Description {res['desc']:.0%} the same as yours")
        if res["ai"]:
            why.append(res["ai"]["reason"][:400])
        main = {"author": {"name": "UGC Watch"}, "title": cand["name"][:250], "url": link, "color": res["colour"],
                "description": f"**{label(res['verdict'])}**\n{KIND.get(cand.get('atype'), 'Accessory')} \u00b7 [Roblox Page]({link})"
                               + ("\n\n" + "\n".join(why) if why else ""),
                "fields": [fld("Creator", who(cand)), fld("Price", cost(cand)), fld("Uploaded", day(cand["created"])),
                           fld("Looks like yours", f"{pct(res['visual']):.0%}"), fld("Takedown", level.split(":")[0])],
                "footer": {"text": "Detected"}, "timestamp": stamp()}
        mine = {"author": {"name": "Your original"}, "title": m["name"][:250], "color": 0x2B2D31,
                "url": f"https://www.roblox.com/catalog/{m['id']}", "description": f"Uploaded {day(m['created'])}"}
        if cand.get("thumb"):
            main["thumbnail"] = {"url": cand["thumb"]}
        if m.get("thumb"):
            mine["thumbnail"] = {"url": m["thumb"]}
        embeds = [main, mine] + (mesh_cards(res["m3d"], res["colour"]) if res.get("m3d") else [])
        if text:
            embeds.append({"author": {"name": "Takedown request text"}, "color": res["colour"],
                           "description": "```\n" + text[:3800] + "\n```",
                           "footer": {"text": "Draft. Check every line is true before sending it to Roblox."}})
        if send(cfg, {"username": "UGC Watch", "embeds": embeds}):
            STATS["alerts"] += 1
        time.sleep(0.8)


def to_cand(d):
    return {"id": d["id"], "name": d.get("name") or "", "descr": d.get("description") or "",
            "created": d.get("itemCreatedUtc") or "", "atype": d.get("assetType"),
            "creator": d.get("creatorName") or "?", "cid": d.get("creatorTargetId"),
            "ctype": d.get("creatorType") or "User", "price": d.get("price")}


CSRF = {"t": ""}


def details_batch(ids):
    """Full details for up to 100 item ids per request (name, creator, upload date, description)."""
    out = []
    for k in range(0, len(ids), 100):
        body = {"items": [{"itemType": "Asset", "id": i} for i in ids[k:k + 100]]}
        delay = 2.0
        for _ in range(6):
            try:
                r = sess.post(CATALOG.replace("/search/items/details", "/catalog/items/details"), json=body,
                              headers={"x-csrf-token": CSRF["t"]}, timeout=25)
            except requests.RequestException as e:
                log("  network error:", e); time.sleep(delay); continue
            if r.status_code == 403 and r.headers.get("x-csrf-token"):
                CSRF["t"] = r.headers["x-csrf-token"]; continue      # first call hands out the token
            if r.status_code == 429:
                STATS["waited"] += delay; time.sleep(delay); delay = min(delay * 2, 60); continue
            if r.status_code == 200:
                out += r.json().get("data", [])
            else:
                log(f"  HTTP {r.status_code} from item details: {r.text[:120]}")
            break
        time.sleep(1.0)
    return out


HARD = {"t": float("inf")}                       # a run must stop in time for its results to be saved


def search_wide(url, pages, **params):
    """120 results per request. v1/search/items returns bare ids, v2 returns full details."""
    cursor = None
    for _ in range(pages):
        if time.time() > HARD["t"]:
            log("  time limit for this run reached, the rest continues next run"); return
        p = {"Limit": 120, **params}
        if cursor:
            p["Cursor"] = cursor
        j = http_json(url, p)
        time.sleep(1.2)
        if not j or not j.get("data"):
            return
        yield j["data"]
        cursor = j.get("nextPageCursor")
        if not cursor:
            return


def is_seen(db, i):
    return db.execute("select 1 from seen where id=?", (i,)).fetchone() is not None


def mark(db, i):
    db.execute("insert or ignore into seen values(?,?)", (i, time.time()))


def get_thumbs(ids):
    urls = thumb_urls(ids)
    ok = [u for u in urls.values() if isinstance(u, str) and u.startswith("http")]
    with ThreadPoolExecutor(8) as pool:          # download and analyse thumbnails in parallel
        return urls, dict(zip(ok, pool.map(fetch_feat, ok)))


def asset_bytes(aid, cfg):
    """Download a Roblox asset file (model, mesh or image). Needs the signed-in cookie; None if refused."""
    for k in range(4):                           # Roblox says "too many requests" now and then: wait and ask again
        r = sess.get(f"https://assetdelivery.roblox.com/v2/assetId/{aid}",
                     headers={"Cookie": ".ROBLOSECURITY=" + cfg["roblox_cookie"]}, timeout=25)
        if r.status_code != 429:
            break
        wait = float(r.headers.get("Retry-After") or 0) or 3 * (k + 1)
        STATS["waited"] += wait; time.sleep(min(wait, 30))
    loc = (r.json().get("locations") or [{}])[0].get("location") if r.text[:1] == "{" else None
    time.sleep(0.4)
    if not loc:
        STATS["mesh_fail"] += 1
        if STATS["mesh_fail"] <= 3:
            log(f"  asset {aid} not given: HTTP {r.status_code} {r.text[:140]}")
        return None
    return sess.get(loc, timeout=40).content         # the file itself comes from the CDN, without the cookie


def _lz4(src, size):
    out, i = bytearray(), 0
    while i < len(src):
        t = src[i]; i += 1
        n = t >> 4
        if n == 15:
            while True:
                b = src[i]; i += 1; n += b
                if b != 255:
                    break
        out += src[i:i + n]; i += n
        if i >= len(src):
            break
        off = src[i] | src[i + 1] << 8; i += 2
        n = t & 15
        if n == 15:
            while True:
                b = src[i]; i += 1; n += b
                if b != 255:
                    break
        for _ in range(n + 4):
            out.append(out[-off])
    return bytes(out[:size])


def mesh_geom(data):
    """Vertex positions and face count from a Roblox mesh file (versions 1 to 7). (None, 0) if unreadable."""
    if data[:8] != b"version ":
        return None, 0
    ver, start = data[8:9], data.index(b"\n") + 1
    if ver == b"1":
        g = re.findall(rb"\[([^\]]+)\]", data[start:])[0::3]
        v = np.array([[float(x) for x in t.split(b",")] for t in g], np.float32)
        return v, len(v) // 3
    if ver in b"23":
        hsize, vsize = struct.unpack_from("<HB", data, start)
        n, nf = struct.unpack_from("<II", data, start + hsize - 8)
    elif ver in b"45":
        hsize, _, n, nf = struct.unpack_from("<HHII", data, start)
        vsize = 40
    else:
        return mesh_points_chunked(data, start)
    if not 0 < n < 2_000_000 or start + hsize + n * vsize > len(data):
        return None, 0
    a = np.frombuffer(data, np.uint8, n * vsize, start + hsize).reshape(n, vsize)
    return a[:, :12].copy().view("<f4").reshape(n, 3), nf


def mesh_points(data):
    return mesh_geom(data)[0]


DRACO = {"tried": False, "mod": None}


def _draco():
    """Decoder for the compressed meshes Roblox uses since mesh version 7; fetched on first use."""
    if not DRACO["tried"]:
        DRACO["tried"] = True
        try:
            import DracoPy
        except ImportError:
            try:
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "DracoPy"], timeout=180, check=True)
                import DracoPy
            except Exception as e:
                log("  could not get the mesh decoder:", repr(e)[:100]); return None
        DRACO["mod"] = DracoPy
    return DRACO["mod"]


def mesh_points_chunked(data, i):
    """Mesh versions 6 and 7: a list of named chunks; the vertices are in COREMESH."""
    seen = []
    while i + 16 <= len(data):
        kind = data[i:i + 8].rstrip(b"\0")
        ver, size = struct.unpack_from("<II", data, i + 8)
        body = data[i + 16:i + 16 + size]; i += 16 + size
        seen.append(f"{kind.decode('latin1')} v{ver} {size}b")
        if kind != b"COREMESH" or len(body) < 8:
            continue
        n = struct.unpack_from("<I", body, 0)[0]
        if ver == 1 and 0 < n < 2_000_000 and 8 + n * 40 <= len(body):
            a = np.frombuffer(body, np.uint8, n * 40, 4).reshape(n, 40)
            return a[:, :12].copy().view("<f4").reshape(n, 3), struct.unpack_from("<I", body, 4 + n * 40)[0]
        d = _draco()
        if d:
            for off in (4, 0, 8):                # compressed block, with or without a length in front
                try:
                    m = d.decode(bytes(body[off:]))
                    return np.asarray(m.points, np.float32).reshape(-1, 3), len(np.asarray(m.faces).reshape(-1, 3))
                except Exception:
                    pass
    if not DRACO.get("noted"):
        DRACO["noted"] = True
        log("  mesh layout not decoded, chunks: " + ", ".join(seen)[:300])
    return None, 0


def _d2(v):
    v = v[np.lexsort(v.T)]                       # fixed order first, so a re-saved copy gives the same numbers
    rng = np.random.default_rng(0)
    d = np.linalg.norm(v[rng.integers(0, len(v), 20000)] - v[rng.integers(0, len(v), 20000)], axis=1)
    hist = np.histogram(d / (d.mean() + 1e-9), bins=48, range=(0, 3))[0].astype(np.float32)
    return hist / hist.sum()


VG = 24                                          # voxel grid size for the shape fingerprint


def _vox(v):
    """The mesh turned to a standard pose (centred, main axes lined up, sized to fit) and marked into a 24x24x24
    grid. Moving, turning, resizing, mirroring, recolouring or re-saving a copy leaves this the same."""
    v = v[np.isfinite(v).all(1)].astype(np.float64)
    c = v - v.mean(0)
    _, axes = np.linalg.eigh(c.T @ c)           # main axes from every point, so point order does not matter
    p = c @ axes[:, ::-1]
    p *= np.where(((p ** 3).mean(0)) < 0, -1.0, 1.0)      # point each axis the same way every time
    p /= (np.abs(p).max() + 1e-9)
    g = np.zeros((VG, VG, VG), bool)
    ix = np.clip(((p + 1) / 2 * (VG - 1)).round().astype(int), 0, VG - 1)
    g[ix[:, 0], ix[:, 1], ix[:, 2]] = True
    return np.packbits(g)


def _grid(a):
    return np.unpackbits(a)[:VG ** 3].reshape(VG, VG, VG).astype(bool)


def _thick(g):
    d = g.copy()
    for ax in range(3):
        d |= np.roll(g, 1, ax) | np.roll(g, -1, ax)
    return d


def vox_score(a, b):
    """(strict, slack), each 0..1, trying each mirror image. strict: cells shared exactly (copies 0.8+, other
    models rarely above 0.6). slack: how much of each shape lies within one cell of the other (remakes of the same
    base shape 0.9+). (None, None) if either is missing."""
    if a is None or b is None:
        return None, None
    ga, gb = _grid(a), _grid(b)
    da, na = _thick(ga), max(1, ga.sum())
    strict = slack = 0.0
    for f in range(8):
        h = gb[::-1 if f & 1 else 1, ::-1 if f & 2 else 1, ::-1 if f & 4 else 1]
        strict = max(strict, (ga & h).sum() / max(1, (ga | h).sum()))
        slack = max(slack, min((ga & _thick(h)).sum() / na, (h & da).sum() / max(1, h.sum())))
    return float(strict), float(slack)


STD_NAMES = {"handle", "mesh", "thumbnailconfiguration", "thumbnailcameratarget", "thumbnailcameravalue",
             "avatarpartscaletype", "originalsize", "surfaceappearance", "wraplayer", "accessory", "specialmesh",
             "meshpart", "weld", "accessoryweld", "rigidconstraint"}


def _strings(body):
    out, i = [], 1                               # first byte is the value type; 1 = text
    if not body or body[0] != 1:
        return out
    while i + 4 <= len(body):
        n = struct.unpack_from("<I", body, i)[0]
        if n > 500 or i + 4 + n > len(body):
            break
        out.append(body[i + 4:i + 4 + n].decode("utf-8", "replace")); i += 4 + n
    return out


GENERIC_NAMES = {"handleaccessory", "accessory", "handle", "mesh", "default", "model", "item", "ugc", "hat", "tail"}


def _inner_name(x):
    """The meaningful part of a name stored inside a file: "Accessory (Black Fluffy Fox Tail)" gives
    "Black Fluffy Fox Tail"; standard part names and Roblox's generic defaults give ""."""
    x = (x or "").strip()
    m = re.fullmatch(r"\w+ \((.*)\)", x)
    if m:
        x = m.group(1).strip()
    low = x.lower()
    if not x or low in STD_NAMES or low in GENERIC_NAMES or low.endswith("attachment") or len(tokens(x)) < 2:
        return ""
    return x


def model_ids(data):
    """Mesh and texture asset ids inside an accessory file, plus the names of the parts inside it."""
    blobs, names = [], []
    if data[:8] == b"<roblox!":
        i = 32
        while i + 16 <= len(data):
            name, clen, ulen = data[i:i + 4], *struct.unpack_from("<II", data, i + 4)
            i += 16
            raw = data[i:i + (clen or ulen)]; i += clen or ulen
            if clen and raw[:4] == b"\x28\xb5\x2f\xfd":
                import zstandard
                raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=ulen)
            elif clen:
                raw = _lz4(raw, ulen)
            if name == b"PROP" and len(raw) > 8:
                n = struct.unpack_from("<I", raw, 4)[0]
                pname, body = raw[8:8 + n].decode("latin1").lower(), raw[8 + n:]
                blobs.append((pname, body))
                if pname == "name":
                    names += _strings(body)
            if name == b"END\0":
                break
    else:                                        # older XML model file
        for m in re.finditer(rb'name="(\w+)"[^>]*>(?:<url>)?([^<]*)', data):
            blobs.append((m.group(1).decode().lower(), m.group(2)))
            if m.group(1) == b"Name":
                names.append(m.group(2).decode("utf-8", "replace"))
    out = {"names": [x for x in map(_inner_name, names) if x]}
    for name, body in blobs:
        ids = re.findall(rb"(?:rbxassetid://|asset/\?id=|assetid=)(\d{5,})", body, re.I)
        kind = "mesh" if "mesh" in name else "tex" if ("texture" in name or "colormap" in name) else None
        if kind and ids and kind not in out:
            out[kind] = int(ids[0])
    return out


AI_MESH = re.compile(r"tripo|meshy|rodin|hyper3d|luma|csm_|kaedim|sloyd|genie|hunyuan", re.I)


def mesh_desc(asset_id, cfg, previews=True):
    """What an item is really built from: mesh and texture ids, file hashes, shape fingerprints,
    vertex and face counts, the texture image and the names stored inside its file."""
    try:
        data = asset_bytes(asset_id, cfg)
        ids = model_ids(data) if data else {}
        if ids.get("mesh"):
            out = {"mesh_id": ids["mesh"], "tex_id": ids.get("tex"), "names": ids.get("names", [])}
            mb = asset_bytes(ids["mesh"], cfg)
            if mb:
                out["sha"] = hashlib.sha256(mb).hexdigest()
                v, nf = mesh_geom(mb)
                if v is not None and len(v) >= 50 and np.isfinite(v).any():
                    out.update(n=len(v), nf=nf, hist=_d2(v[np.isfinite(v).all(1)]), vox=_vox(v))
                else:
                    log(f"  mesh {ids['mesh']} of item {asset_id}: format {mb[:12]!r} not readable, comparing by id and file only")
            tb = asset_bytes(ids["tex"], cfg) if ids.get("tex") else None
            if tb:
                out["tex_sha"] = hashlib.sha256(tb).hexdigest()
                try:                             # the texture image itself, shrunk, for a likeness score
                    g = np.asarray(Image.open(io.BytesIO(tb)).convert("L").resize((32, 32)), np.float32)
                    if g.std() >= 4:             # a flat single-colour texture says nothing
                        out["texv"] = (g - g.mean()) / g.std()
                except Exception:
                    pass
            try:                                 # Roblox's own previews of the bare mesh and of the texture
                want = [i for i in (ids["mesh"], ids.get("tex")) if i]
                tu = thumb_urls(want)
                if any(tu.get(i) == "Pending" for i in want):        # preview still being drawn: ask once more
                    time.sleep(3); tu = thumb_urls(want)
                u = tu.get(ids["mesh"])
                if isinstance(u, str) and u.startswith("http"):
                    out["render_url"] = u
                    if previews:
                        out["render"] = fetch_feat(u)[0]
                u = tu.get(ids.get("tex"))
                if isinstance(u, str) and u.startswith("http"):
                    out["tex_url"] = u
                j = http_json(f"https://economy.roblox.com/v2/assets/{ids['mesh']}/details", tries=2) or {}
                if j.get("Name"):
                    out["mesh_name"], out["mesh_by"] = j["Name"], (j.get("Creator") or {}).get("Name", "?")
                    out["mesh_date"] = j.get("Created") or ""
                    out["mesh_info"] = f"\"{j['Name'][:60]}\" uploaded by {out['mesh_by']} on {out['mesh_date'][:10]}"
            except Exception as e:
                log("  mesh preview unavailable:", repr(e)[:100])
            return out
        if data:
            log(f"  item {asset_id}: no mesh id found in its file (starts {data[:16]!r})")
    except Exception as e:
        STATS["mesh_fail"] += 1
        log("  model file unavailable:", repr(e)[:120])
    return mesh_desc_3d(asset_id, cfg)


def mesh_desc_3d(asset_id, cfg):
    """Fallback: shape fingerprint from the 3D thumbnail. None if unavailable."""
    try:
        h = {"Cookie": ".ROBLOSECURITY=" + cfg["roblox_cookie"]}
        for _ in range(4):
            r = sess.get("https://thumbnails.roblox.com/v1/assets-thumbnail-3d", params={"assetId": asset_id},
                         headers=h, timeout=25)
            j = r.json() if r.text[:1] == "{" else {}
            if j.get("state") == "Completed":
                break
            time.sleep(2)
        else:
            STATS["mesh_fail"] += 1
            if STATS["mesh_fail"] <= 3:
                log(f"  3D model not given for {asset_id}: HTTP {r.status_code} {r.text[:140]}")
            return None
        meta = sess.get(j["imageUrl"], timeout=25).json()
        k = 31
        for ch in meta["obj"][:32]:
            k ^= ord(ch)
        obj = ""
        for n in [k % 8] + [x for x in range(8) if x != k % 8]:     # the file sits on one of eight hosts
            obj = sess.get(f"https://t{n}.rbxcdn.com/{meta['obj']}", timeout=40).text
            if "\nv " in obj[:4000] or obj.startswith("v "):
                break
        v = np.array([[float(x) for x in ln.split()[1:4]] for ln in obj.splitlines() if ln.startswith("v ")], np.float32)
        if len(v) < 50:
            STATS["mesh_fail"] += 1
            log(f"  3D model for {asset_id} unreadable: {len(v)} points, starts {obj[:60]!r}")
            return None
        rng = np.random.default_rng(0)           # same seed: an identical mesh gives an identical fingerprint
        d = np.linalg.norm(v[rng.integers(0, len(v), 20000)] - v[rng.integers(0, len(v), 20000)], axis=1)
        hist = np.histogram(d / (d.mean() + 1e-9), bins=48, range=(0, 3))[0].astype(np.float32)
        time.sleep(1.0)
        return {"n": len(v), "hist": hist / hist.sum()}
    except Exception as e:
        STATS["mesh_fail"] += 1
        log("  3D model unavailable:", repr(e)[:120])
        return None


def mesh_score(a, b):
    """Raw shape-distribution likeness (kept for the log). 1.0 = the very same mesh. None = could not compare."""
    if (a.get("mesh_id") and a.get("mesh_id") == b.get("mesh_id")) or (a.get("sha") and a.get("sha") == b.get("sha")):
        return 1.0
    if a.get("hist") is None or b.get("hist") is None:
        return None
    return float(1 - 0.5 * np.abs(a["hist"] - b["hist"]).sum())


def render_score(a, b):
    """How alike the untextured mesh previews look (same scale as the picture score). None if missing."""
    if a.get("render") is None or b.get("render") is None:
        return None
    return float(visual_score(a["render"], b["render"])[0])


def texture_score(a, b):
    """How alike the two texture images are, 0 to 1. None if missing."""
    if a.get("texv") is None or b.get("texv") is None:
        return None
    return float(max(0.0, (a["texv"] * b["texv"]).mean()))


def same_texture(a, b):
    return bool((a.get("tex_id") and a.get("tex_id") == b.get("tex_id")) or (a.get("tex_sha") and a.get("tex_sha") == b.get("tex_sha")))


def geo_compare(t, y):
    """Everything the two real models have in common."""
    same = bool((t.get("mesh_id") and t.get("mesh_id") == y.get("mesh_id")) or (t.get("sha") and t.get("sha") == y.get("sha")))
    vox, slack = (1.0, 1.0) if same else vox_score(t.get("vox"), y.get("vox"))
    counts = bool((t.get("n") or 0) >= 200 and t.get("n") == y.get("n") and t.get("nf") == y.get("nf"))
    mine_names = {_inner_name(n).lower() for n in y.get("names", [])} - {""}
    inner = [x for x in map(_inner_name, t.get("names", [])) if x and x.lower() in mine_names]
    return {"same_mesh": same, "vox": vox, "slack": slack, "d2": mesh_score(t, y), "counts": counts,
            "same_tex": same_texture(t, y), "texsim": texture_score(t, y), "inner": inner}


def geo_strength(g):
    """One number to rank comparisons: 1 = the same mesh, then near-identical shapes, then texture reuse."""
    return max(1.0 if g["same_mesh"] else 0.0, (g["vox"] or 0.0) + 0.3 * (g["slack"] or 0.0) + (0.1 if g["counts"] else 0.0),
               0.9 if g["same_tex"] else 0.0, (g["texsim"] or 0.0) * 0.9, 0.85 if g["inner"] else 0.0)


def my_models(db, cfg, mine):
    """Real-model fingerprints of your items, kept between runs; a few missing ones are filled in each run."""
    import pickle
    db.execute("create table if not exists mymesh(id integer primary key, at real, blob blob)")
    have = {r[0]: r[1] for r in db.execute("select id, blob from mymesh")}
    todo = [m for m in mine if m["id"] not in have][:12]
    for m in todo:
        if time.time() > HARD["t"]:
            break
        d = mesh_desc(m["id"], cfg, previews=False)
        if d and d.get("mesh_id"):
            d.pop("render", None)
            db.execute("insert or replace into mymesh values(?,?,?)", (m["id"], time.time(), pickle.dumps(d)))
            have[m["id"]] = pickle.dumps(d)
    db.commit()
    names = {m["id"]: m for m in mine}
    return [(names[i], pickle.loads(b)) for i, b in have.items() if i in names]


def decide(db, cfg, mine, cand, feat, raw_png, left, src="new"):
    """Full check of one item. Returns 1 if an alert was sent, 0 if not, None to try again next run."""
    STATS["checked"] += 1
    res = judge(cand, feat, mine, cfg)
    wants = bool(res and (res["verdict"] or res["visual"] >= cfg["ai_min_visual"] or res["name"] >= 0.6
                          or res["desc"] >= 0.3 or res["credit"]) and res["visual_match"])
    if wants and ai_provider(cfg) and (left <= 0 or not ai_ready(cfg, db)) \
            and db.execute("select count(*) from later").fetchone()[0] < 5000:
        db.execute("insert or ignore into later values(?)", (cand["id"],))   # needs an AI look: keep it for the next run
        mark(db, cand["id"]); db.commit()
        return 0
    if wants and ai_provider(cfg) and left > 0:
        vm = res["visual_match"]
        if "png" not in vm:
            vm["png"] = fetch_feat(vm["thumb"])[1] if vm["thumb"] else None
        res = apply_ai(res, ai_compare(cfg, db, vm["png"], raw_png, vm["name"], cand["name"]), cfg)
    want3d = bool(res and cfg.get("roblox_cookie") and res["visual_match"] and (
        res["verdict"] or res["visual"] >= cfg["ai_min_visual"] or res["name"] >= 0.6 or res["desc"] >= 0.3
        or res["credit"] or (res["ai"] and res["ai"]["relation"] != "unrelated")))
    theirs = mesh_desc(cand["id"], cfg) if want3d and time.time() < HARD["t"] else None
    if theirs and theirs.get("mesh_id"):
        if MYM["list"] is None:
            MYM["list"] = my_models(db, cfg, mine)
        vm = res["visual_match"]
        if "mesh" not in vm:                     # the item whose picture looks closest, always compared
            vm["mesh"] = next((d for m, d in MYM["list"] if m["id"] == vm["id"]), None) or mesh_desc(vm["id"], cfg, previews=False)
        pool = {m["id"]: (m, d) for m, d in MYM["list"]}
        if vm["mesh"]:
            pool[vm["id"]] = (vm, vm["mesh"])
        scored = [(geo_strength(g), m, d, g) for m, d in pool.values() if d for g in [geo_compare(theirs, d)]]
        if scored:                               # compared with every item of yours, not only the closest picture
            strength, m, d, g = max(scored, key=lambda x: x[0])
            if strength < 0.75 and vm["mesh"]:
                m, d, g = vm, vm["mesh"], geo_compare(theirs, vm["mesh"])
            if d.get("render") is None and d.get("render_url"):
                d["render"] = fetch_feat(d["render_url"])[0]
            g["render"] = render_score(theirs, d)
            STATS["mesh"] += 1
            r, vox = g["render"], g["vox"]
            slack = g["slack"]
            copy = g["same_mesh"] or (vox or 0) >= 0.8 or (g["counts"] and (slack or 0) >= 0.95) \
                or (vox is None and (r or 0) >= 0.92)
            res.update(mesh=1.0 if copy else (min(vox, 0.98) if vox is not None else None), render=r,
                       texsim=g["texsim"], tex=g["same_tex"], mesh_info=theirs.get("mesh_info"))
            mm = {"id": m["id"], "name": m["name"], "thumb": m.get("thumb", ""), "created": m.get("created", "")}
            res["m3d"] = {"theirs": theirs, "yours": d, "mine_name": m["name"], "mine_id": m["id"], "score": vox,
                          "render": r, "texsim": g["texsim"], "same_tex": g["same_tex"], "counts": g["counts"],
                          "same_mesh": g["same_mesh"], "inner": g["inner"]}
            cand["m3d"] = res["m3d"]
            log(f"  3D check {cand['name'][:40]} vs {m['name'][:30]}: shape {vox} slack {slack}, counts {g['counts']}, d2 {g['d2']}, "
                f"bare-mesh {r}, texture {g['texsim']}, same texture {g['same_tex']}, inner {g['inner']}, "
                f"names {theirs.get('names', [])[:3]} | mesh {theirs.get('mesh_id')}/{theirs.get('tex_id')} "
                f"vs {d.get('mesh_id')}/{d.get('tex_id')} | {theirs.get('mesh_info')}")
            db.execute("insert or replace into mesh3d values(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (cand["id"], time.time(), cand["name"], cand["creator"], m["id"], m["name"],
                        theirs.get("mesh_id"), theirs.get("tex_id"), d.get("mesh_id"), d.get("tex_id"),
                        vox, r, g["texsim"], theirs.get("mesh_info") or ""))
            level = None
            ai_no = bool(res["ai"] and res["ai"]["relation"] == "unrelated")    # Claude looked and saw no link
            looks = (slack or 0) >= 0.6 or (r or 0) >= 0.6 or res["visual"] >= 0.6
            if copy:                             # the real 3D model matches: a fact, so it outranks the picture and the AI
                level = ("Likely COPY / recolour of your model (3D model matches)", 0xE74C3C, 0.97)
            elif not res["verdict"] and g["same_tex"]:           # your very texture file: also a fact
                level = ("Possible REMAKE of your model (uses your texture)", 0xE67E22, 0.8)
            elif not res["verdict"] and not ai_no and (g["texsim"] or 0) >= 0.95:
                level = ("Possible REMAKE of your model (uses your texture)", 0xE67E22, 0.75)
            elif not res["verdict"] and not ai_no and g["inner"] and looks:   # a name alone proves nothing
                level = ("Possible REMAKE of your model (their file carries your internal name)", 0xE67E22, 0.75)
            elif not res["verdict"] and not ai_no and (slack or 0) >= 0.9 and (vox or 0) >= 0.45 and (r or 0) >= 0.7:
                level = ("Possible REMAKE of your model (3D shape very close)", 0xE67E22, 0.7)
            if level:
                res.update(verdict=level[0], colour=level[1], confidence=max(res["confidence"], level[2]), match=mm)
            elif res["verdict"] and res["verdict"].startswith("Possible REMAKE") and (vox or 0) < 0.3 \
                    and r is not None and r < 0.45 and res["visual"] < 0.45:
                res.update(verdict="Similar look and name (3D model differs from yours)", colour=0x3498DB)   # meshes clearly differ
    hit = bool(res and res["verdict"])
    if hit and cand.get("cid") in cfg.get("low_priority_creator_ids", []) and tier(res["verdict"]) > 0:
        hit = False                              # creator you reviewed: only outright copies alert
        res["verdict"] += " (low priority creator, not alerted)"
    if hit and left <= 0:
        return None                              # alert budget used up: leave it for the next run
    if hit:
        log(f"  MATCH {res['verdict']} | {cand['name']} by {cand['creator']} "
            f"(visual {res['visual']:.2f}, name {res['name']:.2f}, desc {res['desc']:.2f})")
        if not alert(cfg, cand, res):
            return None
        time.sleep(0.6)
    if res:
        vm, ai = res["visual_match"] or res["match"], res["ai"]
        db.execute("insert or replace into checked values(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (cand["id"], time.time(), cand["name"], cand["creator"], cand["created"], cand.get("thumb", ""),
                    res["visual"], vm["id"], vm["name"], res["name"], res["desc"],
                    res["verdict"] or "", f"{ai['relation']} ({ai['provider']})" if ai else ""))
    near = (res["visual_match"] or res["match"]) if res else None
    ACTIVITY.append((src, cand, res["visual"] if res else 0.0, near["name"] if near else "", hit))
    mark(db, cand["id"]); db.commit()
    return 1 if hit else 0


def process(db, cfg, mine, raw, types, budget, src="new"):
    """Compare items that are not yours and not seen before. Returns alerts sent."""
    skip = set(cfg["ignore_creator_ids"]) | {cfg["my_creator_id"]}
    fresh = {}
    for d in raw:
        if d.get("itemType") != "Asset" or d["id"] in fresh or is_seen(db, d["id"]):
            continue
        if d.get("creatorTargetId") in skip or d.get("assetType") not in types:
            mark(db, d["id"]); continue
        fresh[d["id"]] = to_cand(d)
    db.commit()
    if not fresh:
        return 0
    urls, feats = get_thumbs(fresh)
    sent = 0
    for cid, cand in fresh.items():
        if time.time() > HARD["t"]:              # leave the rest unchecked so the next run picks it up
            break
        u = urls.get(cid)
        if u in ("Pending", "InReview", None):   # thumbnail not ready: look again next time
            continue
        ok = isinstance(u, str) and u.startswith("http")
        cand["thumb"] = u if ok else ""
        feat, raw_png = feats[u] if ok else (None, None)
        sent += decide(db, cfg, mine, cand, feat, raw_png, budget - sent, src) or 0
    return sent


def cycle(db, cfg):
    db.execute("""create table if not exists checked(id integer primary key, at real, name text, creator text,
        created text, thumb text, visual real, mine_id int, mine_name text, name_s real, desc_s real,
        verdict text, ai text)""")
    if not kv_get(db, "recheck_similar"):        # one-off: look again at close calls under the "similar" rule
        db.execute("delete from seen where id in (select id from checked where verdict='' and visual>=0.5)")
        kv_set(db, "sweep_at", 0); kv_set(db, "recheck_similar", 1)
    if not kv_get(db, "requeue_v1"):             # one-off: close calls that never got an AI look
        db.execute("insert or ignore into later select id from checked where verdict='' and ai='' and visual>=?",
                   (cfg["ai_min_visual"],))
        kv_set(db, "requeue_v1", 1)
    HARD["t"] = time.time() + float(os.environ.get("RUN_MINUTES", 17)) * 60
    MYM["list"] = None
    if not kv_get(db, "fix_inner_v1"):           # one-off: drop flags raised only by Roblox's default file names
        db.execute("update checked set verdict='' where verdict like '%internal name%'")
        kv_set(db, "fix_inner_v1", 1)
    if not kv_get(db, "m3d_v2"):                 # new shape score: drop rows scored the old way
        db.execute("delete from mesh3d"); kv_set(db, "m3d_v2", 1)
    for k in STATS:
        STATS[k] = "" if k == "note" else 0
    ACTIVITY.clear(); PENDING.clear(); TIERS.clear()
    started = time.time()
    try:
        _cycle(db, cfg)
    finally:
        flush_alerts(cfg)
        write_report(db)
        report(db, cfg, started)
        digest(db, cfg)
        post_activity(cfg)


ACTIVITY = []
MYM = {"list": None}


def mesh_cards(m, colour=0x5865F2):
    """Two small cards, their 3D model and yours: bare mesh picture, texture, ids and scores."""
    asset = lambda i: f"https://create.roblox.com/store/asset/{i}"
    ids = lambda d: (f"Mesh [{d['mesh_id']}]({asset(d['mesh_id'])})" if d.get("mesh_id") else "Mesh not found") + \
                    (f" \u00b7 Texture [{d['tex_id']}]({asset(d['tex_id'])})" if d.get("tex_id") else " \u00b7 no texture")
    score = lambda v, f: "n/a" if v is None else f(v)
    t, y = m["theirs"], m["yours"]

    def extra(m):
        out = []
        if m.get("counts") and not m.get("same_mesh"):
            out.append("**Same vertex and face count as yours**")
        if m.get("inner"):
            out.append(f"**File still named \"{m['inner'][0][:50]}\", same as yours**")
        elif t.get("names"):
            out.append(f"Name inside their file: \"{t['names'][0][:50]}\"")
        if AI_MESH.search(t.get("mesh_name") or ""):
            out.append("Mesh name looks like an AI 3D-generator export")
        return out

    theirs = {"author": {"name": "Their 3D model"}, "color": colour,
              "title": (t.get("mesh_name") or "Mesh")[:200], "url": asset(t["mesh_id"]) if t.get("mesh_id") else None,
              "description": ids(t) + (f"\nMesh uploaded by **{t['mesh_by']}** {when(t['mesh_date'])}" if t.get("mesh_by") else "")
                             + "".join("\n" + x for x in extra(m)),
              "fields": [fld("Shape match", "same mesh" if m.get("same_mesh") else score(m["score"], lambda v: f"{v:.0%}")),
                         fld("Bare mesh alike", score(m["render"], lambda v: f"{pct(v):.0%}")),
                         fld("Texture alike", "same file" if m["same_tex"] else score(m["texsim"], lambda v: f"{v:.0%}"))]}
    yours = {"author": {"name": "Your 3D model"}, "color": 0x2B2D31,
             "title": m["mine_name"][:200], "url": f"https://www.roblox.com/catalog/{m['mine_id']}", "description": ids(y)}
    for card, d in ((theirs, t), (yours, y)):
        if d.get("render_url"):
            card["thumbnail"] = {"url": d["render_url"]}       # untextured mesh, small
        if d.get("tex_url"):
            card["author"]["icon_url"] = d["tex_url"]          # texture, as the little icon by the heading
        if not card.get("url"):
            card.pop("url", None)
    return [theirs, yours]


def post_activity(cfg):
    """Second webhook: one small card per item checked this run, closest matches first."""
    url = cfg.get("feed_webhook_url")
    if not url or not ACTIVITY:
        return
    ev = sorted(ACTIVITY, key=lambda e: (not e[4], -e[2]))
    cap = cfg.get("feed_max_cards", 120)
    cards, solo = [], []
    for src, c, v, mname, hit in ev[:cap]:
        link = f"https://www.roblox.com/catalog/{c['id']}"
        card = {"author": {"name": "UGC Watch"}, "title": c["name"][:250], "url": link,
                "color": 0xE67E22 if hit else 0x2F6FED,
                "description": f"{'New' if src == 'new' else 'Existing'} UGC {KIND.get(c.get('atype'), 'Accessory')}\n[Roblox Page]({link})"
                               + ("\n**Flagged, see the alerts channel**" if hit else ""),
                "fields": [fld("Price", cost(c)), fld("Creator", who(c)),
                           fld("Closest to yours", f"{pct(v):.0%} \u00b7 {mname[:40]}" if mname else "none"),
                           fld("Uploaded", when(c.get("created") or ""))],
                "footer": {"text": "Checked"}, "timestamp": stamp()}
        if c.get("thumb"):
            card["thumbnail"] = {"url": c["thumb"]}
        if c.get("m3d"):                         # 3D comparison: its own post, item card first
            solo.append([card] + mesh_cards(c["m3d"], card["color"]))
            continue
        cards.append(card)
    for group in solo:
        send(cfg, {"username": "UGC Watch", "embeds": group}, url)
        time.sleep(1.0)
    if len(ev) > cap:
        cards.append({"description": f"-# plus {len(ev) - cap} more that look nothing like your items", "color": 0x2B2D31})
    for k in range(0, len(cards), 10):          # Discord shows up to ten cards in one post
        send(cfg, {"username": "UGC Watch", "embeds": cards[k:k + 10]}, url)
        time.sleep(1.0)


def write_report(db):
    """REPORT.md: a plain list of what needs attention, what came close, and what was checked."""
    def rows(where, order, n):
        return db.execute("select id,name,creator,created,visual,mine_id,mine_name,verdict,ai "
                          f"from checked {where} order by {order} limit {n}").fetchall()

    def esc(s):
        return str(s).replace("|", "/").replace("\n", " ")[:60]

    def table(rs, result=True):
        out = ["| Their item | Uploaded by | Uploaded | Closest item of yours | Looks alike |" + (" Result |" if result else ""),
               "|---|---|---|---|---|" + ("---|" if result else "")]
        for r in rs:
            out.append(f"| [{esc(r[1])}](https://www.roblox.com/catalog/{r[0]}) | {esc(r[2])} | {r[3][:10]} | "
                       f"[{esc(r[6])}](https://www.roblox.com/catalog/{r[5]}) | {pct(r[4]):.0%} |"
                       + (f" {esc(label(r[7])) if r[7] else 'Not flagged'} |" if result else ""))
        return out if rs else ["Nothing here right now."]

    total = db.execute("select count(*) from seen").fetchone()[0]
    full = db.execute("select count(*) from checked").fetchone()[0]
    strong = "where verdict like 'Likely COPY%' or (verdict like 'Possible REMAKE%' and ai like 'remake%')"
    other = "where verdict<>'' and not (verdict like 'Likely COPY%' or (verdict like 'Possible REMAKE%' and ai like 'remake%'))"
    doc = ["# UGC Watch report", "",
           f"Last updated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}.", "",
           f"- **{total}** marketplace items looked at so far",
           f"- **{full}** were the same kind of item as yours and were compared closely",
           "- **Looks alike** is how closely the catalog pictures match, ignoring colour. Your own recolours score 85% or more and unrelated items score near 0%.", "",
           "## 1. Worth a takedown look", "", "Same or copied model according to the picture match and the AI review.", ""] \
        + table(rows(strong, "visual desc", 100)) + \
          ["", "## 2. Flagged, lower priority", "", "Similar look, similar name or copied listing text, but probably a different model.", ""] \
        + table(rows(other, "at desc", 150)) + \
          ["", "## 3. Closest items that were not flagged", "", "If one of these is clearly a copy, tell Claude so the limits can be adjusted.", ""] \
        + table(rows("where verdict='' and (ai like 'similar%' or ai like 'remake%' or ai like 'same%' or (ai='' and visual>=0.6))", "visual desc", 40), False) + \
          ["", "## 4. Latest items checked", ""] + table(rows("where ai not like 'unrelated%'", "at desc", 100))
    lib = lambda i: f"[{i}](https://create.roblox.com/store/asset/{i})" if i else "not found"
    num = lambda v, f: "n/a" if v is None else f(v)
    doc += ["", "## 5. 3D model and texture checks", "",
            "The real mesh and texture behind each close call, next to the item of yours whose real model is closest. "
            "Shape match compares the meshes lined up in the same pose: different models land below about 65%, "
            "copies 80% or more (also when resized, turned, mirrored or re-saved), 100% = the very same mesh. "
            "Mesh and texture ids open the asset pages.", "",
            "| Their item | Uploaded by | Your item | Their mesh / texture | Your mesh / texture | Shape match | Bare mesh alike | Texture alike | Their mesh file |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in db.execute("select * from mesh3d order by score desc, at desc limit 150"):
        doc.append(f"| [{esc(r[2])}](https://www.roblox.com/catalog/{r[0]}) | {esc(r[3])} | "
                   f"[{esc(r[5])}](https://www.roblox.com/catalog/{r[4]}) | {lib(r[6])} / {lib(r[7])} | {lib(r[8])} / {lib(r[9])} | "
                   f"{num(r[10], lambda v: f'{v:.0%}')} | {num(r[11], lambda v: f'{pct(v):.0%}')} | "
                   f"{num(r[12], lambda v: f'{v:.0%}')} | {esc(r[13])} |")
    (HERE / "REPORT.md").write_text("\n".join(doc) + "\n", encoding="utf-8")


def roblox_user(cfg):
    """Username the Roblox cookie is signed in as. None = no cookie set, '' = sign-in failed."""
    if not cfg.get("roblox_cookie"):
        return None
    try:
        r = sess.get("https://users.roblox.com/v1/users/authenticated",
                     headers={"Cookie": ".ROBLOSECURITY=" + cfg["roblox_cookie"]}, timeout=20)
        return r.json().get("name", "") if r.status_code == 200 else ""
    except Exception:
        return ""


def report(db, cfg, started):
    """Run summary: what was found, when it ran, when it runs next."""
    a = STATS["alerts"]
    if cfg["status_updates"] == "off" or (cfg["status_updates"] == "alerts_only" and not a):
        return
    now = time.time()
    starts = (kv_get(db, "starts", []) + [int(started)])[-7:]      # next run: the usual gap between recent runs
    kv_set(db, "starts", starts)
    gaps = sorted(b - a for a, b in zip(starts, starts[1:]) if b - a >= 20 * 60)
    nxt = int(started) + min(gaps[len(gaps) // 2] if gaps else 3600, 6 * 3600)
    n_mine = db.execute("select count(*) from mine").fetchone()[0]
    t = TIERS
    found = [f"**{t[k]}** {name}" for k, name in (("strong", "strong takedown case"), ("possible", "possible takedown case"),
             ("remake", "possible remake"), ("text", "copied listing text"), ("similar", "similar look and name")) if t.get(k)]
    lines = ["\n".join(found) if found else "No new look-alikes.", "",
             f"Fully checked **{STATS['checked']}** items against your **{n_mine}** in {max(1, round((now - started) / 60))} min",
             f"Checked <t:{int(now)}:f>", f"Next check around <t:{nxt}:f>"]
    sc = kv_get(db, "scan_stats", {})
    if sc.get("files"):
        queued = db.execute("select count(*) from deep").fetchone()[0] if db.execute(
            "select 1 from sqlite_master where name='deep'").fetchone() else 0
        lines += [f"Catalog scanner (<t:{int(sc['at'])}:R>, {sc['files']}/10 machines in): compared **{sc['compared']:,}** "
                  f"items by picture, **{sc['handed']}** look-alikes sent for the full check, **{queued}** still queued"]
    near = db.execute("select id,name,visual,creator from checked where at>=? and verdict='' and visual>=0.43 and (ai like 'similar%' or ai like 'remake%' or ai like 'same%' or (ai='' and visual>=0.6)) "
                      "order by visual desc limit 6", (started,)).fetchall()
    if near:
        lines += ["", "**Closest ones not flagged**"] + [
            f"`{pct(r[2]):>4.0%}` [{r[1][:50]}](https://www.roblox.com/catalog/{r[0]}) \u00b7 {r[3][:22]}" for r in near]
    sp = kv_get(db, "ai_spend", {})
    if sp.get("total"):
        lines += ["", f"AI spend: ${sp.get(time.strftime('%Y-%m-%d'), 0):.3f} today, ${sp['total']:.2f} in total"]
    waiting = db.execute("select count(*) from later").fetchone()[0]
    if waiting:
        lines.append(f"Waiting for an AI review: **{waiting}** items, checked as the daily budget allows")
    user = roblox_user(cfg)
    lines.append("3D model check: off, no Roblox cookie set" if user is None else
                 f"3D model check: signed in to Roblox as **{user}**, {STATS['mesh']} models compared"
                 + (f", {STATS['mesh_fail']} could not be fetched" if STATS["mesh_fail"] else "") if user else
                 "3D model check: Roblox sign-in FAILED, the cookie has expired or is wrong")
    if STATS["note"]:
        lines.append("AI problem: " + STATS["note"])
    repo = os.environ.get("DATA_REPO") or os.environ.get("GITHUB_REPOSITORY")   # the report lives with the data
    if repo:
        lines.append(f"[Full report](https://github.com/{repo}/blob/main/REPORT.md)")
    title = f"{a} new look-alike{'s' if a != 1 else ''} found" if a else "All clear"
    send(cfg, {"username": "UGC Watch", "embeds": [{"title": title, "color": 0xE67E22 if a else 0x2ECC71,
                                                     "description": "\n".join(lines), "timestamp": stamp()}]})


def digest(db, cfg):
    """Once a week: the takedown-worthy cases from the last seven days in one message."""
    last = kv_get(db, "digest_at", 0)
    if not last:
        kv_set(db, "digest_at", time.time()); return
    if time.time() - last < 7 * 86400:
        return
    rs = db.execute("select id,name,creator,visual,mine_name,verdict from checked where at>=? and (verdict like 'Likely COPY%' "
                    "or (verdict like 'Possible REMAKE%' and ai like 'remake%')) order by visual desc limit 25", (last,)).fetchall()
    body = "\n".join(f"[{r[1][:60]}](https://www.roblox.com/catalog/{r[0]}) by {r[2][:30]} \u00b7 {pct(r[3]):.0%} like your {r[4][:40]}"
                     for r in rs) or "No takedown-worthy cases this week."
    send(cfg, {"username": "UGC Watch", "embeds": [{"title": f"Weekly digest: {len(rs)} case{'s' if len(rs) != 1 else ''} worth a takedown look",
                                                     "color": 0x9B59B6, "description": body[:4000],
                                                     "footer": {"text": "Week ending"}, "timestamp": stamp()}]})
    kv_set(db, "digest_at", time.time())


def selftest(db, cfg):
    """Scores each of your items against all your others, to show how well recolours are recognised."""
    if not load_mine(db):
        refresh_mine(db, cfg)
    mine = [m for m in load_mine(db) if m["feat"] is not None]
    fam = lambda m: " ".join(tokens(m["name"]))
    pairs = [(visual_score(a["feat"], b["feat"])[0], a, b) for k, a in enumerate(mine) for b in mine[k + 1:]]
    print(f"{len(mine)} items, {len(pairs)} pairs")
    print("NEAREST OTHER ITEM OF YOURS, PER ITEM")
    for a in mine:
        s, b = max(((s, y if x is a else x) for s, x, y in pairs if x is a or y is a), key=lambda t: t[0])
        print(f"{s:.2f}  {a['name']} ({a['created'][:10]})  <->  {b['name']} ({b['created'][:10]})")
    cross = sorted((p for p in pairs if fam(p[1]) != fam(p[2])), key=lambda p: -p[0])
    print("HIGHEST SCORES BETWEEN DIFFERENTLY NAMED ITEMS")
    for s, a, b in cross[:12]:
        print(f"{s:.2f}  {a['name']}  <->  {b['name']}")
    allp = sorted(p[0] for p in pairs)
    print("ALL PAIRS: " + " ".join(f"{q}%={allp[int(len(allp) * q / 100) - (q == 100)]:.2f}" for q in (10, 25, 50, 75, 90, 95, 100)))


# ------------------------------------------------------- catalog scanner ---
# Roblox stops every catalog search at 1,000 results. The scanner splits the catalog into many small searches
# (item type x price band x sort order, and item type x price band x each word from your item names) that each
# stay under that cap, compares everything it finds with your items by picture, and hands only the look-alikes
# to the private watcher, which runs the AI and 3D checks on them.
TAXONOMY = {8: "thukPQUFe5EHxRrtsgq43u", 43: "tvckLAbtDAGwF9Y2sph2Ce", 44: "8i1sggNTrxoN799hpRrR1S",
            45: "kj535RzJxHTCnXPBLo5KaE", 46: "7yGywzp5JeDsNjhXWJqSw3", 47: "6jYQPDtkVwQZcKtvVpyYAN"}
TYPE_WORDS = {                                   # what copies of each kind are usually called, whatever your names say
    8: "horns horn hat crown halo ears demon devil dragon helmet hood mask bow".split(),
    46: "wings wing sword cape bag backpack angel demon dragon chain cross bow".split(),
    47: "tail tails fluffy fox wolf cat kitsune bunny dragon demon devil furry plush chain belt bow".split(),
    45: "fan sword chain plush bag".split(), 43: "chain choker necklace collar".split(), 44: "plush pet".split()}
PRICE_BANDS = [(0, 64), (65, 65), (66, 75), (76, 90), (91, 110), (111, 150), (151, 250), (251, 600), (601, 10 ** 7)]


def scan_slices(mine):
    by_type = {}
    for m in mine:
        by_type.setdefault(m["atype"], []).append(m)
    out = []
    for t, items in sorted(by_type.items()):
        tax = TAXONOMY.get(t)
        if not tax:
            continue
        for lo, hi in PRICE_BANDS:
            for sort in (3, 2, 1):               # newest, best selling, most favourited
                out.append({"taxonomy": tax, "SortType": sort, "MinPrice": lo, "MaxPrice": hi})
        words = {w for m in items for w in tokens(m["name"]) if len(w) >= 3} | set(TYPE_WORDS.get(t, ()))
        for w in sorted(words):
            for lo, hi in PRICE_BANDS:
                out.append({"taxonomy": tax, "Keyword": w, "SortType": 0, "MinPrice": lo, "MaxPrice": hi})
    return out


def scan_shard(db, cfg, shard, total, minutes=24):
    """One of several parallel scanner machines. Writes its look-alikes to scan/<run>-<shard>.json.gz."""
    QUIET["on"] = True
    mine = [m for m in load_mine(db) if m["feat"] is not None]
    if not mine:
        log("no items of yours in the data yet"); return
    stop = time.time() + minutes * 60
    HARD["t"] = stop + 6 * 60
    slices = scan_slices(mine)[shard::total]
    skip = set(cfg["ignore_creator_ids"]) | {cfg["my_creator_id"]}
    found, done = {}, 0
    for sl in slices:
        if time.time() > stop:
            break
        for page in search_wide(CATALOG.replace("/v1/", "/v2/"), 9, salesTypeFilter=1, **sl):
            for d in page:
                if d.get("itemType") == "Asset" and d.get("creatorTargetId") not in skip:
                    found.setdefault(d["id"], d)
        done += 1
    fresh = [d for i, d in found.items() if not is_seen(db, i)]
    log(f"searched {done} of {len(slices)} slices, {len(found)} items, {len(fresh)} not checked before")
    keep, scored = [], 0
    for k in range(0, len(fresh), 200):          # thumbnails in batches, so memory stays small
        if time.time() > HARD["t"]:
            break
        chunk = {d["id"]: d for d in fresh[k:k + 200]}
        urls, feats = get_thumbs(chunk)
        for i, d in chunk.items():
            u = urls.get(i)
            if not (isinstance(u, str) and u.startswith("http")) or not feats.get(u) or feats[u][0] is None:
                continue
            cand = to_cand(d); cand["thumb"] = u
            res = judge(cand, feats[u][0], mine, cfg)
            scored += 1
            if res and (res["verdict"] or res["visual"] >= cfg.get("scan_min_visual", 0.5) or res["name"] >= 0.6
                        or res["desc"] >= 0.3 or res["credit"]):
                keep.append((max(res["visual"], res["name"] * 0.9, res["desc"]), d))
    keep = [d | {"_score": round(s, 3)} for s, d in sorted(keep, key=lambda x: -x[0])[:cfg.get("scan_max_per_shard", 400)]]
    out = HERE / "scan"
    out.mkdir(exist_ok=True)
    name = f"{os.environ.get('GITHUB_RUN_ID', int(time.time()))}-{shard}.json.gz"
    stats = {"slices": done, "of": len(slices), "found": len(found), "compared": scored, "handed": len(keep)}
    (out / name).write_bytes(gzip.compress(json.dumps({"stats": stats, "items": keep}).encode()))
    log(f"compared {scored} items by picture, {len(keep)} look-alikes handed to the watcher")


def ingest_scans(db):
    """Queue the scanner's look-alikes; remember which files were read so the save step can remove them."""
    db.execute("create table if not exists deep(id integer primary key, score real, raw text)")
    read, n = [], 0
    for f in sorted((HERE / "scan").glob("*.json.gz")) if (HERE / "scan").exists() else []:
        try:
            blob = json.loads(gzip.decompress(f.read_bytes()))
            items = blob["items"] if isinstance(blob, dict) else blob
            st = blob.get("stats", {}) if isinstance(blob, dict) else {"handed": len(blob)}
            tot = kv_get(db, "scan_stats", {})
            run = f.name.split("-")[0]
            if tot.get("run") != run:            # a new scan: start counting again
                tot = {"run": run, "at": time.time(), "found": 0, "compared": 0, "handed": 0, "files": 0, "seen": []}
            if f.name not in tot.setdefault("seen", []):         # a file is only counted once
                for k in ("found", "compared", "handed"):
                    tot[k] = tot.get(k, 0) + st.get(k, 0)
                tot["files"] = tot.get("files", 0) + 1
                tot["seen"].append(f.name)
            kv_set(db, "scan_stats", tot)
            for d in items:
                if not is_seen(db, d["id"]):
                    db.execute("insert or ignore into deep values(?,?,?)", (d["id"], d.get("_score", 0), json.dumps(d)))
                    n += 1
        except Exception as e:
            log("  scan file unreadable:", f.name, repr(e)[:80])
        read.append(f"scan/{f.name}")
    db.commit()
    if read:
        (HERE / "ingested.txt").write_text("\n".join(read) + "\n")
        log(f"catalog scanner: {n} new look-alikes from {len(read)} files")


def _cycle(db, cfg):
    if time.time() - (kv_get(db, "mine_at", 0)) > cfg["refresh_my_items_hours"] * 3600 or not kv_get(db, "store_v2"):
        refresh_mine(db, cfg); kv_set(db, "store_v2", 1)
    mine = load_mine(db)
    if not mine:
        log("no items of yours indexed yet; check my_creator_id"); return
    types = {m["atype"] for m in mine} | set(cfg["extra_asset_types"])
    budget = cfg["max_alerts_per_cycle"]
    deadline = time.time() + cfg["max_run_minutes"] * 60

    if ai_ready(cfg, db):                        # items that were waiting for an AI look
        ids = [r[0] for r in db.execute("select id from later limit 80")]
        if ids:
            q = ",".join(map(str, ids))
            db.execute(f"delete from later where id in ({q})"); db.execute(f"delete from seen where id in ({q})")
            db.commit()
            budget -= process(db, cfg, mine, details_batch(ids), ACCESSORY, budget, "scan")
            for i in ids:                        # anything not reached stays in the queue
                if not is_seen(db, i):
                    db.execute("insert or ignore into later values(?)", (i,)); mark(db, i)
            db.commit()
            log(f"looked again at {len(ids)} items that were waiting for an AI review")

    for sub in sorted({SUBCAT.get(t, 0) for t in types}):   # live feed: recently published
        params = {"Subcategory": sub} if sub else {}       # 0 = all accessories (types with no known subcategory)
        got = stale = 0
        for page in search_wide(CATALOG.replace("/details", ""), cfg.get("feed_wide_pages", 4) * (1 if sub else 4),
                                Category=11, SortType=3, salesTypeFilter=1, **params):
            new = [d["id"] for d in page if d.get("itemType") == "Asset" and not is_seen(db, d["id"])]
            got += len(new)
            items = details_batch(new)
            for i in set(new) - {d["id"] for d in items}:
                mark(db, i)                      # removed or hidden since it was listed
            budget -= process(db, cfg, mine, items, types, budget)
            stale = 0 if new else stale + 1
            if stale >= 2:                       # two whole pages already seen: caught up
                break
        log(f"feed {'subcategory %d' % sub if sub else 'all accessories'}: {got} new items")

    ingest_scans(db)
    rows = db.execute("select id, raw from deep order by score desc limit ?", (int(os.environ.get("DEEP_PER_RUN", 150)),)).fetchall()
    if rows and time.time() < deadline:          # look-alikes found by the catalog scanner
        ids = [r[0] for r in rows]
        db.execute(f"delete from deep where id in ({','.join(map(str, ids))})"); db.commit()
        budget -= process(db, cfg, mine, [json.loads(r[1]) for r in rows], ACCESSORY | types, budget, "scan")
        back = [r for r in rows if not is_seen(db, r[0])]
        for i, raw in back:                      # not reached this run: keep them queued
            db.execute("insert or ignore into deep values(?,?,?)", (i, json.loads(raw).get("_score", 0), raw))
        db.commit()
        log(f"catalog scanner: checked {len(rows) - len(back)} look-alikes, "
            f"{db.execute('select count(*) from deep').fetchone()[0]} still queued")

    if time.time() - kv_get(db, "sweep_at", 0) > cfg["sweep_hours"] * 3600:
        kws = list(keywords(mine).items())       # existing items named like yours, most relevant first
        pos = kv_get(db, "sweep_pos", 0)
        while pos < len(kws):
            if time.time() > deadline:           # long scans are spread over several runs
                log(f"out of time, scan continues next run ({pos}/{len(kws)} search terms done)")
                kv_set(db, "sweep_pos", pos); return
            kw, since = kws[pos]
            n = 0
            for page in search_wide(CATALOG.replace("/v1/", "/v2/"), cfg.get("sweep_wide_pages", 5),
                                    Keyword=kw, Category=11, SortType=0, salesTypeFilter=1):
                page = [d for d in page if (d.get("itemCreatedUtc") or "9") >= since]
                n += len(page)
                budget -= process(db, cfg, mine, page, types, budget, "scan")
            log(f"  scan '{kw}': {n} items uploaded since {since[:10]}")
            pos += 1; kv_set(db, "sweep_pos", pos)
        kv_set(db, "sweep_pos", 0); kv_set(db, "sweep_at", time.time())

    if time.time() - kv_get(db, "stores_at", 0) > 24 * 3600 and time.time() < deadline:
        # once a day: everything sold by creators already flagged, whatever its name or accessory type
        names = [r[0] for r in db.execute("select creator, max(visual) v from checked where verdict<>'' "
                                          "group by creator order by v desc limit 15")]
        for nm in names:
            for page in search_wide(CATALOG.replace("/details", ""), 3, Category=11, SortType=3, CreatorName=nm):
                new = [d["id"] for d in page if d.get("itemType") == "Asset" and not is_seen(db, d["id"])]
                items = details_batch(new)
                for i in set(new) - {d["id"] for d in items}:
                    mark(db, i)
                budget -= process(db, cfg, mine, items, {8, 41, 42, 43, 44, 45, 46, 47}, budget, "scan")
        log(f"checked the stores of {len(names)} flagged creators")
        kv_set(db, "stores_at", time.time())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true"); ap.add_argument("--test", action="store_true")
    ap.add_argument("--check", type=int); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", help="folder holding config.json and the database (default: next to this file)")
    ap.add_argument("--shard", help="catalog scanner part, e.g. 3/10")
    a = ap.parse_args()
    if a.data:
        globals()["HERE"] = pathlib.Path(a.data).resolve()
    if os.environ.get("PUBLIC_LOG"):             # running in the public repo: counts only in the log
        QUIET["on"] = True
    cfg = dict(DEFAULTS)
    cp = HERE / "config.json"
    if cp.exists():
        cfg.update(json.loads(cp.read_text(encoding="utf-8")))
    for key, env in (("discord_webhook_url", "DISCORD_WEBHOOK_URL"), ("anthropic_api_key", "ANTHROPIC_API_KEY"),
                     ("gemini_api_key", "GEMINI_API_KEY"), ("feed_webhook_url", "FEED_WEBHOOK_URL"), ("roblox_cookie", "ROBLOX_COOKIE")):
        if os.environ.get(env):                  # cloud runs: secrets come from the environment
            cfg[key] = os.environ[env].strip()
    if cfg["discord_webhook_url"].startswith("PASTE"):
        cfg["discord_webhook_url"] = ""
    if not cfg["my_creator_id"]:
        sys.exit("Set my_creator_id in config.json")
    db = db_open()
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        scan_shard(db, cfg, i, n); return
    if a.test:
        ok = send(cfg, {"username": "UGC Watch", "content": "UGC Watch is connected."})
        sys.exit(0 if ok else "Webhook test failed")
    if a.selftest:
        selftest(db, cfg); return
    if a.check:
        if not load_mine(db):
            refresh_mine(db, cfg)
        mine = load_mine(db)
        j = http_json(f"https://economy.roblox.com/v2/assets/{a.check}/details") or {}
        cand = {"id": a.check, "name": j.get("Name", ""), "descr": j.get("Description", ""),
                "created": j.get("Created", ""), "atype": j.get("AssetTypeId")}
        u = thumb_urls([a.check]).get(a.check)
        feat, raw_png = fetch_feat(u) if isinstance(u, str) and u.startswith("http") else (None, None)
        res = judge(cand, feat, mine, cfg)
        if not res:
            sys.exit("nothing of yours is older than that item")
        real = dict(res)
        vm = res["visual_match"]
        if vm and ai_provider(cfg):
            ai = ai_compare(cfg, db, fetch_feat(vm["thumb"])[1], raw_png, vm["name"], cand["name"])
            print("AI:", ai or "no answer")
            real = apply_ai(real, ai, cfg)
        res["match"] = vm or res["match"]
        print(f"{cand['name']}\n  closest of yours: {res['match']['name']} ({res['match']['id']})\n"
              f"  visual {res['visual']:.2f} (shape {res['shape']:.2f}, detail {res['detail']:.2f}, "
              f"mirrored {res['mirrored']})\n  name {res['name']:.2f}  description {res['desc']:.2f}  "
              f"credit {res['credit']}\n  verdict: {real['verdict'] or 'no alert'}")
        return
    while True:
        if os.environ.get("GITHUB_EVENT_NAME") == "schedule" and time.time() - kv_get(db, "last_run", 0) < 40 * 60:
            log("ran less than 40 minutes ago, skipping this scheduled start"); return
        kv_set(db, "last_run", time.time())
        try:
            cycle(db, cfg)
        except Exception as e:                   # keep the watcher alive
            log("cycle failed:", repr(e))
        if a.once:
            break
        time.sleep(cfg["poll_minutes"] * 60)


if __name__ == "__main__":
    main()
