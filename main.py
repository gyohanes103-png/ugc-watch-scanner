import argparse, base64, calendar, difflib, gzip, hashlib, io, itertools, json, os, pathlib, re, sqlite3, struct, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
from PIL import Image, ImageFilter, ImageOps
HERE = pathlib.Path(__file__).resolve().parent
CATALOG = 'https://catalog.roblox.com/v1/search/items/details'
THUMBS = 'https://thumbnails.roblox.com/v1/assets'
SUBCAT = {46: 25, 47: 26}
COLOURS = set('black white grey gray red blue pink purple violet green yellow orange brown\nlight dark pastel neon gold golden silver cyan teal lavender beige cream blonde ginger\nombre rainbow and'.split())
STOP = set('the a an of for with to in on my your our is it its this that very'.split())
DEFAULTS = {'discord_webhook_url': '', 'my_creator_id': 0, 'my_creator_type': 'Group', 'ignore_creator_ids': [], 'extra_asset_types': [], 'poll_minutes': 5, 'feed_pages': 6, 'sweep_hours': 12, 'sweep_pages': 15, 'refresh_my_items_hours': 6, 'copy_threshold': 0.8, 'remake_threshold': 0.63, 'combo_threshold': 0.5, 'ai_provider': 'auto', 'anthropic_api_key': '', 'gemini_api_key': '', 'anthropic_workspace_id': '', 'anthropic_model': 'claude-haiku-5-5', 'gemini_model': 'gemini-3.5-flash', 'ai_min_visual': 0.42, 'scan_min_visual': 0.42, 'ai_max_calls_per_day': 10 ** 9, 'text_alerts': True, 'max_alerts_per_cycle': 10 ** 9, 'status_updates': 'every_run', 'max_run_minutes': 12}
sess = requests.Session()
sess.headers['User-Agent'] = 'Mozilla/5.0 (ugc-watch; personal IP monitoring)'
AI_DOWN = set()
STATS = {'checked': 0, 'alerts': 0, 'ai_ok': 0, 'ai_failed': 0, 'waited': 0.0, 'note': '', 'mesh': 0, 'mesh_fail': 0, 'live': 0, 'limited': 0}
QUIET = {'on': False}

def log(*a):
    if QUIET['on']:
        if a and str(a[0]).startswith('  '):
            return
        msg = ' '.join(map(str, a))
        code = hashlib.md5(re.sub('\\d+', '#', msg).encode()).hexdigest()[:4]
        print(time.strftime('%H:%M:%S'), code, *re.findall('\\d+', msg), flush=True)
        return
    print(time.strftime('%H:%M:%S'), *a, flush=True)

class Limiter:
    import threading as _th

    def __init__(self):
        self.st, self.lock = ({}, self._th.Lock())

    def _s(self, host):
        return self.st.setdefault(host, {'budget': 60.0, 'cool': 5.0, 'lo': 0.0, 'hi': 0.0, 'times': [], 'oks': [], 'until': 0.0, 'streak': 0.0, 'waited': 0.0, 'quiet': time.time(), 'busy': 0})

    def before(self, host, max_block=float('inf')):
        with self.lock:
            st, now = (self._s(host), time.time())
            st['times'] = [t for t in st['times'] if now - t < 60]
            wait = max(st['until'] - now, st['times'][0] + 60 - now if len(st['times']) >= st['budget'] else 0)
            if wait > max_block:
                return False
            if not st['streak'] and now - st['quiet'] > 120:
                st['budget'], st['quiet'] = (min(1200.0, st['budget'] * 1.15), now)
        if wait > 0:
            STATS['waited'] += wait
            time.sleep(wait)
        with self.lock:
            self._s(host)['times'].append(time.time())
        return True

    def busy(self, host, retry_after=None):
        with self.lock:
            st, now = (self._s(host), time.time())
            st['busy'] += 1
            STATS['limited'] = STATS.get('limited', 0) + 1
            st['times'] = st['times'][:-1]
            if st['streak']:
                st['lo'] = max(st['lo'], now - st['streak'])
            else:
                ok = len([t for t in st['oks'] if now - t < 60])
                st['budget'] = max(3.0, ok * 0.95)
            exact = None
            try:
                exact = float(retry_after) if retry_after else None
            except ValueError:
                pass
            if exact is not None:
                st['cool'] = st['hi'] = exact
                wait = exact
            elif st['hi'] and st['hi'] - st['lo'] > 0.2:
                wait = (st['lo'] + st['hi']) / 2
            elif st['hi']:
                wait = st['hi']
            else:
                wait = max(st['cool'], st['lo'] * 1.6 or st['cool'])
            start = st['streak'] or now
            st['waited'], st['until'] = (wait, start + wait)
            st['streak'], st['quiet'] = (start, now)

    def ok(self, host):
        with self.lock:
            st = self._s(host)
            st['oks'] = [t for t in st['oks'] if time.time() - t < 60] + [time.time()]
            if st['streak']:
                took = time.time() - st['streak']
                st['hi'] = min(st['hi'] or took, took)
                st['lo'] = min(st['lo'], st['hi'])
                st['cool'] = st['hi']
                st['streak'] = 0.0

    def export(self):
        return {h: {'budget': round(v['budget'], 1), 'cool': round(v['cool'], 2), 'busy': v['busy']} for h, v in self.st.items()}

    def load(self, saved):
        for h, v in (saved or {}).items():
            st = self._s(h)
            st['budget'], st['cool'] = (float(v.get('budget', 60)), float(v.get('cool', 5)))
            st['hi'] = st['cool'] if v.get('busy') else 0.0
LIMIT = Limiter()

def _host(url):
    return url.split('/')[2] if '//' in url else url

def http_json(url, params=None, tries=6, max_wait=90):
    host = _host(url)
    for _ in range(tries):
        if not LIMIT.before(host, max_wait):
            return None
        try:
            r = sess.get(url, params=params, timeout=25)
        except requests.RequestException as e:
            log('  network error:', e)
            time.sleep(2)
            continue
        if r.status_code == 200:
            LIMIT.ok(host)
            return r.json()
        if r.status_code == 429:
            LIMIT.busy(host, r.headers.get('Retry-After'))
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(2)
            continue
        log(f'  HTTP {r.status_code} from {url}: {r.text[:120]}')
        return None
    return None
N = 64

def _mask_rgb(im):
    a = np.asarray(im.convert('RGBA'), dtype=np.float32)
    rgb, alpha = (a[..., :3], a[..., 3])
    if alpha.min() < 200:
        return (alpha > 60, rgb)
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(edge, axis=0)
    return (np.abs(rgb - bg).sum(-1) > 40, rgb)

def features(img_bytes):
    im = Image.open(io.BytesIO(img_bytes))
    mask, rgb = _mask_rgb(im)
    if mask.sum() < 200:
        return None
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    h, w = (y1 - y0, x1 - x0)
    side = max(h, w)
    grey = rgb[y0:y1, x0:x1].mean(-1)
    m = mask[y0:y1, x0:x1]
    mu, sd = (grey[m].mean(), grey[m].std() + 1e-06)
    g = np.where(m, (grey - mu) / sd, 0.0)
    G = np.zeros((side, side), np.float32)
    M = np.zeros((side, side), np.float32)
    oy, ox = ((side - h) // 2, (side - w) // 2)
    G[oy:oy + h, ox:ox + w] = g
    M[oy:oy + h, ox:ox + w] = m
    G = np.asarray(Image.fromarray(G).resize((N, N), Image.BILINEAR))
    M = np.asarray(Image.fromarray(M).resize((N, N), Image.BILINEAR)) > 0.5
    return {'mask': M, 'grey': G * M, 'aspect': w / h, 'colour': [float(c) for c in rgb[mask].mean(0)]}

def _hog(g, m):
    gy, gx = np.gradient(g)
    inner = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    mag = np.hypot(gx, gy) * inner
    ang = np.arctan2(gy, gx) % np.pi / np.pi * 9
    b = np.minimum(ang.astype(int), 8)
    out = np.zeros((8, 8, 9), np.float32)
    cy, cx = np.mgrid[0:N, 0:N] // 8
    np.add.at(out, (cy, cx, b), mag)
    out /= np.linalg.norm(out, axis=2, keepdims=True) + 1e-06
    v = out.ravel()
    return v / (np.linalg.norm(v) + 1e-06)

def _sdf_like(m):
    f = m.astype(np.float32)
    for _ in range(3):
        f = (f + np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 5
    return f

def finish(feat):
    feat['hog'] = _hog(feat['grey'], feat['mask'])
    feat['hog_m'] = _hog(feat['grey'][:, ::-1], feat['mask'][:, ::-1])
    feat['soft'] = _sdf_like(feat['mask'])
    return feat

def _variants(a):
    if 'var' not in a:
        a['var'] = []
        for r in range(4):
            for flip in (False, True):
                g, m = (np.rot90(a['grey'], r), np.rot90(a['mask'], r))
                if flip:
                    g, m = (g[:, ::-1], m[:, ::-1])
                g, m = (np.ascontiguousarray(g), np.ascontiguousarray(m))
                a['var'].append((_sdf_like(m), _hog(g, m), a['aspect'] if r % 2 == 0 else 1 / a['aspect'], 1.0 if r == 0 else 0.92, flip or r > 0))
    return a['var']

def _turned(grey, mask, deg):
    m = np.asarray(Image.fromarray(mask.astype(np.float32)).rotate(deg, Image.BILINEAR, expand=True)) > 0.5
    g = np.asarray(Image.fromarray(grey.astype(np.float32)).rotate(deg, Image.BILINEAR, expand=True))
    if m.sum() < 20:
        return None
    ys, xs = np.where(m)
    y0, y1, x0, x1 = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    h, w = (y1 - y0, x1 - x0)
    side = max(h, w)
    G = np.zeros((side, side), np.float32)
    M = np.zeros((side, side), np.float32)
    oy, ox = ((side - h) // 2, (side - w) // 2)
    G[oy:oy + h, ox:ox + w] = g[y0:y1, x0:x1] * m[y0:y1, x0:x1]
    M[oy:oy + h, ox:ox + w] = m[y0:y1, x0:x1]
    G = np.asarray(Image.fromarray(G).resize((N, N), Image.BILINEAR))
    M = np.asarray(Image.fromarray(M).resize((N, N), Image.BILINEAR)) > 0.5
    return (G * M, M, w / h)

def _turned_variants(a):
    if 'tvar' not in a:
        a['tvar'] = []
        for deg in (-16, -8, 8, 16):
            for flip in (False, True):
                g0, m0 = (a['grey'][:, ::-1], a['mask'][:, ::-1]) if flip else (a['grey'], a['mask'])
                t = _turned(g0, m0, deg)
                if t:
                    g, m, asp = t
                    a['tvar'].append((_sdf_like(m), _hog(g, m), asp, 0.9, True))
    return a['tvar']

def _plain(a, b):
    soft, hog, aspect = _variants(a)[0][:3]
    shape = float(np.minimum(soft, b['soft']).sum() / (np.maximum(soft, b['soft']).sum() + 1e-06))
    return shape * (0.5 + 0.5 * min(aspect, b['aspect']) / max(aspect, b['aspect']))

def visual_score(a, b):
    best = (0.0, 0.0, 0.0, False)
    for soft, hog, aspect, weight, changed in _variants(a) + (_turned_variants(a) if _plain(a, b) >= 0.25 else []):
        shape = float(np.minimum(soft, b['soft']).sum() / (np.maximum(soft, b['soft']).sum() + 1e-06))
        detail = float(np.dot(hog, b['hog']))
        shape *= 0.5 + 0.5 * min(aspect, b['aspect']) / max(aspect, b['aspect'])
        score = weight * shape * (0.55 + 0.45 * max(detail, 0.0))
        if score > best[0]:
            best = (score, shape, detail, changed)
    return best

def tokens(s, drop_colours=True):
    t = re.findall('[a-z0-9]+', (s or '').lower())
    return [x for x in t if x not in STOP and (not (drop_colours and x in COLOURS))]

def name_score(a, b):
    ta, tb = (tokens(a), tokens(b))
    if not ta or not tb:
        return 0.0
    jac = len(set(ta) & set(tb)) / len(set(ta) | set(tb))
    return max(jac, difflib.SequenceMatcher(None, ' '.join(ta), ' '.join(tb)).ratio())

def desc_score(mine, theirs):
    a, b = (tokens(mine, False), tokens(theirs, False))
    if len(a) < 4 or len(b) < 4:
        return 0.0
    if len(a) < 6:
        return 1.0 if f" {' '.join(a)} " in f" {' '.join(b)} " else 0.0
    grams = lambda t: {tuple(t[k:k + 3]) for k in range(len(t) - 2)}
    ga = grams(a)
    return min(1.0, len(ga & grams(b)) / len(ga))

def handles(s):
    return set(re.findall('@\\w{3,}', s or ''))

def db_open():
    db = sqlite3.connect(HERE / 'ugc_watch.db')
    db.executescript('\n      create table if not exists seen(id integer primary key, at real);\n      create table if not exists mine(id integer primary key, name text, descr text,\n                                      created text, atype int, thumb text, feat blob);\n      create table if not exists kv(k text primary key, v text);\n      create table if not exists later(id integer primary key);\n      create table if not exists mesh3d(id integer primary key, at real, name text, creator text, mine_id int,\n        mine_name text, mesh_id int, tex_id int, my_mesh_id int, my_tex_id int, score real, render real,\n        texsim real, info text);')
    return db

def kv_get(db, k, d=None):
    r = db.execute('select v from kv where k=?', (k,)).fetchone()
    return json.loads(r[0]) if r else d

def kv_set(db, k, v):
    db.execute('insert or replace into kv values(?,?)', (k, json.dumps(v)))
    db.commit()

def pack(f):
    buf = io.BytesIO()
    np.savez_compressed(buf, mask=f['mask'], grey=f['grey'], aspect=f['aspect'], colour=f['colour'])
    return buf.getvalue()

def unpack(b):
    z = np.load(io.BytesIO(b))
    return finish({'mask': z['mask'], 'grey': z['grey'], 'aspect': float(z['aspect']), 'colour': z['colour'].tolist()})

def search(pages, **params):
    cursor = None
    for _ in range(pages):
        p = {'Category': 11, 'Limit': 30, 'SortType': 3, **params}
        if cursor:
            p['Cursor'] = cursor
        j = http_json(CATALOG, p)
        time.sleep(1.2)
        if not j or not j.get('data'):
            return
        yield j['data']
        cursor = j.get('nextPageCursor')
        if not cursor:
            return

def thumb_urls(ids):
    out = {}
    ids = list(ids)
    for i in range(0, len(ids), 100):
        j = http_json(THUMBS, {'assetIds': ','.join(map(str, ids[i:i + 100])), 'size': '420x420', 'format': 'Png', 'isCircular': 'false'})
        for d in (j or {}).get('data', []):
            out[d['targetId']] = d.get('imageUrl') if d.get('state') == 'Completed' else d.get('state')
    return out

def fetch_feat(url):
    try:
        r = sess.get(url, timeout=25)
        if r.status_code != 200:
            return (None, None)
        f = features(r.content)
        return (finish(f) if f else None, r.content)
    except Exception as e:
        log('  thumbnail failed:', e)
        return (None, None)

def refresh_mine(db, cfg):
    log('refreshing your items...')
    items = []
    ids = []
    for page in search_wide(CATALOG.replace('/details', ''), 40, Category=1, SortType=3, CreatorTargetId=cfg['my_creator_id'], CreatorType=cfg['my_creator_type'], IncludeNotForSale='true'):
        ids += [d['id'] for d in page if d.get('itemType') == 'Asset']
    listed = len(ids)
    items = [d for d in details_batch(ids) if d.get('assetType') in ACCESSORY and 'DO NOT ONSALE' not in (d.get('name') or '').upper()]
    before = db.execute('select count(*) from mine').fetchone()[0]
    if not items or len(items) < before * 0.8:
        log(f'  could not list your whole store ({listed} listed, {len(items)} usable, {before} known); keeping the known ones')
        return
    have = {r[0] for r in db.execute('select id from mine where feat is not null')}
    urls = thumb_urls([d['id'] for d in items if d['id'] not in have])
    for d in items:
        feat = None
        if d['id'] not in have:
            u = urls.get(d['id'])
            f = fetch_feat(u)[0] if isinstance(u, str) and u.startswith('http') else None
            feat = pack(f) if f else None
            db.execute('insert or replace into mine values(?,?,?,?,?,?,?)', (d['id'], d['name'], d.get('description') or '', d.get('itemCreatedUtc') or '', d.get('assetType'), u if isinstance(u, str) else '', feat))
        else:
            db.execute('update mine set name=?, descr=? where id=?', (d['name'], d.get('description') or '', d['id']))
    db.execute(f"delete from mine where id not in ({','.join((str(d['id']) for d in items))})")
    db.commit()
    kv_set(db, 'mine_at', time.time())
    log(f'  {len(items)} of your items indexed ({listed} listed in your store)')
    if before and len(items) >= before + 5:
        db.execute("delete from seen where id not in (select id from checked where verdict<>'')")
        db.commit()
        for k in ('sweep_at', 'sweep_pos', 'stores_at'):
            kv_set(db, k, 0)
        log('  your item list grew, so earlier items will be compared again')

def load_mine(db):
    rows = db.execute('select id,name,descr,created,atype,thumb,feat from mine').fetchall()
    return [{'id': r[0], 'name': r[1], 'descr': r[2], 'created': r[3], 'atype': r[4], 'thumb': r[5], 'feat': unpack(r[6]) if r[6] else None} for r in rows]

def keywords(mine):
    out = {}
    for m in mine:
        t = tokens(m['name'])
        for k in {' '.join(t), ' '.join(t[-2:])}:
            if len(k) >= 6 and (k not in out or m['created'] < out[k]):
                out[k] = m['created']
    return out

def judge(cand, feat, mine, cfg):
    older = [m for m in mine if not cand['created'] or not m['created'] or m['created'] <= cand['created']]
    if not older:
        return None
    vis = (0.0, 0.0, 0.0, False)
    vm = None
    if feat:
        for m in older:
            if m['feat'] is None:
                continue
            s = visual_score(feat, m['feat'])
            if s[0] > vis[0]:
                vis, vm = (s, m)
    nm = max(older, key=lambda m: name_score(cand['name'], m['name']))
    ns = name_score(cand['name'], nm['name'])
    dm = max(older, key=lambda m: desc_score(m['descr'], cand['descr']))
    ds = desc_score(dm['descr'], cand['descr'])
    my_handles = set().union(*(handles(m['descr']) for m in mine)) if mine else set()
    stolen_credit = sorted(handles(cand['descr']) & my_handles)
    v = vis[0]
    if v >= cfg['copy_threshold']:
        verdict, colour = ('Likely COPY / recolour of your model', 15158332)
    elif v >= cfg['remake_threshold']:
        verdict, colour = ('Possible REMAKE of your model', 15105570)
    elif v >= cfg['combo_threshold'] and (ns >= 0.8 or ds >= 0.3 or stolen_credit):
        verdict, colour = ('Possible REMAKE: similar shape + matching name/description', 15105570)
    elif cfg['text_alerts'] and (stolen_credit or ds >= 0.5 or (ns >= 0.9 and ds >= 0.25)):
        verdict, colour = ('Listing text copied (model looks different)', 15844367)
    else:
        verdict, colour = (None, 9807270)
    text = 0.3 * ns + 0.4 * max(ds, 1.0 if stolen_credit else 0.0)
    conf = min(0.99, v + (1 - v) * text) if v >= cfg['combo_threshold'] else min(0.6, 0.25 + 0.35 * ds + (0.2 if stolen_credit else 0))
    return {'verdict': verdict, 'colour': colour, 'confidence': conf, 'visual': v, 'shape': vis[1], 'detail': vis[2], 'mirrored': vis[3], 'name': ns, 'desc': ds, 'credit': stolen_credit, 'ai': None, 'visual_match': vm, 'match': vm if vm and v >= cfg['combo_threshold'] else dm if ds >= 0.25 or stolen_credit else vm or nm}
AI_PROMPT = 'You are checking whether a newly published Roblox accessory copies an existing one.\nImage 1 is the ORIGINAL ("{mine}"). Image 2 is the NEW item ("{theirs}").\nBoth are catalog thumbnails rendered from the same fixed camera. Ignore colour and texture colour.\nLook at silhouette, pose/curl, proportions, and the shape, size and placement of individual\ndetails (tufts, spikes, folds, parts).\nAnswer with JSON only:\n{{"relation": "same_model" | "remake" | "similar_style" | "unrelated",\n  "confidence": 0-100, "reason": "one short sentence naming the deciding details"}}\nsame_model = the same 3D mesh, possibly recoloured, retextured, mirrored or rescaled.\nremake = a different mesh clearly modelled after the original (same pose and distinctive details).\nsimilar_style = same kind of item or trend, but independently made.\nunrelated = different kind of item.'

def _ai_png(img_bytes):
    im = Image.open(io.BytesIO(img_bytes)).convert('RGBA').resize((384, 384))
    bg = Image.new('RGBA', im.size, (120, 124, 130, 255))
    bg.alpha_composite(im)
    out = io.BytesIO()
    bg.convert('RGB').save(out, 'PNG')
    return base64.b64encode(out.getvalue()).decode()

def ai_providers(cfg):
    p = cfg['ai_provider']
    return [n for n, k in (('anthropic', 'anthropic_api_key'), ('gemini', 'gemini_api_key')) if p in ('auto', n) and cfg[k] and (n not in AI_DOWN)]

def ai_provider(cfg):
    return (ai_providers(cfg) or [None])[0]

def ai_configured(cfg):
    return cfg['ai_provider'] != 'off' and bool(cfg['anthropic_api_key'] or cfg['gemini_api_key'])

def ai_allowance(cfg, db):
    day, daily = (time.strftime('%Y-%m-%d'), cfg.get('ai_daily_budget_usd', 0.06))
    a = kv_get(db, 'ai_allow')
    if not a or a['day'] != day:
        used = kv_get(db, 'ai_spend', {}).get(a['day'], 0) if a else 0
        carry = min(0.3, max(0.0, a['carry'] + daily - used)) if a else cfg.get('ai_catchup_usd', 0.2)
        a = {'day': day, 'carry': carry}
        kv_set(db, 'ai_allow', a)
    return daily + a['carry']

def ai_ready(cfg, db):
    day = time.strftime('%Y-%m-%d')
    if not ai_provider(cfg) or kv_get(db, 'ai_calls', {}).get(day, 0) >= cfg['ai_max_calls_per_day']:
        return False
    if kv_get(db, 'ai_spend', {}).get(day, 0) >= ai_allowance(cfg, db):
        STATS['note'] = 'daily AI budget reached, the rest waits for tomorrow'
        return False
    return True
SEVERITY = ['unrelated', 'similar_style', 'remake', 'same_model']

def ai_compare(cfg, db, mine_png, theirs_png, mine_name, theirs_name):
    for p in ai_providers(cfg):
        o = ai_one(cfg, db, mine_png, theirs_png, mine_name, theirs_name, p)
        if o:
            return o
    return None
    got = []
    if len(got) < 2:
        return got[0] if got else None
    a, b = sorted(got, key=lambda o: SEVERITY.index(o['relation']))
    agree = a['relation'] == b['relation']
    return {'relation': a['relation'], 'confidence': max(a['confidence'], b['confidence']) if agree else min(a['confidence'], b['confidence']), 'reason': ' | '.join((f"{o['provider']}: {o['relation'].replace('_', ' ')}, {o['reason']}" for o in got))[:600], 'provider': 'both agree' if agree else 'they disagree'}

def ai_one(cfg, db, mine_png, theirs_png, mine_name, theirs_name, prov):
    if not prov or not mine_png or (not theirs_png):
        return None
    day = time.strftime('%Y-%m-%d')
    used = kv_get(db, 'ai_calls', {})
    if used.get(day, 0) >= cfg['ai_max_calls_per_day']:
        return None
    spend = kv_get(db, 'ai_spend', {})
    if spend.get(day, 0) >= ai_allowance(cfg, db):
        STATS['note'] = 'daily AI budget reached, the rest waits for tomorrow'
        return None
    prompt = AI_PROMPT.format(mine=mine_name[:80], theirs=theirs_name[:80])
    a, b = (_ai_png(mine_png), _ai_png(theirs_png))
    try:
        if prov == 'anthropic':
            r = requests.post('https://api.anthropic.com/v1/messages', timeout=60, headers={'x-api-key': cfg['anthropic_api_key'], 'anthropic-version': '2023-06-01', 'content-type': 'application/json', **({'anthropic-workspace-id': cfg['anthropic_workspace_id']} if cfg['anthropic_workspace_id'] else {})}, json={'model': cfg['anthropic_model'], 'max_tokens': 1500, 'messages': [{'role': 'user', 'content': [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': a}}, {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': b}}, {'type': 'text', 'text': prompt}]}]})
        else:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{cfg['gemini_model']}:generateContent", timeout=60, headers={'x-goog-api-key': cfg['gemini_api_key'], 'content-type': 'application/json'}, json={'contents': [{'parts': [{'inline_data': {'mime_type': 'image/png', 'data': a}}, {'inline_data': {'mime_type': 'image/png', 'data': b}}, {'text': prompt}]}]})
        if r.status_code != 200:
            STATS['ai_failed'] += 1
            STATS['note'] = f'{prov} returned HTTP {r.status_code}'
            if prov == 'anthropic' and ('credit' in r.text.lower() or r.status_code in (401, 402, 403)) and (kv_get(db, 'ai_alert_day') != day):
                kv_set(db, 'ai_alert_day', day)
                send(cfg, {'username': 'UGC Watch', 'embeds': [{'title': 'Claude reviews have stopped', 'color': 15158332, 'description': 'Claude refused the last review, which usually means the API credit has run out or the key was changed. Add credit in the Claude Console to bring AI reviews back. Until then every item is still checked by the picture and text comparison.'}]})
            if r.status_code == 429:
                AI_DOWN.add(prov)
            log(f'  AI check failed ({prov} HTTP {r.status_code}): {r.text[:300]}')
            return None
        body = r.json()
        parts = body.get('content', []) if prov == 'anthropic' else body['candidates'][0]['content']['parts']
        text = ''.join((p.get('text', '') for p in parts))
        STATS['ai_ok'] += 1
        u = body.get('usage', {}) if prov == 'anthropic' else {}
        cost = (u.get('input_tokens', 0) * cfg.get('ai_price_in', 0.1) + u.get('output_tokens', 0) * cfg.get('ai_price_out', 0.5)) / 1000000.0
        kv_set(db, 'ai_spend', {day: spend.get(day, 0) + cost, 'total': spend.get('total', 0) + cost})
        kv_set(db, 'ai_calls', {day: used.get(day, 0) + 1})
        m = re.search('\\{.*\\}', text, re.S)
        if not m:
            STATS['ai_failed'] += 1
            STATS['note'] = f'{prov} gave no usable answer'
            log(f'  AI check: no answer from {prov}: {str(body)[:300]}')
            return None
        j = json.loads(m.group(0))
        if j.get('relation') not in ('same_model', 'remake', 'similar_style', 'unrelated'):
            return None
        return {'relation': j['relation'], 'confidence': float(j.get('confidence', 0)), 'reason': str(j.get('reason', ''))[:300], 'provider': prov}
    except Exception as e:
        log('  AI check error:', repr(e))
        return None

def apply_ai(res, ai, cfg):
    res['ai'] = ai
    if not ai:
        return res
    strong_text = bool(res['credit']) or res['desc'] >= 0.5
    rel, c = (ai['relation'], ai['confidence'] / 100)
    if rel == 'same_model':
        res.update(verdict='Likely COPY / recolour of your model (AI agrees)', colour=15158332, confidence=max(res['confidence'], min(0.99, 0.5 + 0.5 * c)), match=res['visual_match'] or res['match'])
    elif rel == 'remake':
        res.update(verdict='Possible REMAKE of your model (AI agrees)', colour=15105570, confidence=max(min(res['confidence'], 0.9), 0.4 + 0.4 * c), match=res['visual_match'] or res['match'])
    elif strong_text and cfg['text_alerts']:
        res.update(verdict='Listing text copied (AI: model is different)', colour=15844367, confidence=min(res['confidence'], 0.6))
    elif res['visual'] >= cfg['copy_threshold'] and c < 0.8:
        pass
    elif rel == 'similar_style' and res['verdict']:
        res.update(verdict='Similar look and name (AI: same style, different model)', colour=3447003, confidence=min(res['confidence'], 0.5))
    else:
        res['verdict'] = None
    return res

def send(cfg, payload, url=None):
    url = url or cfg['discord_webhook_url']
    if not url:
        log('  (no webhook set) ', payload['embeds'][0]['title'])
        return True
    for _ in range(5):
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code == 429:
            time.sleep(float(r.json().get('retry_after', 2)) + 0.3)
            continue
        return r.status_code < 300
    return False

def when(iso):
    try:
        return f"<t:{calendar.timegm(time.strptime(iso[:19], '%Y-%m-%dT%H:%M:%S'))}:f>"
    except Exception:
        return iso[:10] or 'unknown'

def pct(v):
    return max(0.0, (v - 0.4) / 0.6)

def label(v):
    for a, b in (('Likely COPY', 'Likely copy or recolour'), ('Possible REMAKE', 'Possible remake'), ('Listing text', 'Listing text copied, model looks different'), ('Similar look', 'Similar look and name')):
        if v.startswith(a):
            return b
    return v

def takedown(cand, res, m):
    ai = res['ai']['relation'] if res['ai'] else ''
    copy = res['verdict'].startswith('Likely COPY')
    if copy and (ai == 'same_model' or (not ai and res['visual'] >= 0.9) or (res.get('mesh') or 0) >= 0.985):
        level = 'Strong: same model as yours'
    elif copy:
        level = 'Possible: compare both in the 3D viewer first'
    else:
        return ('Not on this evidence', '')
    d0, d1 = (m['created'][:10], cand['created'][:10])
    try:
        gap = (calendar.timegm(time.strptime(d1, '%Y-%m-%d')) - calendar.timegm(time.strptime(d0, '%Y-%m-%d'))) // 86400
    except Exception:
        gap = -1
    verb = 'reproduces' if level.startswith('Strong') else 'appears to be copied from'
    t = [f"Original work: {m['name']}", f"https://www.roblox.com/catalog/{m['id']}", f'Published {d0} (UTC) by my group.', '', f"Infringing item: {cand['name']}", f"https://www.roblox.com/catalog/{cand['id']}", f"Published {d1} (UTC) by {cand['creator']}.", '', f'The infringing item {verb} the 3D model of my original accessory. Rendered from the same catalog camera, the two items share the same silhouette, proportions and placement of individual details' + (' (the copy is mirrored or rotated)' if res['mirrored'] else '') + f"; an automated image comparison scored them {pct(res['visual']):.0%} alike."]
    if res['ai']:
        t.append('Side-by-side review: ' + res['ai']['reason'])
    if gap >= 0:
        t.append(f"My item was published {gap} day{('s' if gap != 1 else '')} before the infringing item.")
    if res['credit'] or res['desc'] >= 0.5:
        t.append('Their listing description also reuses text from my listing' + (', including my modeller credit ' + ', '.join(res['credit']) if res['credit'] else '') + '.')
    t.append('I am the rights holder of the original item and did not authorise this use.')
    return (level, '\n'.join(t))
PENDING, TIERS = ({}, {})
RANK = [('Likely COPY', 'copy'), ('Possible REMAKE', 'remake'), ('Listing text', 'text'), ('Similar look', 'similar')]

def tier(v):
    return next((k for k, (p, _) in enumerate(RANK) if v.startswith(p)), 9)
KIND = {8: 'Hat', 41: 'Hair Accessory', 42: 'Face Accessory', 43: 'Neck Accessory', 44: 'Shoulder Accessory', 45: 'Front Accessory', 46: 'Back Accessory', 47: 'Waist Accessory'}
ACCESSORY = {8, 41, 42, 43, 44, 45, 46, 47}

def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def who(c):
    return f"[{c['creator']}](https://www.roblox.com/{('communities' if c.get('ctype') == 'Group' else 'users')}/{c.get('cid')})"

def cost(c):
    return 'Off sale' if c.get('price') is None else str(c['price'])

def fld(n, v):
    return {'name': n, 'value': str(v), 'inline': True}

def alert(cfg, cand, res):
    return send_alert(cfg, cand, res)

def flush_alerts(cfg):
    pass
TIER_NAMES = {'strong': 'strong takedown case', 'possible': 'possible takedown case', 'remake': 'possible remake', 'text': 'copied listing text', 'similar': 'similar look and name', 'copy': 'likely copy'}

def send_alert(cfg, cand, res):
    day = when
    m = res['match']
    level, text = takedown(cand, res, m)
    key = 'strong' if level.startswith('Strong') else 'possible' if text else RANK[min(tier(res['verdict']), 3)][1]
    TIERS[key] = TIERS.get(key, 0) + 1
    PENDING.setdefault('found', []).append((key, cand['id'], cand['name'], cand['creator']))
    link = f"https://www.roblox.com/catalog/{cand['id']}"
    why = []
    if res.get('mesh'):
        if res['mesh'] >= 0.5:
            why.append('3D model is the same as yours' if res['mesh'] >= 1.0 else f"3D shape {res['mesh']:.0%} the same as yours")
    m3 = res.get('m3d') or {}
    if m3.get('counts') and (not m3.get('same_mesh')):
        why.append('Same vertex and face count as your mesh')
    if m3.get('inner'):
        why.append(f'''Their file is still named "{m3['inner'][0][:60]}", same as yours''')
    if AI_MESH.search((m3.get('theirs') or {}).get('mesh_name') or ''):
        why.append('Their mesh name looks like an AI 3D-generator export')
    if res.get('render') is not None:
        why.append(f"Bare mesh without texture looks {pct(res['render']):.0%} like yours")
    if res.get('tex'):
        why.append('Uses the same texture image as yours')
    elif (res.get('texsim') or 0) >= 0.6:
        why.append(f"Texture image {res['texsim']:.0%} the same as yours")
    if res.get('mesh_info'):
        why.append('Their mesh: ' + res['mesh_info'])
    if res['credit']:
        why.append('Uses your credit ' + ', '.join(res['credit']))
    elif res['desc'] >= 0.5:
        why.append(f"Description {res['desc']:.0%} the same as yours")
    if res['ai']:
        why.append(res['ai']['reason'][:400])
    main = {'author': {'name': 'UGC Watch'}, 'title': cand['name'][:250], 'url': link, 'color': res['colour'], 'description': f"**{label(res['verdict'])}**\n{KIND.get(cand.get('atype'), 'Accessory')} · [Roblox Page]({link})" + ('\n\n' + '\n'.join(why) if why else ''), 'fields': [fld('Creator', who(cand)), fld('Price', cost(cand)), fld('Uploaded', day(cand['created'])), fld('Looks like yours', f"{pct(res['visual']):.0%}"), fld('Takedown', level.split(':')[0])], 'footer': {'text': 'Detected'}, 'timestamp': stamp()}
    mine = {'author': {'name': 'Your original'}, 'title': m['name'][:250], 'color': 2829617, 'url': f"https://www.roblox.com/catalog/{m['id']}", 'description': f"Uploaded {day(m['created'])}"}
    if cand.get('thumb'):
        main['thumbnail'] = {'url': cand['thumb']}
    if m.get('thumb'):
        mine['thumbnail'] = {'url': m['thumb']}
    embeds = [main, mine] + (mesh_cards(res['m3d'], res['colour']) if res.get('m3d') else [])
    if text:
        embeds.append({'author': {'name': 'Takedown request text'}, 'color': res['colour'], 'description': '```\n' + text[:3800] + '\n```', 'footer': {'text': 'Draft. Check every line is true before sending it to Roblox.'}})
    ok = send(cfg, {'username': 'UGC Watch', 'embeds': embeds})
    if ok:
        STATS['alerts'] += 1
    time.sleep(0.8)
    return ok

def to_cand(d):
    return {'id': d['id'], 'name': d.get('name') or '', 'descr': d.get('description') or '', 'created': d.get('itemCreatedUtc') or '', 'atype': d.get('assetType'), 'creator': d.get('creatorName') or '?', 'cid': d.get('creatorTargetId'), 'ctype': d.get('creatorType') or 'User', 'price': d.get('price')}
CSRF = {'t': ''}

def details_batch(ids):
    out = []
    for k in range(0, len(ids), 100):
        body = {'items': [{'itemType': 'Asset', 'id': i} for i in ids[k:k + 100]]}
        for _ in range(8):
            LIMIT.before('catalog-details')
            try:
                r = sess.post(CATALOG.replace('/search/items/details', '/catalog/items/details'), json=body, headers={'x-csrf-token': CSRF['t']}, timeout=25)
            except requests.RequestException as e:
                log('  network error:', e)
                time.sleep(2)
                continue
            if r.status_code == 403 and r.headers.get('x-csrf-token'):
                CSRF['t'] = r.headers['x-csrf-token']
                continue
            if r.status_code == 429:
                LIMIT.busy('catalog-details', r.headers.get('Retry-After'))
                continue
            if r.status_code == 200:
                LIMIT.ok('catalog-details')
                out += r.json().get('data', [])
            else:
                log(f'  HTTP {r.status_code} from item details: {r.text[:120]}')
            break
    return out
HARD = {'t': float('inf')}

def search_wide(url, pages, tries=6, max_wait=90, **params):
    cursor = None
    for _ in range(pages):
        if time.time() > HARD['t']:
            log('  time limit for this run reached, the rest continues next run')
            return
        live_poll()
        p = {'Limit': 120, **params}
        if cursor:
            p['Cursor'] = cursor
        j = http_json(url, p, tries=tries, max_wait=max_wait)
        if not j or not j.get('data'):
            return
        yield j['data']
        cursor = j.get('nextPageCursor')
        if not cursor:
            return

def is_seen(db, i):
    return db.execute('select 1 from seen where id=?', (i,)).fetchone() is not None

def mark(db, i):
    db.execute('insert or ignore into seen values(?,?)', (i, time.time()))

def get_thumbs(ids):
    urls = thumb_urls(ids)
    ok = [u for u in urls.values() if isinstance(u, str) and u.startswith('http')]
    with ThreadPoolExecutor(8) as pool:
        return (urls, dict(zip(ok, pool.map(fetch_feat, ok))))

def asset_bytes(aid, cfg):
    for k in range(8):
        LIMIT.before('assetdelivery')
        r = sess.get(f'https://assetdelivery.roblox.com/v2/assetId/{aid}', headers={'Cookie': '.ROBLOSECURITY=' + cfg['roblox_cookie']}, timeout=25)
        if r.status_code != 429:
            LIMIT.ok('assetdelivery')
            break
        LIMIT.busy('assetdelivery', r.headers.get('Retry-After'))
    loc = (r.json().get('locations') or [{}])[0].get('location') if r.text[:1] == '{' else None
    if not loc:
        STATS['mesh_fail'] += 1
        if STATS['mesh_fail'] <= 3:
            log(f'  asset {aid} not given: HTTP {r.status_code} {r.text[:140]}')
        return None
    return sess.get(loc, timeout=40).content

def _lz4(src, size):
    out, i = (bytearray(), 0)
    while i < len(src):
        t = src[i]
        i += 1
        n = t >> 4
        if n == 15:
            while True:
                b = src[i]
                i += 1
                n += b
                if b != 255:
                    break
        out += src[i:i + n]
        i += n
        if i >= len(src):
            break
        off = src[i] | src[i + 1] << 8
        i += 2
        n = t & 15
        if n == 15:
            while True:
                b = src[i]
                i += 1
                n += b
                if b != 255:
                    break
        for _ in range(n + 4):
            out.append(out[-off])
    return bytes(out[:size])

def mesh_geom(data):
    if data[:8] != b'version ':
        return (None, 0)
    ver, start = (data[8:9], data.index(b'\n') + 1)
    if ver == b'1':
        g = re.findall(b'\\[([^\\]]+)\\]', data[start:])[0::3]
        v = np.array([[float(x) for x in t.split(b',')] for t in g], np.float32)
        return (v, len(v) // 3)
    if ver in b'23':
        hsize, vsize = struct.unpack_from('<HB', data, start)
        n, nf = struct.unpack_from('<II', data, start + hsize - 8)
    elif ver in b'45':
        hsize, _, n, nf = struct.unpack_from('<HHII', data, start)
        vsize = 40
    else:
        return mesh_points_chunked(data, start)
    if not 0 < n < 2000000 or start + hsize + n * vsize > len(data):
        return (None, 0)
    a = np.frombuffer(data, np.uint8, n * vsize, start + hsize).reshape(n, vsize)
    return (a[:, :12].copy().view('<f4').reshape(n, 3), nf)

def mesh_points(data):
    return mesh_geom(data)[0]
DRACO = {'tried': False, 'mod': None}

def _draco():
    if not DRACO['tried']:
        DRACO['tried'] = True
        try:
            import DracoPy
        except ImportError:
            try:
                import subprocess
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'DracoPy'], timeout=180, check=True)
                import DracoPy
            except Exception as e:
                log('  could not get the mesh decoder:', repr(e)[:100])
                return None
        DRACO['mod'] = DracoPy
    return DRACO['mod']

def mesh_points_chunked(data, i):
    seen = []
    while i + 16 <= len(data):
        kind = data[i:i + 8].rstrip(b'\x00')
        ver, size = struct.unpack_from('<II', data, i + 8)
        body = data[i + 16:i + 16 + size]
        i += 16 + size
        seen.append(f"{kind.decode('latin1')} v{ver} {size}b")
        if kind != b'COREMESH' or len(body) < 8:
            continue
        n = struct.unpack_from('<I', body, 0)[0]
        if ver == 1 and 0 < n < 2000000 and (8 + n * 40 <= len(body)):
            a = np.frombuffer(body, np.uint8, n * 40, 4).reshape(n, 40)
            return (a[:, :12].copy().view('<f4').reshape(n, 3), struct.unpack_from('<I', body, 4 + n * 40)[0])
        d = _draco()
        if d:
            for off in (4, 0, 8):
                try:
                    m = d.decode(bytes(body[off:]))
                    return (np.asarray(m.points, np.float32).reshape(-1, 3), len(np.asarray(m.faces).reshape(-1, 3)))
                except Exception:
                    pass
    if not DRACO.get('noted'):
        DRACO['noted'] = True
        log('  mesh layout not decoded, chunks: ' + ', '.join(seen)[:300])
    return (None, 0)

def _d2(v):
    v = v[np.lexsort(v.T)]
    rng = np.random.default_rng(0)
    d = np.linalg.norm(v[rng.integers(0, len(v), 20000)] - v[rng.integers(0, len(v), 20000)], axis=1)
    hist = np.histogram(d / (d.mean() + 1e-09), bins=48, range=(0, 3))[0].astype(np.float32)
    return hist / hist.sum()
VG = 24

def _vox(v):
    v = v[np.isfinite(v).all(1)].astype(np.float64)
    c = v - v.mean(0)
    _, axes = np.linalg.eigh(c.T @ c)
    p = c @ axes[:, ::-1]
    p *= np.where((p ** 3).mean(0) < 0, -1.0, 1.0)
    p /= np.percentile(np.abs(p), 99.5, axis=0).max() + 1e-09
    p = np.clip(p, -1, 1)
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
    if a is None or b is None:
        return (None, None)
    ga, gb = (_grid(a), _grid(b))
    da, na = (_thick(ga), max(1, ga.sum()))
    strict = slack = 0.0
    tries = []
    for perm in itertools.permutations(range(3)):
        gp = gb.transpose(perm)
        for f in range(8):
            h = gp[::-1 if f & 1 else 1, ::-1 if f & 2 else 1, ::-1 if f & 4 else 1]
            iou = (ga & h).sum() / max(1, (ga | h).sum())
            tries.append((iou, h))
            strict = max(strict, iou)
            slack = max(slack, min((ga & _thick(h)).sum() / na, (h & da).sum() / max(1, h.sum())))
    for _, h in sorted(tries, key=lambda t: -t[0])[:3]:
        pts = np.argwhere(h).astype(np.float64) - (VG - 1) / 2
        for ax in range(3):
            i, j = [k for k in range(3) if k != ax]
            for deg in range(-40, 41, 8):
                if not deg:
                    continue
                c, sn = (np.cos(np.radians(deg)), np.sin(np.radians(deg)))
                q = pts.copy()
                q[:, i], q[:, j] = (c * pts[:, i] - sn * pts[:, j], sn * pts[:, i] + c * pts[:, j])
                ix = np.clip((q + (VG - 1) / 2).round().astype(int), 0, VG - 1)
                g = np.zeros_like(ga)
                g[ix[:, 0], ix[:, 1], ix[:, 2]] = True
                strict = max(strict, (ga & g).sum() / max(1, (ga | g).sum()))
                slack = max(slack, min((ga & _thick(g)).sum() / na, (g & da).sum() / max(1, g.sum())))
    return (float(strict), float(slack))
STD_NAMES = {'handle', 'mesh', 'thumbnailconfiguration', 'thumbnailcameratarget', 'thumbnailcameravalue', 'avatarpartscaletype', 'originalsize', 'surfaceappearance', 'wraplayer', 'accessory', 'specialmesh', 'meshpart', 'weld', 'accessoryweld', 'rigidconstraint'}

def _strings(body):
    out, i = ([], 1)
    if not body or body[0] != 1:
        return out
    while i + 4 <= len(body):
        n = struct.unpack_from('<I', body, i)[0]
        if n > 500 or i + 4 + n > len(body):
            break
        out.append(body[i + 4:i + 4 + n].decode('utf-8', 'replace'))
        i += 4 + n
    return out
GENERIC_NAMES = {'handleaccessory', 'accessory', 'handle', 'mesh', 'default', 'model', 'item', 'ugc', 'hat', 'tail'}

def _inner_name(x):
    x = (x or '').strip()
    m = re.fullmatch('\\w+ \\((.*)\\)', x)
    if m:
        x = m.group(1).strip()
    low = x.lower()
    if not x or low in STD_NAMES or low in GENERIC_NAMES or low.endswith('attachment') or (len(tokens(x)) < 2):
        return ''
    return x

def model_ids(data):
    blobs, names = ([], [])
    if data[:8] == b'<roblox!':
        i = 32
        while i + 16 <= len(data):
            name, clen, ulen = (data[i:i + 4], *struct.unpack_from('<II', data, i + 4))
            i += 16
            raw = data[i:i + (clen or ulen)]
            i += clen or ulen
            if clen and raw[:4] == b'(\xb5/\xfd':
                import zstandard
                raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=ulen)
            elif clen:
                raw = _lz4(raw, ulen)
            if name == b'PROP' and len(raw) > 8:
                n = struct.unpack_from('<I', raw, 4)[0]
                pname, body = (raw[8:8 + n].decode('latin1').lower(), raw[8 + n:])
                blobs.append((pname, body))
                if pname == 'name':
                    names += _strings(body)
            if name == b'END\x00':
                break
    else:
        for m in re.finditer(b'name="(\\w+)"[^>]*>(?:<url>)?([^<]*)', data):
            blobs.append((m.group(1).decode().lower(), m.group(2)))
            if m.group(1) == b'Name':
                names.append(m.group(2).decode('utf-8', 'replace'))
    out = {'names': [x for x in map(_inner_name, names) if x]}
    for name, body in blobs:
        ids = re.findall(b'(?:rbxassetid://|asset/\\?id=|assetid=)(\\d{5,})', body, re.I)
        kind = 'mesh' if 'mesh' in name else 'tex' if 'texture' in name or 'colormap' in name else None
        if kind and ids and (kind not in out):
            out[kind] = int(ids[0])
    return out
AI_MESH = re.compile('tripo|meshy|rodin|hyper3d|luma|csm_|kaedim|sloyd|genie|hunyuan', re.I)

def tex_feats(im):
    out = {}
    g = np.asarray(im.convert('L').resize((32, 32)), np.float32)
    if g.std() >= 4:
        out['texv'] = (g - g.mean()) / g.std()
        c = np.asarray(im.convert('RGB').resize((64, 64)), np.float32)
        e = sum((np.hypot(*np.gradient(c[..., k])) for k in range(3)))
        if e.std() > 0:
            out['texe'] = (e - e.mean()) / e.std()
    return out

def mesh_desc(asset_id, cfg, previews=True):
    try:
        data = asset_bytes(asset_id, cfg)
        ids = model_ids(data) if data else {}
        if ids.get('mesh'):
            out = {'mesh_id': ids['mesh'], 'tex_id': ids.get('tex'), 'names': ids.get('names', [])}
            mb = asset_bytes(ids['mesh'], cfg)
            if mb:
                out['sha'] = hashlib.sha256(mb).hexdigest()
                v, nf = mesh_geom(mb)
                if v is not None and len(v) >= 50 and np.isfinite(v).any():
                    out.update(n=len(v), nf=nf, hist=_d2(v[np.isfinite(v).all(1)]), vox=_vox(v))
                    fv = v[np.isfinite(v).all(1)]
                    pick = np.random.default_rng(0).choice(len(fv), min(2500, len(fv)), replace=False)
                    out['pts'] = fv[np.sort(pick)].astype(np.float16)
                else:
                    log(f"  mesh {ids['mesh']} of item {asset_id}: format {mb[:12]!r} not readable, comparing by id and file only")
            tb = asset_bytes(ids['tex'], cfg) if ids.get('tex') else None
            if tb:
                out['tex_sha'] = hashlib.sha256(tb).hexdigest()
                try:
                    im = Image.open(io.BytesIO(tb)).convert('RGB')
                    out.update(tex_feats(im))
                    out['tex_small'] = np.asarray(im.resize((64, 64)), np.uint8)
                except Exception:
                    pass
            try:
                want = [i for i in (ids['mesh'], ids.get('tex')) if i]
                tu = thumb_urls(want)
                if any((tu.get(i) == 'Pending' for i in want)):
                    time.sleep(3)
                    tu = thumb_urls(want)
                u = tu.get(ids['mesh'])
                if isinstance(u, str) and u.startswith('http'):
                    out['render_url'] = u
                    if previews:
                        out['render'] = fetch_feat(u)[0]
                u = tu.get(ids.get('tex'))
                if isinstance(u, str) and u.startswith('http'):
                    out['tex_url'] = u
                j = http_json(f"https://economy.roblox.com/v2/assets/{ids['mesh']}/details", tries=2) or {}
                if j.get('Name'):
                    out['mesh_name'], out['mesh_by'] = (j['Name'], (j.get('Creator') or {}).get('Name', '?'))
                    out['mesh_date'] = j.get('Created') or ''
                    out['mesh_info'] = f'''"{j['Name'][:60]}" uploaded by {out['mesh_by']} on {out['mesh_date'][:10]}'''
            except Exception as e:
                log('  mesh preview unavailable:', repr(e)[:100])
            return out
        if data:
            log(f'  item {asset_id}: no mesh id found in its file (starts {data[:16]!r})')
    except Exception as e:
        STATS['mesh_fail'] += 1
        log('  model file unavailable:', repr(e)[:120])
    return mesh_desc_3d(asset_id, cfg)

def mesh_desc_3d(asset_id, cfg):
    try:
        h = {'Cookie': '.ROBLOSECURITY=' + cfg['roblox_cookie']}
        for _ in range(4):
            r = sess.get('https://thumbnails.roblox.com/v1/assets-thumbnail-3d', params={'assetId': asset_id}, headers=h, timeout=25)
            j = r.json() if r.text[:1] == '{' else {}
            if j.get('state') == 'Completed':
                break
            time.sleep(2)
        else:
            STATS['mesh_fail'] += 1
            if STATS['mesh_fail'] <= 3:
                log(f'  3D model not given for {asset_id}: HTTP {r.status_code} {r.text[:140]}')
            return None
        meta = sess.get(j['imageUrl'], timeout=25).json()
        k = 31
        for ch in meta['obj'][:32]:
            k ^= ord(ch)
        obj = ''
        for n in [k % 8] + [x for x in range(8) if x != k % 8]:
            obj = sess.get(f"https://t{n}.rbxcdn.com/{meta['obj']}", timeout=40).text
            if '\nv ' in obj[:4000] or obj.startswith('v '):
                break
        v = np.array([[float(x) for x in ln.split()[1:4]] for ln in obj.splitlines() if ln.startswith('v ')], np.float32)
        if len(v) < 50:
            STATS['mesh_fail'] += 1
            log(f'  3D model for {asset_id} unreadable: {len(v)} points, starts {obj[:60]!r}')
            return None
        rng = np.random.default_rng(0)
        d = np.linalg.norm(v[rng.integers(0, len(v), 20000)] - v[rng.integers(0, len(v), 20000)], axis=1)
        hist = np.histogram(d / (d.mean() + 1e-09), bins=48, range=(0, 3))[0].astype(np.float32)
        time.sleep(1.0)
        return {'n': len(v), 'hist': hist / hist.sum()}
    except Exception as e:
        STATS['mesh_fail'] += 1
        log('  3D model unavailable:', repr(e)[:120])
        return None

def mesh_score(a, b):
    if a.get('mesh_id') and a.get('mesh_id') == b.get('mesh_id') or (a.get('sha') and a.get('sha') == b.get('sha')):
        return 1.0
    if a.get('hist') is None or b.get('hist') is None:
        return None
    return float(1 - 0.5 * np.abs(a['hist'] - b['hist']).sum())

def render_score(a, b):
    if a.get('render') is None or b.get('render') is None:
        return None
    return float(visual_score(a['render'], b['render'])[0])

def texture_score(a, b):
    if a.get('texv') is None or b.get('texv') is None:
        return None
    s = float(max(0.0, (a['texv'] * b['texv']).mean()))
    if a.get('texe') is not None and b.get('texe') is not None:
        s = max(s, float((a['texe'] * b['texe']).mean()))
    return s

def same_texture(a, b):
    return bool(a.get('tex_id') and a.get('tex_id') == b.get('tex_id') or (a.get('tex_sha') and a.get('tex_sha') == b.get('tex_sha')))

def geo_compare(t, y):
    same = bool(t.get('mesh_id') and t.get('mesh_id') == y.get('mesh_id') or (t.get('sha') and t.get('sha') == y.get('sha')))
    vox, slack = (1.0, 1.0) if same else vox_score(t.get('vox'), y.get('vox'))
    counts = bool((t.get('n') or 0) >= 200 and t.get('n') == y.get('n') and (t.get('nf') == y.get('nf')))
    mine_names = {_inner_name(n).lower() for n in y.get('names', [])} - {''}
    inner = [x for x in map(_inner_name, t.get('names', [])) if x and x.lower() in mine_names]
    return {'same_mesh': same, 'vox': vox, 'slack': slack, 'd2': mesh_score(t, y), 'counts': counts, 'same_tex': same_texture(t, y), 'texsim': texture_score(t, y), 'inner': inner}

def geo_strength(g):
    return max(1.0 if g['same_mesh'] else 0.0, (g['vox'] or 0.0) + 0.3 * (g['slack'] or 0.0) + (0.1 if g['counts'] else 0.0), 0.9 if g['same_tex'] else 0.0, (g['texsim'] or 0.0) * 0.9, 0.85 if g['inner'] else 0.0)

def my_models(db, cfg, mine):
    import pickle
    db.execute('create table if not exists mymesh(id integer primary key, at real, blob blob)')
    have = {r[0]: r[1] for r in db.execute('select id, blob from mymesh')}
    if not kv_get(db, 'mymesh_v4'):
        db.execute('delete from mymesh')
        kv_set(db, 'mymesh_v4', 1)
        kv_set(db, 'mymesh_full', 0)
        have = {}
    todo = [m for m in mine if m['id'] not in have]
    for m in todo:
        if time.time() > HARD['t']:
            break
        live_poll()
        d = mesh_desc(m['id'], cfg, previews=False)
        if d and d.get('mesh_id'):
            d.pop('render', None)
            db.execute('insert or replace into mymesh values(?,?,?)', (m['id'], time.time(), pickle.dumps(d)))
            have[m['id']] = pickle.dumps(d)
    db.commit()
    names = {m['id']: m for m in mine}
    return [(names[i], pickle.loads(b)) for i, b in have.items() if i in names]

def decide(db, cfg, mine, cand, feat, raw_png, left, src='new'):
    STATS['checked'] += 1
    res = judge(cand, feat, mine, cfg)
    wants = bool(res and (res['verdict'] or res['visual'] >= cfg['ai_min_visual'] or res['name'] >= 0.6 or (res['desc'] >= 0.3) or res['credit']) and res['visual_match'])
    if wants and ai_provider(cfg) and (left <= 0 or not ai_ready(cfg, db)):
        db.execute('insert or ignore into later values(?)', (cand['id'],))
        mark(db, cand['id'])
        db.commit()
        if src == 'new':
            near = res['visual_match'] or res['match']
            ACTIVITY.append((src, cand, res['visual'], near['name'] if near else '', False))
        return 0
    if wants and ai_provider(cfg) and (left > 0):
        vm = res['visual_match']
        if 'png' not in vm:
            vm['png'] = fetch_feat(vm['thumb'])[1] if vm['thumb'] else None
        res = apply_ai(res, ai_compare(cfg, db, vm['png'], raw_png, vm['name'], cand['name']), cfg)
    want3d = bool(res and cfg.get('roblox_cookie') and res['visual_match'] and (res['verdict'] or res['visual'] >= cfg['ai_min_visual'] or res['name'] >= 0.6 or (res['desc'] >= 0.3) or res['credit'] or (res['ai'] and res['ai']['relation'] != 'unrelated')))
    if want3d and src != 'new':
        live_poll()
    theirs = mesh_desc(cand['id'], cfg) if want3d and time.time() < HARD['t'] else None
    if theirs and theirs.get('mesh_id'):
        if MYM['list'] is None:
            MYM['list'] = my_models(db, cfg, mine)
        vm = res['visual_match']
        if 'mesh' not in vm:
            vm['mesh'] = next((d for m, d in MYM['list'] if m['id'] == vm['id']), None) or mesh_desc(vm['id'], cfg, previews=False)
        pool = {m['id']: (m, d) for m, d in MYM['list']}
        if vm['mesh']:
            pool[vm['id']] = (vm, vm['mesh'])
        scored = [(geo_strength(g), m, d, g) for m, d in pool.values() if d for g in [geo_compare(theirs, d)]]
        if scored:
            strength, m, d, g = max(scored, key=lambda x: x[0])
            if strength < 0.75 and vm['mesh']:
                m, d, g = (vm, vm['mesh'], geo_compare(theirs, vm['mesh']))
            if d.get('render') is None and d.get('render_url'):
                d['render'] = fetch_feat(d['render_url'])[0]
            g['render'] = render_score(theirs, d)
            STATS['mesh'] += 1
            r, vox = (g['render'], g['vox'])
            slack = g['slack']
            copy = g['same_mesh'] or (vox or 0) >= cfg.get('vox_copy', 0.8) or (g['counts'] and (slack or 0) >= 0.95) or (vox is None and (r or 0) >= 0.92)
            res.update(mesh=1.0 if copy else min(vox, 0.98) if vox is not None else None, render=r, texsim=g['texsim'], tex=g['same_tex'], mesh_info=theirs.get('mesh_info'))
            mm = {'id': m['id'], 'name': m['name'], 'thumb': m.get('thumb', ''), 'created': m.get('created', '')}
            res['m3d'] = {'theirs': theirs, 'yours': d, 'mine_name': m['name'], 'mine_id': m['id'], 'score': vox, 'render': r, 'texsim': g['texsim'], 'same_tex': g['same_tex'], 'counts': g['counts'], 'same_mesh': g['same_mesh'], 'inner': g['inner']}
            cand['m3d'] = res['m3d']
            log(f"  3D check {cand['name'][:40]} vs {m['name'][:30]}: shape {vox} slack {slack}, counts {g['counts']}, d2 {g['d2']}, bare-mesh {r}, texture {g['texsim']}, same texture {g['same_tex']}, inner {g['inner']}, names {theirs.get('names', [])[:3]} | mesh {theirs.get('mesh_id')}/{theirs.get('tex_id')} vs {d.get('mesh_id')}/{d.get('tex_id')} | {theirs.get('mesh_info')}")
            db.execute('insert or replace into mesh3d values(?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (cand['id'], time.time(), cand['name'], cand['creator'], m['id'], m['name'], theirs.get('mesh_id'), theirs.get('tex_id'), d.get('mesh_id'), d.get('tex_id'), vox, r, g['texsim'], theirs.get('mesh_info') or ''))
            level = None
            ai_no = bool(res['ai'] and res['ai']['relation'] == 'unrelated')
            looks = (slack or 0) >= 0.6 or (r or 0) >= 0.6 or res['visual'] >= 0.6
            if copy:
                level = ('Likely COPY / recolour of your model (3D model matches)', 15158332, 0.97)
            elif not res['verdict'] and g['same_tex']:
                level = ('Possible REMAKE of your model (uses your texture)', 15105570, 0.8)
            elif not res['verdict'] and (not ai_no) and ((g['texsim'] or 0) >= cfg.get('tex_copy', 0.95)):
                level = ('Possible REMAKE of your model (uses your texture)', 15105570, 0.75)
            elif not res['verdict'] and (not ai_no) and g['inner'] and looks:
                level = ('Possible REMAKE of your model (their file carries your internal name)', 15105570, 0.75)
            elif not res['verdict'] and (not ai_no) and ((slack or 0) >= 0.9) and ((vox or 0) >= 0.45) and ((r or 0) >= 0.7):
                level = ('Possible REMAKE of your model (3D shape very close)', 15105570, 0.7)
            if level:
                res.update(verdict=level[0], colour=level[1], confidence=max(res['confidence'], level[2]), match=mm)
            elif res['verdict'] and res['verdict'].startswith('Possible REMAKE') and ((vox or 0) < 0.3) and (r is not None) and (r < 0.45) and (res['visual'] < 0.45):
                res.update(verdict='Similar look and name (3D model differs from yours)', colour=3447003)
    if res and res['verdict'] and res['verdict'].startswith('Listing text') and (not res['credit']) and (res['desc'] < 0.75) and (res['ai'] and res['ai']['relation'] == 'unrelated' or (res.get('render') is not None and res['render'] < 0.3)):
        res['verdict'] = None
    hit = bool(res and res['verdict'])
    if hit and cand.get('cid') in cfg.get('low_priority_creator_ids', []) and (tier(res['verdict']) > 0):
        hit = False
        res['verdict'] += ' (low priority creator, not alerted)'
    if hit and left <= 0:
        return None
    if hit:
        log(f"  MATCH {res['verdict']} | {cand['name']} by {cand['creator']} (visual {res['visual']:.2f}, name {res['name']:.2f}, desc {res['desc']:.2f})")
        if not alert(cfg, cand, res):
            return None
        time.sleep(0.6)
    if res:
        vm, ai = (res['visual_match'] or res['match'], res['ai'])
        db.execute('insert or replace into checked values(?,?,?,?,?,?,?,?,?,?,?,?,?)', (cand['id'], time.time(), cand['name'], cand['creator'], cand['created'], cand.get('thumb', ''), res['visual'], vm['id'], vm['name'], res['name'], res['desc'], res['verdict'] or '', f"{ai['relation']} ({ai['provider']})" if ai else ''))
    near = res['visual_match'] or res['match'] if res else None
    ACTIVITY.append((src, cand, res['visual'] if res else 0.0, near['name'] if near else '', hit))
    mark(db, cand['id'])
    db.commit()
    return 1 if hit else 0

def process(db, cfg, mine, raw, types, budget, src='new'):
    skip = set(cfg['ignore_creator_ids']) | {cfg['my_creator_id']}
    fresh = {}
    for d in raw:
        if d.get('itemType') != 'Asset' or d['id'] in fresh or is_seen(db, d['id']):
            continue
        if d.get('creatorTargetId') in skip or d.get('assetType') not in types:
            mark(db, d['id'])
            continue
        fresh[d['id']] = to_cand(d)
    db.commit()
    if not fresh:
        return 0
    urls, feats = get_thumbs(fresh)
    sent = 0
    for cid, cand in fresh.items():
        if time.time() > HARD['t']:
            break
        if src != 'new':
            live_poll()
        u = urls.get(cid)
        if u in ('Pending', 'InReview', None):
            continue
        ok = isinstance(u, str) and u.startswith('http')
        cand['thumb'] = u if ok else ''
        feat, raw_png = feats[u] if ok else (None, None)
        sent += decide(db, cfg, mine, cand, feat, raw_png, budget - sent, src) or 0
    return sent

def cycle(db, cfg):
    db.execute('create table if not exists checked(id integer primary key, at real, name text, creator text,\n        created text, thumb text, visual real, mine_id int, mine_name text, name_s real, desc_s real,\n        verdict text, ai text)')
    if not kv_get(db, 'recheck_similar'):
        db.execute("delete from seen where id in (select id from checked where verdict='' and visual>=0.5)")
        kv_set(db, 'sweep_at', 0)
        kv_set(db, 'recheck_similar', 1)
    if not kv_get(db, 'requeue_v1'):
        db.execute("insert or ignore into later select id from checked where verdict='' and ai='' and visual>=?", (cfg['ai_min_visual'],))
        kv_set(db, 'requeue_v1', 1)
    HARD['t'] = time.time() + float(os.environ.get('RUN_MINUTES', 17)) * 60
    MYM['list'] = None
    if not kv_get(db, 'fix_inner_v1'):
        db.execute("update checked set verdict='' where verdict like '%internal name%'")
        kv_set(db, 'fix_inner_v1', 1)
    if not kv_get(db, 'm3d_v2'):
        db.execute('delete from mesh3d')
        kv_set(db, 'm3d_v2', 1)
    for k in STATS:
        STATS[k] = '' if k == 'note' else 0
    ACTIVITY.clear()
    PENDING.clear()
    TIERS.clear()
    started = time.time()
    LIVE.update(ctx=None, last=0, busy=False)
    try:
        _cycle(db, cfg)
        live_until_end()
    finally:
        merge_limits(db, LIMIT.export())
        flush_alerts(cfg)
        write_report(db)
        report(db, cfg, started)
        digest(db, cfg)
        post_activity(cfg)
ACTIVITY = []
MYM = {'list': None}

def mesh_cards(m, colour=5793266):
    asset = lambda i: f'https://create.roblox.com/store/asset/{i}'
    ids = lambda d: (f"Mesh [{d['mesh_id']}]({asset(d['mesh_id'])})" if d.get('mesh_id') else 'Mesh not found') + (f" · Texture [{d['tex_id']}]({asset(d['tex_id'])})" if d.get('tex_id') else ' · no texture')
    score = lambda v, f: 'n/a' if v is None else f(v)
    t, y = (m['theirs'], m['yours'])

    def extra(m):
        out = []
        if m.get('counts') and (not m.get('same_mesh')):
            out.append('**Same vertex and face count as yours**')
        if m.get('inner'):
            out.append(f'''**File still named "{m['inner'][0][:50]}", same as yours**''')
        elif t.get('names'):
            out.append(f'''Name inside their file: "{t['names'][0][:50]}"''')
        if AI_MESH.search(t.get('mesh_name') or ''):
            out.append('Mesh name looks like an AI 3D-generator export')
        return out
    theirs = {'author': {'name': 'Their 3D model'}, 'color': colour, 'title': (t.get('mesh_name') or 'Mesh')[:200], 'url': asset(t['mesh_id']) if t.get('mesh_id') else None, 'description': ids(t) + (f"\nMesh uploaded by **{t['mesh_by']}** {when(t['mesh_date'])}" if t.get('mesh_by') else '') + ''.join(('\n' + x for x in extra(m))), 'fields': [fld('Shape match', 'same mesh' if m.get('same_mesh') else score(m['score'], lambda v: f'{v:.0%}')), fld('Bare mesh alike', score(m['render'], lambda v: f'{pct(v):.0%}')), fld('Texture alike', 'same file' if m['same_tex'] else score(m['texsim'], lambda v: f'{v:.0%}'))]}
    yours = {'author': {'name': 'Your 3D model'}, 'color': 2829617, 'title': m['mine_name'][:200], 'url': f"https://www.roblox.com/catalog/{m['mine_id']}", 'description': ids(y)}
    for card, d in ((theirs, t), (yours, y)):
        if d.get('render_url'):
            card['thumbnail'] = {'url': d['render_url']}
        if d.get('tex_url'):
            card['author']['icon_url'] = d['tex_url']
        if not card.get('url'):
            card.pop('url', None)
    return [theirs, yours]

def post_activity(cfg):
    ACTIVITY.clear()

def write_report(db):

    def rows(where, order, n):
        return db.execute(f'select id,name,creator,created,visual,mine_id,mine_name,verdict,ai from checked {where} order by {order} limit {n}').fetchall()

    def esc(s):
        return str(s).replace('|', '/').replace('\n', ' ')[:60]

    def table(rs, result=True):
        out = ['| Their item | Uploaded by | Uploaded | Closest item of yours | Looks alike |' + (' Result |' if result else ''), '|---|---|---|---|---|' + ('---|' if result else '')]
        for r in rs:
            out.append(f'| [{esc(r[1])}](https://www.roblox.com/catalog/{r[0]}) | {esc(r[2])} | {r[3][:10]} | [{esc(r[6])}](https://www.roblox.com/catalog/{r[5]}) | {pct(r[4]):.0%} |' + (f" {(esc(label(r[7])) if r[7] else 'Not flagged')} |" if result else ''))
        return out if rs else ['Nothing here right now.']
    total = db.execute('select count(*) from seen').fetchone()[0]
    full = db.execute('select count(*) from checked').fetchone()[0]
    strong = "where verdict like 'Likely COPY%' or (verdict like 'Possible REMAKE%' and ai like 'remake%')"
    other = "where verdict<>'' and not (verdict like 'Likely COPY%' or (verdict like 'Possible REMAKE%' and ai like 'remake%'))"
    doc = ['# UGC Watch report', '', f"Last updated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}.", '', f'- **{total}** marketplace items looked at so far', f'- **{full}** were the same kind of item as yours and were compared closely', '- **Looks alike** is how closely the catalog pictures match, ignoring colour. Your own recolours score 85% or more and unrelated items score near 0%.', '', '## 1. Worth a takedown look', '', 'Same or copied model according to the picture match and the AI review.', ''] + table(rows(strong, 'visual desc', 100)) + ['', '## 2. Flagged, lower priority', '', 'Similar look, similar name or copied listing text, but probably a different model.', ''] + table(rows(other, 'at desc', 150)) + ['', '## 3. Closest items that were not flagged', '', 'If one of these is clearly a copy, tell Claude so the limits can be adjusted.', ''] + table(rows("where verdict='' and (ai like 'similar%' or ai like 'remake%' or ai like 'same%' or (ai='' and visual>=0.6))", 'visual desc', 40), False) + ['', '## 4. Latest items checked', ''] + table(rows("where ai not like 'unrelated%'", 'at desc', 100))
    lib = lambda i: f'[{i}](https://create.roblox.com/store/asset/{i})' if i else 'not found'
    num = lambda v, f: 'n/a' if v is None else f(v)
    doc += ['', '## 5. 3D model and texture checks', '', 'The real mesh and texture behind each close call, next to the item of yours whose real model is closest. Shape match compares the meshes lined up in the same pose: different models land below about 65%, copies 80% or more (also when resized, turned, mirrored or re-saved), 100% = the very same mesh. Mesh and texture ids open the asset pages.', '', '| Their item | Uploaded by | Your item | Their mesh / texture | Your mesh / texture | Shape match | Bare mesh alike | Texture alike | Their mesh file |', '|---|---|---|---|---|---|---|---|---|']
    for r in db.execute('select * from mesh3d order by score desc, at desc limit 150'):
        doc.append(f"| [{esc(r[2])}](https://www.roblox.com/catalog/{r[0]}) | {esc(r[3])} | [{esc(r[5])}](https://www.roblox.com/catalog/{r[4]}) | {lib(r[6])} / {lib(r[7])} | {lib(r[8])} / {lib(r[9])} | {num(r[10], lambda v: f'{v:.0%}')} | {num(r[11], lambda v: f'{pct(v):.0%}')} | {num(r[12], lambda v: f'{v:.0%}')} | {esc(r[13])} |")
    (HERE / 'REPORT.md').write_text('\n'.join(doc) + '\n', encoding='utf-8')

def roblox_user(cfg):
    if not cfg.get('roblox_cookie'):
        return None
    try:
        r = sess.get('https://users.roblox.com/v1/users/authenticated', headers={'Cookie': '.ROBLOSECURITY=' + cfg['roblox_cookie']}, timeout=20)
        return r.json().get('name', '') if r.status_code == 200 else ''
    except Exception:
        return ''

def report(db, cfg, started):
    a = STATS['alerts']
    if cfg['status_updates'] == 'off' or (cfg['status_updates'] == 'alerts_only' and (not a)):
        return
    now = time.time()
    starts = (kv_get(db, 'starts', []) + [int(started)])[-7:]
    kv_set(db, 'starts', starts)
    gaps = sorted((y - x for x, y in zip(starts, starts[1:]) if y - x >= 20 * 60))
    nxt = int(started) + min(gaps[len(gaps) // 2] if gaps else 1800, 6 * 3600)
    n_mine = db.execute('select count(*) from mine').fetchone()[0]
    found = PENDING.get('found', [])
    if found:
        desc = [f'**[{nm[:60]}](https://www.roblox.com/catalog/{i})** by {cr[:24]} · {TIER_NAMES.get(k, k)}' for k, i, nm, cr in found[:20]]
        if len(found) > 20:
            desc.append(f'...and {len(found) - 20} more in the alerts above')
    else:
        desc = ['Nothing new looks like your items.']
    near = db.execute("select id,name,visual,creator from checked where at>=? and verdict='' and visual>=0.43 and (ai like 'similar%' or ai like 'remake%' or ai like 'same%' or (ai='' and visual>=0.6)) order by visual desc limit 5", (started,)).fetchall()
    fields = []
    add = lambda n, v, inline=True: fields.append({'name': n, 'value': v, 'inline': inline})
    add('Live feed', f"**{STATS.get('live', 0)}** new uploads posted" + (f"\nRoblox said busy {STATS['limited']} times" if STATS.get('limited') else ''))
    add('Full checks', f"**{STATS['checked']}** items vs your **{n_mine}**\n**{STATS['mesh']}** 3D models compared")
    sc = kv_get(db, 'scan_stats', {})
    queued = db.execute('select count(*) from deep').fetchone()[0] if db.execute("select 1 from sqlite_master where name='deep'").fetchone() else 0
    if sc.get('files'):
        add('Catalog scan', (f"**{sc['compared']:,}** compared by picture\n" if sc.get('compared') else '') + (f"**{sc['models']:,}** compared by 3D model\n" if sc.get('models') else '') + f"**{sc['handed']:,}** sent for full checks" + (f', **{queued}** waiting' if queued else '') + f"\n{sc['files']}/{sc.get('total', 10)} machines · <t:{int(sc['at'])}:R>")
    cal, L = (kv_get(db, 'calib', {}), kv_get(db, 'learn', {}))
    learn = [f"Copy line **{cfg['copy_threshold']:.0%}** · 3D **{cfg.get('vox_copy', 0.8):.0%}**"]
    if cal.get('same_n'):
        learn.append(f"From {cal['same_n']} recolour pairs, {cal.get('diff_n', 0)} different-model pairs" + (f", {cal['aug_n']} changed pictures" if cal.get('aug_n') else '') + (f", {cal['aug3_n']} changed 3D models" if cal.get('aug3_n') else '') + (f", {cal['augt_n']} changed textures" if cal.get('augt_n') else ''))
    if L.get('checked'):
        learn.append(f"{L['checked']:,} past checks · {L.get('confirmed', 0)} copies/remakes confirmed")
    add('Learned', '\n'.join(learn))
    rl = kv_get(db, 'rate_limits', {})
    names = {'catalog.roblox.com': 'Catalog search', 'catalog-details': 'Item details', 'assetdelivery': '3D files', 'thumbnails.roblox.com': 'Pictures'}
    if rl:
        add('Roblox limits learned', '\n'.join((f"{names.get(h, h)}: ~{v['budget']:.0f}/min, busy lasts {v['cool']:.1f} s" for h, v in rl.items() if h in names)) or 'nothing hit yet')
    sp = kv_get(db, 'ai_spend', {})
    ai = f"${sp.get(time.strftime('%Y-%m-%d'), 0):.2f} today" if sp.get('total') else 'not used yet'
    waiting = db.execute('select count(*) from later').fetchone()[0]
    add('Claude review', ai + (f'\n**{waiting}** waiting' if waiting else '') + (f"\n{STATS['note']}" if STATS['note'] else ''))
    user = roblox_user(cfg)
    add('Roblox sign-in', 'off (no cookie)' if user is None else f'**{user}**' if user else '**FAILED**: cookie expired')
    add('This run', f'{max(1, round((now - started) / 60))} min · next <t:{nxt}:R>')
    if near:
        add('Closest not flagged', '\n'.join((f'`{pct(r[2]):>4.0%}` [{r[1][:45]}](https://www.roblox.com/catalog/{r[0]}) · {r[3][:20]}' for r in near)), False)
    repo = os.environ.get('DATA_REPO') or os.environ.get('GITHUB_REPOSITORY')
    if repo:
        desc.append(f'\n[Full report](https://github.com/{repo}/blob/main/REPORT.md)')
    title = f"{a} possible cop{('ies' if a != 1 else 'y')} found" if a else 'All clear'
    send(cfg, {'username': 'UGC Watch', 'embeds': [{'author': {'name': 'UGC Watch · run summary'}, 'title': title, 'color': 15105570 if a else 3066993, 'description': '\n'.join(desc), 'fields': fields, 'footer': {'text': 'Checked'}, 'timestamp': stamp()}]})

def digest(db, cfg):
    last = kv_get(db, 'digest_at', 0)
    if not last:
        kv_set(db, 'digest_at', time.time())
        return
    if time.time() - last < 7 * 86400:
        return
    rs = db.execute("select id,name,creator,visual,mine_name,verdict from checked where at>=? and (verdict like 'Likely COPY%' or (verdict like 'Possible REMAKE%' and ai like 'remake%')) order by visual desc limit 25", (last,)).fetchall()
    body = '\n'.join((f'[{r[1][:60]}](https://www.roblox.com/catalog/{r[0]}) by {r[2][:30]} · {pct(r[3]):.0%} like your {r[4][:40]}' for r in rs)) or 'No takedown-worthy cases this week.'
    send(cfg, {'username': 'UGC Watch', 'embeds': [{'title': f"Weekly digest: {len(rs)} case{('s' if len(rs) != 1 else '')} worth a takedown look", 'color': 10181046, 'description': body[:4000], 'footer': {'text': 'Week ending'}, 'timestamp': stamp()}]})
    kv_set(db, 'digest_at', time.time())

def calibrate(db, cfg, mine):
    import pickle
    try:
        meshes = {r[0]: pickle.loads(r[1]) for r in db.execute('select id, blob from mymesh')}
    except sqlite3.OperationalError:
        meshes = {}
    items = [m for m in mine if m['feat'] is not None and (meshes.get(m['id']) or {}).get('mesh_id')]
    key = f"{len(items)}:{sum((m['id'] for m in items)) % 999983}:{time.strftime('%Y-%m-%d')}"
    cal = kv_get(db, 'calib', {})
    if cal.get('key') != key and len(items) >= 6:
        same, diff, vdiff, rng = ([], [], [], np.random.default_rng(0))
        for k, a in enumerate(items):
            live_poll()
            for b in items[k + 1:]:
                v = max(visual_score(a['feat'], b['feat'])[0], visual_score(b['feat'], a['feat'])[0])
                ma, mb = (meshes[a['id']], meshes[b['id']])
                if ma['mesh_id'] == mb['mesh_id'] or (ma.get('sha') and ma.get('sha') == mb.get('sha')):
                    same.append(v)
                else:
                    diff.append(v)
                    if ma.get('vox') is not None and mb.get('vox') is not None:
                        vdiff.append((ma['vox'], mb['vox']))
        aug = augment_scores(items)
        aug3 = augment_models(items, meshes)
        augt = augment_textures(items, meshes)
        if len(vdiff) > 250:
            vdiff = [vdiff[i] for i in rng.choice(len(vdiff), 250, replace=False)]
        strict = []
        for x, y in vdiff:
            live_poll()
            strict.append(vox_score(x, y)[0])
        cal = {'key': key, 'items': len(items), 'same_n': len(same), 'diff_n': len(diff), 'at': time.time()}
        if same:
            cal.update(same_lo=float(np.percentile(same, 10)), same_min=float(min(same)), same_med=float(np.median(same)))
        if diff:
            cal.update(diff_hi=float(np.percentile(diff, 99)), diff_max=float(max(diff)))
        if strict:
            cal.update(vox_diff_hi=float(np.percentile(strict, 99)), vox_diff_max=float(max(strict)))
        if aug:
            allv = [v for vs in aug.values() for v in vs]
            cal.update(aug={k: float(np.median(v)) for k, v in aug.items()}, aug_lo=float(np.percentile(allv, 10)), aug_n=len(allv))
        if aug3:
            all3 = [v for vs in aug3.values() for v in vs]
            cal.update(aug3={k: float(np.median(v)) for k, v in aug3.items()}, aug3_lo=float(np.percentile(all3, 5)), aug3_n=len(all3))
        if augt:
            allt = [v for vs in augt.values() for v in vs]
            cal.update(augt={k: float(np.median(v)) for k, v in augt.items()}, augt_lo=float(np.percentile(allt, 5)), augt_n=len(allt))
        lo = min([cal[k] for k in ('same_lo', 'aug_lo') if k in cal] or [1.0])
        if diff and lo > cal['diff_hi'] + 0.03:
            cal['copy'] = float(min(DEFAULTS['copy_threshold'], max(0.7, lo - 0.02, cal['diff_hi'] + 0.02)))
        floor3 = cal['vox_diff_hi'] + 0.05 if strict else 0.65
        line3 = 0.8
        if strict:
            line3 = max(line3, floor3)
        if aug3 and cal['aug3_lo'] - 0.02 < line3:
            line3 = max(floor3, 0.65, cal['aug3_lo'] - 0.02)
        cal['vox_copy'] = float(min(0.92, line3))
        if augt:
            cal['tex_copy'] = float(min(0.95, max(0.85, cal['augt_lo'] - 0.02)))
        kv_set(db, 'calib', cal)
        log(f'tuned on your items: {len(same)} recolour pairs, {len(diff)} different-model pairs')
    history(db)
    learned(db, cfg)
AUG = {'turned 12°': lambda im: im.rotate(12, expand=True), 'turned -12°': lambda im: im.rotate(-12, expand=True), 'turned 25°': lambda im: im.rotate(25, expand=True), 'mirrored': lambda im: im.transpose(Image.FLIP_LEFT_RIGHT), 'upside down': lambda im: im.transpose(Image.FLIP_TOP_BOTTOM), 'quarter turn': lambda im: im.transpose(Image.ROTATE_90), 'smaller': lambda im: _pad(im.resize((im.width * 2 // 3, im.height * 2 // 3)), im.size), 'recoloured': lambda im: Image.merge('RGBA', (lambda c: (c[2], c[0], c[1], c[3]))(im.split())), 'inverted colours': lambda im: Image.merge('RGBA', (lambda c: (*(ImageOps.invert(x) for x in c[:3]), c[3]))(im.split())), 'grey': lambda im: Image.merge('RGBA', (*[im.convert('L')] * 3, im.split()[3])), 'blurred': lambda im: im.filter(ImageFilter.GaussianBlur(2))}

def _pad(im, size):
    out = Image.new('RGBA', size, (0, 0, 0, 0))
    out.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return out

def augment_scores(items):
    out = {}
    for m in items:
        if not m.get('thumb') or time.time() > HARD['t']:
            continue
        live_poll()
        try:
            r = sess.get(m['thumb'], timeout=20)
            im = Image.open(io.BytesIO(r.content)).convert('RGBA')
            mask, _ = _mask_rgb(im)
            im.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))
        except Exception:
            continue
        for name, f in AUG.items():
            try:
                b = io.BytesIO()
                f(im).save(b, 'PNG')
                fa = features(b.getvalue())
                if fa:
                    out.setdefault(name, []).append(float(visual_score(finish(fa), m['feat'])[0]))
            except Exception:
                pass
    return out

def _rot(axis, deg):
    c, s_ = (np.cos(np.radians(deg)), np.sin(np.radians(deg)))
    i, j = [k for k in range(3) if k != axis]
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = (c, -s_, s_, c)
    return r
AUG3 = {'turned 30°': lambda p, g: p @ _rot(2, 30).T, 'turned 90°': lambda p, g: p @ _rot(1, 90).T, 'tilted 17°+40°': lambda p, g: p @ (_rot(0, 17) @ _rot(1, 40)).T, 'upside down': lambda p, g: p @ _rot(0, 180).T, 'mirrored': lambda p, g: p * [-1, 1, 1], '3x bigger': lambda p, g: p * 3 + 5, 'stretched 10%': lambda p, g: p * [1.1, 1, 1], 're-saved (tiny noise)': lambda p, g: p + g.normal(0, 0.004, p.shape) * np.ptp(p, 0).max(), 'simplified (half the points)': lambda p, g: p[::2], 'everything at once': lambda p, g: p[::2] * [-1, 1, 1] @ (_rot(0, 17) @ _rot(2, 60)).T * 2 + g.normal(0, 0.003, (len(p[::2]), 3)) * np.ptp(p, 0).max()}

def augment_models(items, meshes):
    out, g = ({}, np.random.default_rng(1))
    for m in items:
        p = (meshes.get(m['id']) or {}).get('pts')
        if p is None or len(p) < 100 or time.time() > HARD['t']:
            continue
        live_poll()
        p = p.astype(np.float64)
        base = _vox(p)
        for name, f in AUG3.items():
            try:
                out.setdefault(name, []).append(float(vox_score(base, _vox(f(p, g)))[0]))
            except Exception:
                pass
    return out
AUGT = {'recoloured': lambda a: a[..., [2, 0, 1]], 'darker': lambda a: (a * 0.55).astype(np.uint8), 'brighter': lambda a: np.clip(a * 1.4 + 30, 0, 255).astype(np.uint8), 'inverted': lambda a: 255 - a, 'grey': lambda a: np.repeat(a.mean(-1, keepdims=True), 3, -1).astype(np.uint8), 'blurred': lambda a: np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(1.5)))}

def augment_textures(items, meshes):
    out = {}
    for m in items:
        d = meshes.get(m['id']) or {}
        t = d.get('tex_small')
        if t is None or d.get('texv') is None:
            continue
        for name, f in AUGT.items():
            try:
                fa = tex_feats(Image.fromarray(f(t)))
                if fa.get('texv') is not None:
                    out.setdefault(name, []).append(float(texture_score(fa, tex_feats(Image.fromarray(t)))))
            except Exception:
                pass
    return out

def history(db):
    try:
        conf = [r[0] for r in db.execute("select visual from checked where ai like 'same%' or ai like 'remake%' or verdict like 'Likely COPY%'")]
        unrel = [r[0] for r in db.execute("select visual from checked where ai like 'unrelated%'")]
    except sqlite3.OperationalError:
        return
    L = {'confirmed': len(conf), 'unrelated': len(unrel), 'checked': db.execute('select count(*) from checked').fetchone()[0]}
    if len(conf) >= 5:
        lo = float(np.percentile(conf, 10))
        L.update(confirmed_lo=lo, scan_min_visual=float(min(0.42, max(0.35, lo - 0.05))), ai_min_visual=float(min(0.42, max(0.35, lo - 0.05))))
    if unrel:
        L['unrelated_hi'] = float(np.percentile(unrel, 95))
    kv_set(db, 'learn', L)

def learned(db, cfg):
    cal, L = (kv_get(db, 'calib', {}), kv_get(db, 'learn', {}))
    if cal.get('copy'):
        cfg['copy_threshold'] = min(cfg['copy_threshold'], cal['copy'])
    if cal.get('vox_copy'):
        cfg['vox_copy'] = cal['vox_copy']
    if cal.get('tex_copy'):
        cfg['tex_copy'] = min(0.95, cal['tex_copy'])
    for k in ('scan_min_visual', 'ai_min_visual'):
        if L.get(k):
            cfg[k] = min(cfg.get(k, 0.5), L[k])

def selftest(db, cfg):
    if not load_mine(db):
        refresh_mine(db, cfg)
    mine = [m for m in load_mine(db) if m['feat'] is not None]
    fam = lambda m: ' '.join(tokens(m['name']))
    pairs = [(visual_score(a['feat'], b['feat'])[0], a, b) for k, a in enumerate(mine) for b in mine[k + 1:]]
    print(f'{len(mine)} items, {len(pairs)} pairs')
    print('NEAREST OTHER ITEM OF YOURS, PER ITEM')
    for a in mine:
        s, b = max(((s, y if x is a else x) for s, x, y in pairs if x is a or y is a), key=lambda t: t[0])
        print(f"{s:.2f}  {a['name']} ({a['created'][:10]})  <->  {b['name']} ({b['created'][:10]})")
    cross = sorted((p for p in pairs if fam(p[1]) != fam(p[2])), key=lambda p: -p[0])
    print('HIGHEST SCORES BETWEEN DIFFERENTLY NAMED ITEMS')
    for s, a, b in cross[:12]:
        print(f"{s:.2f}  {a['name']}  <->  {b['name']}")
    allp = sorted((p[0] for p in pairs))
    print('ALL PAIRS: ' + ' '.join((f'{q}%={allp[int(len(allp) * q / 100) - (q == 100)]:.2f}' for q in (10, 25, 50, 75, 90, 95, 100))))
TAXONOMY = {8: 'thukPQUFe5EHxRrtsgq43u', 43: 'tvckLAbtDAGwF9Y2sph2Ce', 44: '8i1sggNTrxoN799hpRrR1S', 45: 'kj535RzJxHTCnXPBLo5KaE', 46: '7yGywzp5JeDsNjhXWJqSw3', 47: '6jYQPDtkVwQZcKtvVpyYAN'}
TYPE_WORDS = {8: 'horns horn hat crown halo ears demon devil dragon helmet hood mask bow'.split(), 46: 'wings wing sword cape bag backpack angel demon dragon chain cross bow'.split(), 47: 'tail tails fluffy fox wolf cat kitsune bunny dragon demon devil furry plush chain belt bow'.split(), 45: 'fan sword chain plush bag'.split(), 43: 'chain choker necklace collar'.split(), 44: 'plush pet'.split()}
PRICE_BANDS = [(0, 64), (65, 65), (66, 75), (76, 90), (91, 110), (111, 150), (151, 250), (251, 600), (601, 10 ** 7)]

def scan_slices(mine, offenders=()):
    by_type = {}
    for m in mine:
        by_type.setdefault(m['atype'], []).append(m)
    out = []
    for t, items in sorted(by_type.items()):
        tax = TAXONOMY.get(t)
        if not tax:
            continue
        for lo, hi in PRICE_BANDS:
            for sort in (3, 2, 1):
                out.append({'taxonomy': tax, 'SortType': sort, 'MinPrice': lo, 'MaxPrice': hi})
        words = {w for m in items for w in tokens(m['name']) if len(w) >= 3} | set(TYPE_WORDS.get(t, ()))
        for w in sorted(words):
            for lo, hi in PRICE_BANDS:
                out.append({'taxonomy': tax, 'Keyword': w, 'SortType': 0, 'MinPrice': lo, 'MaxPrice': hi})
    for kw in sorted(keywords(mine)):
        for sort in (0, 3):
            out.append({'Keyword': kw, 'SortType': sort})
    for nm in offenders:
        for sort in (3, 2):
            out.append({'CreatorName': nm, 'SortType': sort})
    return out

def scan_shard(db, cfg, shard, total, minutes=24):
    QUIET['on'] = True
    mine = [m for m in load_mine(db) if m['feat'] is not None]
    if not mine:
        log('no items of yours in the data yet')
        return
    stop = time.time() + minutes * 60
    HARD['t'] = time.time() + cfg.get('scan_total_minutes', 34) * 60
    try:
        offenders = [r[0] for r in db.execute("select distinct creator from checked where verdict<>''")]
    except sqlite3.OperationalError:
        offenders = []
    learned(db, cfg)
    LIMIT.load(kv_get(db, 'rate_limits'))
    slices = scan_slices(mine, offenders)[shard::total]
    skip = set(cfg['ignore_creator_ids']) | {cfg['my_creator_id']}
    found, done = ({}, 0)
    for sl in slices:
        if time.time() > stop:
            break
        for page in search_wide(CATALOG.replace('/v1/', '/v2/'), 9, salesTypeFilter=1, **sl):
            for d in page:
                if d.get('itemType') == 'Asset' and d.get('creatorTargetId') not in skip:
                    found.setdefault(d['id'], d)
        done += 1
    fresh = [d for i, d in found.items() if not is_seen(db, i)]
    log(f'searched {done} of {len(slices)} slices, {len(found)} items, {len(fresh)} not checked before')
    keep, scored, pic = ([], 0, {})
    for k in range(0, len(fresh), 200):
        if time.time() > HARD['t']:
            break
        chunk = {d['id']: d for d in fresh[k:k + 200]}
        urls, feats = get_thumbs(chunk)
        for i, d in chunk.items():
            u = urls.get(i)
            if not (isinstance(u, str) and u.startswith('http')) or not feats.get(u) or feats[u][0] is None:
                continue
            cand = to_cand(d)
            cand['thumb'] = u
            res = judge(cand, feats[u][0], mine, cfg)
            scored += 1
            pic[i] = res['visual'] if res else 0.0
            if res and (res['verdict'] or res['visual'] >= cfg.get('scan_min_visual', 0.42) or res['name'] >= 0.6 or (res['desc'] >= 0.3) or res['credit']):
                keep.append((max(res['visual'], res['name'] * 0.9, res['desc']), d))
    m3done, m3hits = scan_models(db, cfg, [d for d in fresh if d['id'] in pic], pic)
    kept = {d['id'] for _, d in keep}
    for d, g in m3hits:
        if d['id'] not in kept:
            keep.append((max(0.5, g.get('strict') or 0), d | {'_m3': g}))
    keep = [d | {'_score': round(s, 3)} for s, d in sorted(keep, key=lambda x: -x[0])]
    out = HERE / 'scan'
    out.mkdir(exist_ok=True)
    name = f"{os.environ.get('GITHUB_RUN_ID', int(time.time()))}-{shard}.json.gz"
    stats = {'slices': done, 'of': len(slices), 'found': len(found), 'compared': scored, 'handed': len(keep), 'total': total, 'models': len(m3done), 'model_hits': len(m3hits), 'limits': LIMIT.export()}
    (out / name).write_bytes(gzip.compress(json.dumps({'stats': stats, 'items': keep, 'm3': m3done}).encode()))
    log(f'compared {scored} items by picture, {len(m3done)} by 3D model, {len(keep)} look-alikes handed to the watcher')

def scan_models(db, cfg, items, pic):
    import pickle
    if not cfg.get('roblox_cookie'):
        return ([], [])
    try:
        mymods = [pickle.loads(r[0]) for r in db.execute('select blob from mymesh')]
    except sqlite3.OperationalError:
        mymods = []
    try:
        done_before = {r[0] for r in db.execute('select id from m3seen')}
    except sqlite3.OperationalError:
        done_before = set()
    mymods = [m for m in mymods if m.get('mesh_id')]
    if not mymods:
        return ([], [])
    mesh_ids = {m['mesh_id']: m for m in mymods}
    tex_ids = {m['tex_id']: m for m in mymods if m.get('tex_id')}
    names = {_inner_name(n).lower() for m in mymods for n in m.get('names', [])} - {''}
    cache = {}
    todo = sorted((d for d in items if d['id'] not in done_before), key=lambda d: -pic.get(d['id'], 0))

    def one(d):
        if time.time() > HARD['t']:
            return None
        try:
            data = asset_bytes(d['id'], cfg)
            ids = model_ids(data) if data else {}
            if not ids.get('mesh'):
                return (d['id'], None)
            g = {'mesh_id': ids['mesh'], 'tex_id': ids.get('tex')}
            if ids['mesh'] in mesh_ids:
                g.update(same_mesh=True, mine=mesh_ids[ids['mesh']].get('mesh_id'))
            if ids.get('tex') in tex_ids:
                g['same_tex'] = True
            inner = [x for x in map(_inner_name, ids.get('names', [])) if x and x.lower() in names]
            if inner:
                g['inner'] = inner[:2]
            if not g.get('same_mesh'):
                if ids['mesh'] not in cache:
                    mb = asset_bytes(ids['mesh'], cfg)
                    v, nf = mesh_geom(mb) if mb else (None, 0)
                    ok = v is not None and len(v) >= 50 and np.isfinite(v).any()
                    cache[ids['mesh']] = {'vox': _vox(v), 'hist': _d2(v[np.isfinite(v).all(1)]), 'n': len(v), 'nf': nf} if ok else None
                t = cache[ids['mesh']]
                if t:
                    near = sorted(mymods, key=lambda m: -(mesh_score(t, m) or 0))[:4]
                    best = max((vox_score(t['vox'], m.get('vox')) + (m,) for m in near if m.get('vox') is not None), key=lambda x: x[0] or 0, default=None)
                    if best:
                        g.update(strict=round(best[0], 3), slack=round(best[1], 3))
                    g['counts'] = any((t['n'] >= 200 and t['n'] == m.get('n') and (t['nf'] == m.get('nf')) for m in mymods))
            hit = g.get('same_mesh') or g.get('same_tex') or g.get('inner') or g.get('counts') or ((g.get('strict') or 0) >= 0.5) or ((g.get('slack') or 0) >= 0.85)
            return (d['id'], g if hit else None)
        except Exception:
            return None
    done, hits = ([], [])
    byid = {d['id']: d for d in todo}
    with ThreadPoolExecutor(8) as pool:
        for r in pool.map(one, todo):
            if r is None:
                continue
            done.append(r[0])
            if r[1]:
                hits.append((byid[r[0]], r[1]))
    return (done, hits)

def ingest_scans(db):
    db.execute('create table if not exists deep(id integer primary key, score real, raw text)')
    read, n = ([], 0)
    for f in sorted((HERE / 'scan').glob('*.json.gz')) if (HERE / 'scan').exists() else []:
        try:
            blob = json.loads(gzip.decompress(f.read_bytes()))
            items = blob['items'] if isinstance(blob, dict) else blob
            st = blob.get('stats', {}) if isinstance(blob, dict) else {'handed': len(blob)}
            tot = kv_get(db, 'scan_stats', {})
            run = f.name.split('-')[0]
            if tot.get('run') != run:
                tot = {'run': run, 'at': time.time(), 'found': 0, 'compared': 0, 'handed': 0, 'files': 0, 'seen': [], 'slices': 0, 'of': 0}
            if f.name not in tot.setdefault('seen', []):
                tot['total'] = st.get('total', tot.get('total', 10))
                for k in ('found', 'compared', 'handed', 'slices', 'of', 'models', 'model_hits'):
                    tot[k] = tot.get(k, 0) + st.get(k, 0)
                tot['files'] = tot.get('files', 0) + 1
                tot['seen'].append(f.name)
            kv_set(db, 'scan_stats', tot)
            db.execute('create table if not exists m3seen(id integer primary key)')
            if isinstance(blob, dict):
                db.executemany('insert or ignore into m3seen values(?)', [(i,) for i in blob.get('m3') or []])
                merge_limits(db, (blob.get('stats') or {}).get('limits'))
            for d in items:
                if not is_seen(db, d['id']):
                    db.execute('insert or ignore into deep values(?,?,?)', (d['id'], d.get('_score', 0), json.dumps(d)))
                    n += 1
        except Exception as e:
            log('  scan file unreadable:', f.name, repr(e)[:80])
        read.append(f'scan/{f.name}')
    db.commit()
    if read:
        f = HERE / 'ingested.txt'
        before = set(f.read_text().split()) if f.exists() else set()
        f.write_text('\n'.join(sorted(before | set(read))) + '\n')
        log(f'catalog scanner: {n} new look-alikes from {len(read)} files')

def _cycle(db, cfg):
    if time.time() - kv_get(db, 'mine_at', 0) > cfg['refresh_my_items_hours'] * 3600 or not kv_get(db, 'store_v2'):
        refresh_mine(db, cfg)
        kv_set(db, 'store_v2', 1)
    mine = load_mine(db)
    if not mine:
        log('no items of yours indexed yet; check my_creator_id')
        return
    types = {m['atype'] for m in mine} | set(cfg['extra_asset_types'])
    cfg.update(ai_max_calls_per_day=10 ** 9, max_alerts_per_cycle=10 ** 9)
    budget = cfg['max_alerts_per_cycle']
    deadline = HARD['t']
    LIMIT.load(kv_get(db, 'rate_limits'))
    LIVE.update(ctx=(db, cfg, mine, types), last=0, busy=False)
    live_poll(catch_up=True)
    if cfg.get('roblox_cookie') and (not kv_get(db, 'mymesh_full')):
        MYM['list'] = my_models(db, cfg, mine)
        if db.execute('select count(*) from mymesh').fetchone()[0] >= len(mine) - 2:
            kv_set(db, 'mymesh_full', 1)
    calibrate(db, cfg, mine)
    if ai_ready(cfg, db):
        ids = [r[0] for r in db.execute('select id from later limit ?', (int(os.environ.get('DEEP_PER_RUN', 10 ** 9)),))]
        if ids:
            q = ','.join(map(str, ids))
            db.execute(f'delete from later where id in ({q})')
            db.execute(f'delete from seen where id in ({q})')
            db.commit()
            budget -= process(db, cfg, mine, details_batch(ids), ACCESSORY, budget, 'scan')
            for i in ids:
                if not is_seen(db, i):
                    db.execute('insert or ignore into later values(?)', (i,))
                    mark(db, i)
            db.commit()
            log(f'looked again at {len(ids)} items that were waiting for an AI review')
    ingest_scans(db)
    check_deep(db, cfg, mine, types)

def live_poll(catch_up=False):
    if LIVE['busy'] or not LIVE.get('ctx'):
        return
    db, cfg, mine, types = LIVE['ctx']
    if not catch_up and time.time() - LIVE['last'] < cfg.get('live_poll_seconds', 15):
        return
    LIVE['busy'] = True
    waited0 = STATS['waited']
    try:
        db.execute('create table if not exists fed(id integer primary key, at real)')
        db.execute('create table if not exists feedwait(id integer primary key, at real)')
        first = not kv_get(db, 'feed_since')
        if first:
            kv_set(db, 'feed_since', time.time())
        cut = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(time.time() - cfg.get('feed_max_age_days', 3) * 86400))
        got = posted = 0
        pages = cfg.get('feed_catch_up_pages', 60) if catch_up or first else cfg.get('feed_live_pages', 10)
        url = CATALOG.replace('/v1/', '/v2/')
        recheck = [r[0] for r in db.execute('select id from feedwait where at<? order by at limit 100', (time.time() - cfg.get('feed_recheck_seconds', 600),))]
        lists = [{'IncludeNotForSale': 'true'}]
        if catch_up or first or time.time() - LIVE.get('onsale_at', 0) > cfg.get('feed_onsale_seconds', 120):
            lists.append({'salesTypeFilter': 1})
            LIVE['onsale_at'] = time.time()
        for params in lists:
            quick = {} if catch_up or first else {'tries': 3, 'max_wait': 12}
            for page in search_wide(url, pages, Category=1, SortType=3, **quick, **params):
                page = [d for d in page if d.get('itemType') == 'Asset']
                new = [d for d in page if not db.execute('select 1 from fed where id=?', (d['id'],)).fetchone() and (not db.execute('select 1 from feedwait where id=?', (d['id'],)).fetchone())]
                if not new:
                    break
                got += len(new)
                if first:
                    db.executemany('insert or ignore into fed values(?,?)', [(d['id'], time.time()) for d in new])
                    continue
                posted += feed_items(db, cfg, mine, types, new, cut)
                db.commit()
        if recheck and (not first):
            db.executemany('update feedwait set at=? where id=?', [(time.time(), i) for i in recheck])
            items = details_batch(recheck)
            for i in set(recheck) - {d['id'] for d in items}:
                db.execute('delete from feedwait where id=?', (i,))
            posted += feed_items(db, cfg, mine, types, items, cut, recheck=True)
        flush_feed(cfg)
        db.commit()
        waited = STATS['waited'] - waited0
        if got or waited:
            log(f"feed: {got} new listings, {posted} posted, waited {round(waited)} s for Roblox, {STATS.get('limited', 0)} busy replies so far")
    except Exception as e:
        log('live feed error:', repr(e)[:160])
    finally:
        LIVE['busy'] = False
        LIVE['last'] = time.time()

def feed_items(db, cfg, mine, types, items, cut, recheck=False):
    show = []
    for d in items:
        limited = set(d.get('itemRestrictions') or []) & {'Collectible', 'Limited', 'LimitedUnique'}
        if limited or d.get('creatorTargetId') == 1 or (d.get('itemCreatedUtc') or '')[:19] < cut:
            db.execute('insert or ignore into fed values(?,?)', (d['id'], time.time()))
            db.execute('delete from feedwait where id=?', (d['id'],))
        elif d.get('price') is None and d.get('lowestPrice') is None:
            if not recheck:
                db.execute('insert or ignore into feedwait values(?,?)', (d['id'], time.time()))
        else:
            db.execute('delete from feedwait where id=?', (d['id'],))
            show.append(d)
    show.sort(key=lambda d: d.get('itemCreatedUtc') or '')
    n = 0
    for d in show:
        process(db, cfg, mine, [d], ACCESSORY | types, cfg.get('max_alerts_per_cycle', 10 ** 9))
        n += feed_cards(db, cfg, [d])
    return n
ASSET_KIND = {8: 'Hat', 41: 'Hair Accessory', 42: 'Face Accessory', 43: 'Neck Accessory', 44: 'Shoulder Accessory', 45: 'Front Accessory', 46: 'Back Accessory', 47: 'Waist Accessory', 61: 'Emote', 64: 'T-Shirt Accessory', 65: 'Shirt Accessory', 66: 'Pants Accessory', 67: 'Jacket Accessory', 68: 'Sweater Accessory', 69: 'Shorts Accessory', 70: 'Left Shoe Accessory', 71: 'Right Shoe Accessory', 72: 'Dress Skirt Accessory', 76: 'Eyebrow Accessory', 77: 'Eyelash Accessory', 78: 'Mood Animation', 79: 'Dynamic Head', 92: 'Avatar Background'}

def feed_cards(db, cfg, items):
    url = cfg.get('feed_webhook_url')
    info = {e[1]['id']: e for e in ACTIVITY if e[0] == 'new'}
    ACTIVITY[:] = [e for e in ACTIVITY if e[0] != 'new']
    pics = thumb_urls([d['id'] for d in items])
    cards = []
    for d in items:
        u = pics.get(d['id'])
        if u in ('Pending', 'InReview', None) and d['id'] not in info:
            db.execute('insert or ignore into feedwait values(?,?)', (d['id'], time.time()))
            continue
        db.execute('insert or ignore into fed values(?,?)', (d['id'], time.time()))
        db.execute('delete from feedwait where id=?', (d['id'],))
        c = to_cand(d)
        link = f"https://www.roblox.com/catalog/{c['id']}"
        e = info.get(c['id'])
        hit = bool(e and e[4])
        mine_item = c['cid'] == cfg['my_creator_id']
        fields = [fld('Price', cost(c)), fld('Creator', who(c))]
        if mine_item:
            fields.append(fld('Closest to yours', 'your own item'))
        elif e:
            fields.append(fld('Closest to yours', f'{pct(e[2]):.0%} · {e[3][:40]}' if e[3] else 'none'))
        fields.append(fld('Uploaded', when(c.get('created') or '')))
        card = {'author': {'name': 'UGC Watch'}, 'title': c['name'][:250], 'url': link, 'color': 15105570 if hit else 3066993 if mine_item else 3108845, 'description': f"New UGC {ASSET_KIND.get(c.get('atype'), 'Item')}\n[Roblox Page]({link})" + ('\n**Flagged, see the alerts channel**' if hit else ''), 'fields': fields, 'footer': {'text': 'Found'}, 'timestamp': stamp()}
        if isinstance(u, str) and u.startswith('http'):
            card['thumbnail'] = {'url': u}
        cards.append(card)
    if url:
        if cards and (not FEEDBUF['cards']):
            FEEDBUF['since'] = time.time()
        FEEDBUF['cards'] += cards
        if len(FEEDBUF['cards']) >= 10 or time.time() - FEEDBUF['since'] >= 3:
            flush_feed(cfg)
    STATS['live'] += len(cards)
    return len(cards)
FEEDBUF = {'cards': [], 'since': 0.0}

def flush_feed(cfg):
    url = cfg.get('feed_webhook_url')
    cards, FEEDBUF['cards'] = (FEEDBUF['cards'], [])
    for k in range(0, len(cards), 10):
        send(cfg, {'username': 'UGC Watch', 'embeds': cards[k:k + 10]}, url)
        time.sleep(1.0)

def merge_limits(db, new):
    old = kv_get(db, 'rate_limits', {})
    for h, v in (new or {}).items():
        o = old.get(h)
        old[h] = v if not o else {'budget': round((o['budget'] + v['budget']) / 2, 1), 'cool': round((o['cool'] + v['cool']) / 2, 2), 'busy': v.get('busy', 0)}
    kv_set(db, 'rate_limits', old)

def check_deep(db, cfg, mine, types):
    rows = db.execute('select id, raw from deep order by score desc').fetchall()
    if not rows:
        return
    ids = [r[0] for r in rows]
    db.execute(f"delete from deep where id in ({','.join(map(str, ids))})")
    db.commit()
    process(db, cfg, mine, [json.loads(r[1]) for r in rows], ACCESSORY | types, cfg.get('max_alerts_per_cycle', 10 ** 9), 'scan')
    back = [r for r in rows if not is_seen(db, r[0])]
    for i, raw in back:
        db.execute('insert or ignore into deep values(?,?,?)', (i, json.loads(raw).get('_score', 0), raw))
    db.commit()
    log(f"catalog scanner: checked {len(rows) - len(back)} look-alikes, {db.execute('select count(*) from deep').fetchone()[0]} still queued")

def pull_scans():
    import subprocess
    if not (HERE / '.git').exists():
        return False
    try:
        subprocess.run(['git', '-C', str(HERE), 'fetch', '-q', 'origin', 'main'], timeout=60, check=True)
        names = subprocess.run(['git', '-C', str(HERE), 'ls-tree', '--name-only', 'origin/main', 'scan/'], capture_output=True, text=True, timeout=30).stdout.split()
        new = [n for n in names if not (HERE / n).exists()]
        if new:
            subprocess.run(['git', '-C', str(HERE), 'checkout', 'origin/main', '--'] + new, timeout=60, check=True)
        return bool(new)
    except Exception as e:
        log('scan pull failed:', repr(e)[:120])
        return False

def live_until_end():
    if not LIVE.get('ctx'):
        return
    db, cfg, mine, types = LIVE['ctx']
    pulled = time.time()
    while time.time() < HARD['t'] - 75:
        live_poll()
        if time.time() - pulled > cfg.get('scan_pull_seconds', 240):
            pulled = time.time()
            if pull_scans():
                ingest_scans(db)
                check_deep(db, cfg, mine, types)
        time.sleep(max(1.0, cfg.get('live_poll_seconds', 15) - (time.time() - LIVE['last'])))
LIVE = {'ctx': None, 'last': 0, 'busy': False}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--check', type=int)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--data', help='folder holding config.json and the database (default: next to this file)')
    ap.add_argument('--shard', help='catalog scanner part, e.g. 3/10')
    a = ap.parse_args()
    if a.data:
        globals()['HERE'] = pathlib.Path(a.data).resolve()
    if os.environ.get('PUBLIC_LOG'):
        QUIET['on'] = True
    cfg = dict(DEFAULTS)
    cp = HERE / 'config.json'
    if cp.exists():
        cfg.update(json.loads(cp.read_text(encoding='utf-8')))
    for key, env in (('discord_webhook_url', 'DISCORD_WEBHOOK_URL'), ('anthropic_api_key', 'ANTHROPIC_API_KEY'), ('gemini_api_key', 'GEMINI_API_KEY'), ('feed_webhook_url', 'FEED_WEBHOOK_URL'), ('roblox_cookie', 'ROBLOX_COOKIE')):
        if os.environ.get(env):
            cfg[key] = os.environ[env].strip()
    if cfg['discord_webhook_url'].startswith('PASTE'):
        cfg['discord_webhook_url'] = ''
    if not cfg['my_creator_id']:
        sys.exit('Set my_creator_id in config.json')
    db = db_open()
    if a.shard:
        i, n = map(int, a.shard.split('/'))
        scan_shard(db, cfg, i, n)
        return
    if a.test:
        ok = send(cfg, {'username': 'UGC Watch', 'content': 'UGC Watch is connected.'})
        sys.exit(0 if ok else 'Webhook test failed')
    if a.selftest:
        selftest(db, cfg)
        return
    if a.check:
        if not load_mine(db):
            refresh_mine(db, cfg)
        mine = load_mine(db)
        j = http_json(f'https://economy.roblox.com/v2/assets/{a.check}/details') or {}
        cand = {'id': a.check, 'name': j.get('Name', ''), 'descr': j.get('Description', ''), 'created': j.get('Created', ''), 'atype': j.get('AssetTypeId')}
        u = thumb_urls([a.check]).get(a.check)
        feat, raw_png = fetch_feat(u) if isinstance(u, str) and u.startswith('http') else (None, None)
        res = judge(cand, feat, mine, cfg)
        if not res:
            sys.exit('nothing of yours is older than that item')
        real = dict(res)
        vm = res['visual_match']
        if vm and ai_provider(cfg):
            ai = ai_compare(cfg, db, fetch_feat(vm['thumb'])[1], raw_png, vm['name'], cand['name'])
            print('AI:', ai or 'no answer')
            real = apply_ai(real, ai, cfg)
        res['match'] = vm or res['match']
        print(f"{cand['name']}\n  closest of yours: {res['match']['name']} ({res['match']['id']})\n  visual {res['visual']:.2f} (shape {res['shape']:.2f}, detail {res['detail']:.2f}, mirrored {res['mirrored']})\n  name {res['name']:.2f}  description {res['desc']:.2f}  credit {res['credit']}\n  verdict: {real['verdict'] or 'no alert'}")
        return
    while True:
        if os.environ.get('GITHUB_EVENT_NAME') == 'schedule' and time.time() - kv_get(db, 'last_run', 0) < 40 * 60:
            log('ran less than 40 minutes ago, skipping this scheduled start')
            return
        kv_set(db, 'last_run', time.time())
        try:
            cycle(db, cfg)
        except Exception as e:
            log('cycle failed:', repr(e))
        if a.once:
            break
        time.sleep(cfg['poll_minutes'] * 60)
if __name__ == '__main__':
    main()
