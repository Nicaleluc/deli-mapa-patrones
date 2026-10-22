"""Mapa 2D de tickets coloreados por patron + K-NN por click (K=7).

Vista    : PCA(50) -> t-SNE(2, metric=cosine) sobre embeddings de tickets.
Decision : K-NN sobre los 7 tickets mas cercanos del mapa; para cada vecino se
           usa el patron mas cercano por coseno en 768D (precalculado), y el
           voto se pondera por rango. Tambien se informa el patron asignado
           (DS_Patron_Ticket) para contrastar.

Uso:
    py -u mapa_patrones.py                   # TODOS los patrones con tickets
    py -u mapa_patrones.py --patrones 18     # solo los 18 de mas volumen
    py -u mapa_patrones.py --refresh         # rehace muestra + proyeccion

Salidas: salida/mapa_patrones.html · datos/{base.npz, base_meta.json}
"""

import argparse
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from conexion import get_connection

DIM = 768
OUT = os.path.join(BASE_DIR, 'datos')       # base.npz + base_meta.json (cache)
SALIDA = os.path.join(BASE_DIR, 'salida')   # mapa_patrones.html
os.makedirs(OUT, exist_ok=True)
os.makedirs(SALIDA, exist_ok=True)

CAMPOS = """
    p.Id, p.FrecuenciaUso, p.Problema, p.Categoria, p.Subcategoria,
    p.Producto, p.Motivo, p.Prioridad, p.TipoGestion
"""


def _vec(v):
    if isinstance(v, (bytes, bytearray)):
        v = v.decode('utf-8', 'ignore')
    arr = np.asarray(json.loads(v), dtype=np.float32)
    return arr if arr.shape[0] == DIM else None


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _meta_patrones(cur):
    cur.execute(f"SELECT {CAMPOS} FROM DS_Patron p WITH (NOLOCK) WHERE p.Estado = 'activo'")
    meta = {}
    for pid, frec, prob, cat, sub, prod, mot, pri, tg in cur.fetchall():
        meta[int(pid)] = dict(frec=int(frec or 0), problema=prob or '', categoria=cat or '',
                              subcategoria=sub or '', producto=prod or '', motivo=mot or '',
                              prioridad=pri or '', tipogestion=tg or '')
    return meta


def _vectores_patrones(cur):
    cur.execute(f"""
        SELECT {CAMPOS}, pe.Vector
        FROM DS_Patron p WITH (NOLOCK)
        JOIN DS_Patron_Embedding pe WITH (NOLOCK) ON pe.Id = p.EmbeddingId
        WHERE p.Estado = 'activo'
    """)
    meta, vecs, ids = {}, [], []
    for pid, frec, prob, cat, sub, prod, mot, pri, tg, v in cur.fetchall():
        arr = _vec(v)
        if arr is None:
            continue
        ids.append(int(pid))
        vecs.append(arr)
        meta[int(pid)] = dict(frec=int(frec or 0), problema=prob or '', categoria=cat or '',
                              subcategoria=sub or '', producto=prod or '', motivo=mot or '',
                              prioridad=pri or '', tipogestion=tg or '')
    return np.array(ids, dtype=np.int32), _norm(np.vstack(vecs)), meta


def construir_base(cur, top_patrones, por_patron, min_ticket=400000):
    ids_all, P_all, meta_all = _vectores_patrones(cur)
    print(f"[base] {len(ids_all)} patrones activos con embedding")

    cur.execute("""
        SELECT pt.PatronId, COUNT(*) n
        FROM DS_Patron_Ticket pt WITH (NOLOCK)
        JOIN DS_Patron p WITH (NOLOCK) ON p.Id = pt.PatronId AND p.Estado = 'activo'
        WHERE pt.TicketId >= ?
        GROUP BY pt.PatronId ORDER BY COUNT(*) DESC
    """, min_ticket)
    ranking = cur.fetchall()
    if top_patrones <= 0:
        elegidos = [int(r[0]) for r in ranking if int(r[0]) in meta_all]
    else:
        elegidos = [int(r[0]) for r in ranking[:top_patrones] if int(r[0]) in meta_all]
    print(f"[base] {len(elegidos)} patrones se dibujan, de {len(ranking)} con tickets")

    marks = ','.join('?' * len(elegidos))
    tope = por_patron if por_patron and por_patron > 0 else 2147483647
    cur.execute(f"""
        SELECT TicketId, PatronId FROM (
            SELECT pt.TicketId, pt.PatronId,
                   ROW_NUMBER() OVER (PARTITION BY pt.PatronId ORDER BY pt.TicketId DESC) rn
            FROM DS_Patron_Ticket pt WITH (NOLOCK)
            WHERE pt.PatronId IN ({marks}) AND pt.TicketId >= ?
        ) x WHERE rn <= ?
    """, *elegidos, min_ticket, tope)
    pares = [(int(t), int(p)) for t, p in cur.fetchall()]
    print(f"[base] {len(pares)} pares (ticket, patron) muestreados "
          f"(hasta {por_patron if por_patron else 'sin tope'} por patron, tickets >= {min_ticket})")

    ids_unicos = sorted({t for t, _ in pares})
    vectores = {}
    LOTE = 400
    for i in range(0, len(ids_unicos), LOTE):
        lote = ids_unicos[i:i + LOTE]
        cur.execute(
            f"SELECT TicketId, Embedding FROM DS_Ticket_Embedding WITH (NOLOCK) "
            f"WHERE Tipo = 'contexto' AND TicketId IN ({','.join('?' * len(lote))})", *lote)
        for tid, v in cur.fetchall():
            arr = _vec(v)
            if arr is not None:
                vectores[int(tid)] = arr
        print(f"\r[base] vectores {len(vectores)}/{len(ids_unicos)}", end='', flush=True)
    print()

    mejor = {}
    for tid, pid in pares:
        if tid not in vectores:
            continue
        if tid not in mejor or meta_all[pid]['frec'] > meta_all[mejor[tid]]['frec']:
            mejor[tid] = pid

    tids = np.array(sorted(mejor), dtype=np.int64)
    pids = np.array([mejor[int(t)] for t in tids], dtype=np.int32)
    X = _norm(np.vstack([vectores[int(t)] for t in tids]))
    print(f"[base] matriz de tickets: {X.shape}")

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    todo = np.vstack([X, P_all])
    pca50 = PCA(n_components=50, random_state=0).fit(todo)
    var50 = float(pca50.explained_variance_ratio_.sum() * 100)
    print(f"[base] PCA 50 dims retiene {var50:.1f}% de la varianza; t-SNE...")
    ts = TSNE(n_components=2, perplexity=40, init='pca', random_state=0,
              max_iter=900, n_jobs=-1, metric='cosine')
    xy = ts.fit_transform(pca50.transform(X)).astype(np.float32)
    print("[base] t-SNE listo")

    np.savez_compressed(
        os.path.join(OUT, 'base.npz'),
        X=X, xy=xy, ticketIds=tids, patronIds=pids,
        P=P_all, patronVecIds=ids_all, var50=np.array([var50], dtype=np.float32),
        params=np.array([top_patrones, por_patron], dtype=np.int32),
        desde=np.array([min_ticket], dtype=np.int64))
    with open(os.path.join(OUT, 'base_meta.json'), 'w', encoding='utf-8') as fh:
        json.dump({str(k): v for k, v in meta_all.items()}, fh, ensure_ascii=False)
    return dict(X=X, xy=xy, tids=tids, pids=pids, P=P_all, pmid=ids_all,
                meta=meta_all, var50=var50, params=(top_patrones, por_patron))


def cargar_base(params, min_ticket=400000):
    ruta = os.path.join(OUT, 'base.npz')
    if not os.path.exists(ruta):
        return None
    d = np.load(ruta)
    if 'params' in d and tuple(int(x) for x in d['params']) != tuple(params[:2]):
        print(f"[base] cache con otros parametros {tuple(int(x) for x in d['params'])} "
              f"!= {tuple(params[:2])}; se rehace")
        return None
    if 'desde' in d and int(d['desde'][0]) != int(min_ticket):
        print(f"[base] cache armado con tickets >= {int(d['desde'][0])}; "
              f"pediste >= {min_ticket}; se rehace")
        return None
    return dict(X=d['X'], xy=d['xy'], tids=d['ticketIds'], pids=d['patronIds'],
                P=d['P'], pmid=d['patronVecIds'], meta={}, var50=float(d['var50'][0]))


def cargar_meta_json():
    ruta = os.path.join(OUT, 'base_meta.json')
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding='utf-8') as fh:
        return {int(k): v for k, v in json.load(fh).items()}



def construir_demo(n_patrones=18, n_tickets=2500, seed=7):
    """Dataset 100% sintetico: permite publicar el repo sin datos reales.

    Genera n_patrones centroides en 768d, les asigna tickets con ruido y
    proyecta igual que la base real (PCA50 + t-SNE coseno).
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(seed)
    centros = _norm(rng.normal(size=(n_patrones, DIM)).astype(np.float32))
    cluster = rng.integers(1, n_patrones + 1, n_tickets).astype(np.int32)
    # ruido normalizado: en 768d un gaussiano crudo tiene norma ~27 y tapa los clusters
    ruido = _norm(rng.normal(size=(n_tickets, DIM)).astype(np.float32))
    X = _norm(centros[cluster - 1] + 0.78 * ruido)
    # 8% de etiquetas cruzadas: el demo muestra el desacuerdo entre patrón asignado y vector
    pids = cluster.copy()
    cruz = rng.random(n_tickets) < 0.08
    if cruz.any():
        pids[cruz] = rng.integers(1, n_patrones + 1, int(cruz.sum())).astype(np.int32)
    tids = np.arange(1_000_001, 1_000_001 + n_tickets, dtype=np.int64)

    cats = ['Facturación', 'Caja / Tesorería', 'Stock', 'Integraciones', 'Usuarios']
    motivos = ['Error', 'Consulta', 'Configuración']
    prioridades = ['Alta', 'Media', 'Baja']
    gestiones = ['Tecnico', 'Sistema', 'Usuario', 'Externo']
    meta = {}
    for i in range(n_patrones):
        pid = i + 1
        meta[pid] = dict(
            frec=int((pids == pid).sum()),
            problema=f'Patrón sintético {pid:02d} — caso de ejemplo del demo',
            categoria=cats[i % len(cats)],
            subcategoria=f'Subcategoría {i % 4 + 1}',
            producto=f'Producto {i % 3 + 1}',
            motivo=motivos[i % len(motivos)],
            prioridad=prioridades[i % len(prioridades)],
            tipogestion=gestiones[i % len(gestiones)])

    todo = np.vstack([X, centros])
    pca50 = PCA(n_components=50, random_state=0).fit(todo)
    var50 = float(pca50.explained_variance_ratio_.sum() * 100)
    print(f"[demo] {n_tickets} tickets sintéticos · {n_patrones} patrones · PCA 50 dims {var50:.1f}%")
    ts = TSNE(n_components=2, perplexity=min(40, max(5, n_tickets // 60)), init='pca',
              random_state=0, max_iter=750, n_jobs=-1, metric='cosine')
    xy = ts.fit_transform(pca50.transform(X)).astype(np.float32)
    print("[demo] t-SNE listo")

    return dict(X=X, xy=xy, tids=tids, pids=pids, P=centros,
                pmid=np.arange(1, n_patrones + 1, dtype=np.int32),
                meta=meta, var50=var50)


def construir_html(base, archivo='mapa_patrones.html'):
    meta, X, xy, pids = base['meta'], base['X'], base['xy'], base['pids']
    P, pmid = base['P'], base['pmid']

    presentes = [int(p) for p in np.unique(pids) if int(p) in meta]
    presentes.sort(key=lambda p: -meta[p]['frec'])
    n_por_patron = {p: int((pids == p).sum()) for p in presentes}
    centro = {}
    for p in presentes:
        m = pids == p
        centro[p] = xy[m].mean(axis=0) if m.any() else np.array([0.0, 0.0])

    # para cada ticket: patron mas cercano por coseno en 768D (insumo del K-NN por click)
    sims = X @ P.T
    orden = np.argsort(-sims, axis=1)
    idx_valido = np.array([p in meta for p in pids])
    coincide = float((pmid[orden[idx_valido, 0]] == pids[idx_valido]).mean()) if idx_valido.any() else 0.0

    from sklearn.manifold import trustworthiness
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(7)
    idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
    trust = float(trustworthiness(X[idx], xy[idx], n_neighbors=10))
    sil2 = float(silhouette_score(xy[idx], pids[idx]))
    sil768 = float(silhouette_score(X[idx], pids[idx], metric='cosine'))

    puntos = []
    for j, (t, p) in enumerate(zip(base['tids'], pids)):
        if int(p) not in meta:
            continue
        np1 = int(pmid[orden[j, 0]])
        puntos.append({'i': int(t), 'p': int(p),
                       'x': round(float(xy[j, 0]), 2), 'y': round(float(xy[j, 1]), 2),
                       'np': np1, 'ns': round(float(sims[j, orden[j, 0]]), 3)})

    payload = dict(
        puntos=puntos,
        patrones=[dict(id=p, n=n_por_patron[p],
                       cx=round(float(centro[p][0]), 2), cy=round(float(centro[p][1]), 2),
                       problema=meta[p]['problema'][:120],
                       categoria=meta[p]['categoria'], subcategoria=meta[p]['subcategoria'],
                       producto=meta[p]['producto'], motivo=meta[p]['motivo'],
                       prioridad=meta[p]['prioridad'], tipogestion=meta[p]['tipogestion'],
                       frec=meta[p]['frec']) for p in presentes],
        metricas=dict(coincide=round(coincide, 4), trust=round(trust, 4),
                      sil2=round(sil2, 4), sil768=round(sil768, 4),
                      var50=round(base['var50'], 1),
                      n=len(puntos), patrones=len(presentes),
                      patrones_total=len(pmid)))

    ruta = os.path.join(SALIDA, archivo)
    with open(ruta, 'w', encoding='utf-8') as fh:
        fh.write(PLANTILLA.replace('__DATA__', json.dumps(payload, ensure_ascii=False)))
    print(f"[html] {os.path.abspath(ruta)}")
    print(f"[metricas] puntos={len(puntos)} · patrones dibujados={len(presentes)}/{len(pmid)} · "
          f"trust={trust:.3f} · silhouette 2D={sil2:.3f} vs 768D={sil768:.3f} · "
          f"asignado==mas cercano {coincide*100:.1f}%")
    return ruta


PLANTILLA = r"""<!DOCTYPE html>
<html lang="es-AR"><head><meta charset="utf-8"><title>Mapa de patrones — K-NN</title>
<style>
:root{--bg:#f6f7fb;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:13.5px/1.45 "Segoe UI",system-ui,sans-serif}
h1{font-size:18px;margin:0 0 3px}
.sub{color:var(--muted);font-size:12px;margin-bottom:10px}
.wrap{display:grid;grid-template-columns:392px minmax(0,1fr);gap:12px;align-items:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:11px 12px}
.stage{position:relative}
canvas{width:100%;height:735px;display:block;cursor:crosshair;border-radius:8px}
canvas.drag{cursor:grabbing}
h2{font-size:11.5px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin:0 0 8px}
.kpi{display:flex;justify-content:space-between;gap:8px;font-size:11.5px;padding:2px 0;border-bottom:1px dashed #eef2f7}
.kpi b{font-variant-numeric:tabular-nums;text-align:right}
.tip{position:fixed;pointer-events:none;background:#0f172a;color:#fff;padding:6px 9px;border-radius:7px;
  font-size:11.5px;line-height:1.4;opacity:0;transition:opacity .08s;z-index:9;max-width:330px}
button,select,input{font:inherit;font-size:12px;padding:3px 8px;border:1px solid var(--line);background:#fff;border-radius:7px;color:var(--ink)}
button{cursor:pointer}button:hover{background:#f1f5f9}
button.mini{font-size:10.5px;padding:1px 6px;border-radius:5px}
button.solo{font-size:9.5px;padding:0 5px;border-radius:4px;color:#475569;line-height:15px}
.warn{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:7px 9px;font-size:11px;color:#9a3412;margin-top:8px}
.badge{display:inline-block;background:#eef2ff;color:#3730a3;border-radius:5px;padding:0 5px;font-size:10px}
.tools{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:8px}
.hint{display:block;margin-top:6px;font-size:10.5px;color:#94a3b8;line-height:1.5}
.filtros{max-height:calc(100vh - 190px);overflow:auto;padding-right:2px}
details.fac{border-top:1px solid #eef2f7;padding:5px 0}
details.fac:first-of-type{border-top:0}
details.fac>summary{cursor:pointer;font-weight:600;font-size:12px;padding:3px 0;display:flex;justify-content:space-between;align-items:center;gap:6px}
details.fac>summary::-webkit-details-marker{display:none}
details.fac>summary:before{content:'▸';color:#94a3b8;font-size:10px;margin-right:2px}
details.fac[open]>summary:before{content:'▾'}
details.fac>summary .cnt{font-weight:400;color:var(--muted);font-size:10.5px;margin-left:auto}
.facBtns{display:flex;gap:5px;margin:4px 0 5px}
.facList{max-height:190px;overflow:auto}
.it{display:flex;align-items:center;gap:6px;font-size:11.5px;padding:1px 3px;border-radius:5px}
.it:hover{background:#f8fafc}
.it input{margin:0;accent-color:#2563eb}
.it-l{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.it-c{color:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}
.dot{width:8px;height:8px;border-radius:50%;flex:0 0 auto}
table{width:100%;border-collapse:collapse;font-size:11.5px}
td,th{padding:4px 2px;border-bottom:1px solid #eef2f7;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:10.5px;text-transform:uppercase}
.mono{font-variant-numeric:tabular-nums}
.sum{color:var(--muted);font-size:11px;padding:4px 0 2px}
.knn{position:absolute;top:10px;right:10px;width:322px;max-height:430px;overflow:auto;
  background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 11px;
  box-shadow:0 8px 24px rgba(15,23,42,.12);z-index:5}
.knnh{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.knnh b{font-size:12px}
.sug{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px;margin-bottom:8px}
.sug .big{font-size:15px;font-weight:700;display:flex;align-items:center;gap:6px}
.sug .why{font-size:11px;color:var(--muted);margin-top:3px}
@media(max-width:1150px){.wrap{grid-template-columns:1fr}.filtros{max-height:none}}
</style></head><body>
<h1>Mapa de tickets por patrón — vista 2D (t-SNE) + K-NN en 768D</h1>
<div class="sub" id="sub"></div>
<div class="wrap">
  <div>
    <div class="card" style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <h2 style="margin:0">Filtros</h2>
        <button id="limpiar" class="mini">limpiar todo</button>
      </div>
      <div class="sum" id="resumen"></div>
      <div class="filtros" id="filtros"></div>
    </div>
    <div class="card">
      <details><summary style="cursor:pointer;font-size:11.5px;text-transform:uppercase;letter-spacing:.5px;color:#64748b;font-weight:600">Calidad del mapa</summary>
        <div id="metricas" style="margin-top:8px"></div>
        <div class="warn">La vista 2D preserva <b>vecindad</b>, no distancia. Los patrones candidatos se calculan por coseno sobre los 768 valores.</div>
      </details>
    </div>
  </div>

  <div class="card stage">
    <canvas id="cv"></canvas>
    <div id="knnCard" class="knn" style="display:none">
      <div class="knnh"><b id="knnTit">K-NN · K=7</b><button id="knnClose" class="mini">×</button></div>
      <div id="knnBody"></div>
    </div>
    <div class="tools">
      <button id="reset">Encuadrar</button>
      <button id="zoomIn">+</button>
      <button id="zoomOut">−</button>
      <button id="quitarPunto">quitar punto</button>
      <button id="zoomPunto">zoom al punto</button>
      <label class="sub" style="margin:0 0 0 6px">K
        <input id="kIn" type="number" min="1" max="60" step="1" value="7" style="width:56px;text-align:center">
      </label>
      <label class="sub" style="margin:0 0 0 6px" title="similitud mínima: 1 = solo el más cercano · 0 = todos los K">umbral
        <input id="umbralIn" type="number" min="0" max="1" step="0.05" value="0.8" style="width:64px;text-align:center">
      </label>
      <span class="sub" style="margin:0 0 0 6px">puntos</span>
      <button id="ptMinus">−</button>
      <span id="ptVal" class="sub" style="margin:0;min-width:26px;text-align:center">2,4</span>
      <button id="ptPlus">+</button>
      <span style="flex:1"></span>
      <label class="sub" style="margin:0">color:
        <select id="color">
          <option value="patron">por patrón</option>
          <option value="categoria">por categoría</option>
          <option value="producto">por producto</option>
          <option value="prioridad">por prioridad</option>
        </select>
      </label>

    </div>
    <div class="hint"><b>clic en el mapa = poner un punto y calcular K vecinos</b> · umbral = similitud mínima (1 = solo el más cercano · 0 = todos los K) · rueda = zoom · arrastre = mover · ejes en unidades t-SNE (adimensionales)</div>
  </div>
</div>
<div class="tip" id="tip"></div>
<script>
const D = __DATA__;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d'), tip=document.getElementById('tip');
let W=0,H=0,esc=1,ox=0,oy=0,hover=null,drag=null,puntoPx=2.4;
let puntoKNN=null, resKNN=null;
let K=7, UMBRAL=0.8;
const dpr=Math.min(2, window.devicePixelRatio||1);

// ---------- colores ----------
const PAL=['#2563eb','#dc2626','#059669','#d97706','#7c3aed','#0891b2','#db2777',
           '#65a30d','#ea580c','#4f46e5','#0d9488','#b45309','#9333ea','#475569'];
const MAPA_COLOR={};
function coloresPor(campo){
  if(!MAPA_COLOR[campo]){
    const vals=[...new Set(D.patrones.map(p=>p[campo]||'(sin dato)'))].sort();
    const m={}; vals.forEach((v,i)=>{m[v]=PAL[i%PAL.length];});
    MAPA_COLOR[campo]=m;
  }
  return MAPA_COLOR[campo];
}
D.patrones.forEach((p,i)=>{
  const hue=(i*137.508+i*0.7)%360, light=(i%3===0)?44:(i%3===1?52:60);
  p.colorPat=`hsl(${hue.toFixed(0)} 68% ${light}%)`;
});
const byId=new Map(D.patrones.map(p=>[p.id,p]));
let colorMode='patron';
function col(p){
  if(colorMode==='patron') return p.colorPat;
  return coloresPor(colorMode)[p[colorMode]||'(sin dato)'];
}

// ---------- filtros facetados ----------
const FACETS=[
 {k:'categoria',  t:'Categorías',   get:p=>p.categoria||'(sin categoría)'},
 {k:'subcategoria',t:'Subcategoría',get:p=>p.subcategoria||'(sin subcategoría)'},
 {k:'producto',   t:'Producto',     get:p=>p.producto||'(sin producto)'},
 {k:'motivo',     t:'Motivo',       get:p=>p.motivo||'(sin motivo)'},
 {k:'prioridad',  t:'Prioridad',    get:p=>p.prioridad||'(sin prioridad)'},
 {k:'tipogestion',t:'TipoGestión',  get:p=>p.tipogestion||'(sin tipo)'},
 {k:'patron',     t:'Patrón',       get:p=>'#'+p.id},
];
const sel={}, cnt={};
FACETS.forEach(f=>{
  const c=new Map();
  D.patrones.forEach(p=>{const v=f.get(p); c.set(v,(c.get(v)||0)+1);});
  cnt[f.k]=c; sel[f.k]=new Set(c.keys());
});
const ORDEN_PRI={'Urgente':0,'Alta':1,'Media':2,'Baja':3};
function valores(f){
  const arr=[...cnt[f.k].keys()];
  if(f.k==='prioridad') return arr.sort((a,b)=>(ORDEN_PRI[a]??9)-(ORDEN_PRI[b]??9));
  if(f.k==='patron') return arr.sort((a,b)=>parseInt(a.slice(1))-parseInt(b.slice(1)));
  return arr.sort((a,b)=>cnt[f.k].get(b)-cnt[f.k].get(a)||a.localeCompare(b));
}
let visPat=new Set();
function patronVisible(p){
  for(const f of FACETS) if(!sel[f.k].has(f.get(p))) return false;
  return true;
}
function recomputar(){
  visPat=new Set(D.patrones.filter(patronVisible).map(p=>p.id));
  const puntosVis=D.puntos.reduce((a,p)=>a+(visPat.has(p.p)?1:0),0);
  document.getElementById('resumen').innerHTML=
    `<b>${visPat.size}</b> de ${D.patrones.length} patrones · <b>${puntosVis}</b> de ${D.puntos.length} tickets`;
  FACETS.forEach(f=>{
    const el=document.getElementById('cnt_'+f.k);
    if(el) el.textContent=`${sel[f.k].size}/${cnt[f.k].size}`;
  });
  if(puntoKNN){ resKNN=calcularKNN(puntoKNN); pintarTarjetaKNN(); }
  draw();
}

const cont=document.getElementById('filtros');
FACETS.forEach(f=>{
  const det=document.createElement('details');
  det.className='fac'; det.open=(f.k!=='patron');
  const sum=document.createElement('summary');
  sum.innerHTML=`<span>${f.t}</span><span class="cnt" id="cnt_${f.k}"></span>`;
  det.appendChild(sum);

  const btns=document.createElement('div'); btns.className='facBtns';
  const bAll=document.createElement('button'); bAll.className='mini'; bAll.textContent='seleccionar todo';
  const bNone=document.createElement('button'); bNone.className='mini'; bNone.textContent='ninguno';
  btns.append(bAll,bNone);
  if(f.k==='patron'){
    const inp=document.createElement('input');
    inp.id='buscar'; inp.placeholder='buscar #id o texto'; inp.style.flex='1'; inp.style.minWidth='90px';
    btns.appendChild(inp);
    inp.oninput=()=>pintarLista(f);
  }
  det.appendChild(btns);
  const lista=document.createElement('div'); lista.className='facList'; lista.id='lista_'+f.k;
  det.appendChild(lista);
  cont.appendChild(det);

  bAll.onclick=()=>{ sel[f.k]=new Set(cnt[f.k].keys()); pintarLista(f); };
  bNone.onclick=()=>{ sel[f.k]=new Set(); pintarLista(f); };
  pintarLista(f);
});

function pintarLista(f){
  const lista=document.getElementById('lista_'+f.k);
  const q=(f.k==='patron' && document.getElementById('buscar'))
            ? document.getElementById('buscar').value.trim().toLowerCase() : '';
  lista.innerHTML='';
  valores(f).forEach(v=>{
    if(f.k==='patron' && q){
      const pr=D.patrones.find(x=>'#'+x.id===v);
      if(!(v.includes(q) || (pr&&pr.problema.toLowerCase().includes(q)))) return;
    }
    const p=f.k==='patron'?D.patrones.find(x=>'#'+x.id===v):null;
    const it=document.createElement('div'); it.className='it';
    const cb=document.createElement('input'); cb.type='checkbox'; cb.checked=sel[f.k].has(v);
    cb.onchange=()=>{ if(cb.checked) sel[f.k].add(v); else sel[f.k].delete(v); recomputar(); };
    const dot=document.createElement('span'); dot.className='dot';
    dot.style.background = p ? col(p) : '#cbd5e1';
    const lab=document.createElement('span'); lab.className='it-l';
    lab.textContent=v; if(p) lab.title=p.problema;
    const c=document.createElement('span'); c.className='it-c';
    c.textContent = f.k==='patron' ? `${p?p.n:0} tk` : cnt[f.k].get(v);
    const solo=document.createElement('button'); solo.className='solo'; solo.textContent='solo';
    solo.onclick=e=>{ e.preventDefault(); e.stopPropagation();
      sel[f.k]=new Set([v]); pintarLista(f); };
    it.append(cb,dot,lab,c,solo);
    lista.appendChild(it);
  });
  recomputar();
}

// ---------- K-NN por click ----------
// Voto PRINCIPAL: patron ASIGNADO de cada vecino (vínculo ticket-patrón), ponderado por rango.
// Voto SECUNDARIO: patron mas cercano por coseno 768D de cada vecino (lo que diria el matcher).
function calcularKNN(pt){
  const cand=[];
  D.puntos.forEach(p=>{
    if(!visPat.has(p.p)) return;
    const dx=p.x-pt.x, dy=p.y-pt.y;
    cand.push({p, d:Math.sqrt(dx*dx+dy*dy)});
  });
  if(!cand.length) return null;
  cand.sort((a,b)=>a.d-b.d);
  const k=Math.max(1, Math.min(K, cand.length));
  const kMasCercanos=cand.slice(0,k);
  const dmin=kMasCercanos[0].d;                        // distancia al mas cercano
  const distK=kMasCercanos[kMasCercanos.length-1].d;   // distancia al K-esimo
  const span=distK-dmin;
  // similitud relativa: 1 = el mas cercano, 0 = el K-esimo. umbral alto = conjunto chico.
  const corte=(span<1e-9) ? distK : dmin + (1-UMBRAL)*span;
  const vec=kMasCercanos.filter(v=>v.d<=corte+1e-9);
  const descartados=kMasCercanos.length-vec.length;
  const cercaADist=cand[0].d;

  if(!vec.length){
    return {pt, vec:[], candidatos:[], cosenos:[], totalAsig:1, totalCos:1,
            k, umbral:UMBRAL, corte, dmin, distK, descartados, distCercano:cercaADist,
            distMax:null, distMin:null};
  }

  const vAsig=new Map(), vCos=new Map(), accCos=new Map();
  vec.forEach((v,idx)=>{
    const w=1/(idx+1);                                   // 1, 1/2, 1/3...
    vAsig.set(v.p.p, (vAsig.get(v.p.p)||0)+w);
    const c=v.p.np, sim=v.p.ns||0;
    vCos.set(c, (vCos.get(c)||0)+w*sim);
    const a=accCos.get(c)||{sum:0,n:0}; a.sum+=sim; a.n++; accCos.set(c,a);
  });
  const ordenar=(m, campo)=>[...m.entries()].map(([id,w])=>{
    const a=accCos.get(id);
    return {id, w, cos:(a?a.sum/a.n:null), votos:vec.filter(v=>v.p[campo]===id).length};
  }).sort((a,b)=>b.w-a.w);

  const totalAsig=vec.reduce((a,_,i)=>a+1/(i+1),0)||1;
  const totalCos=[...vCos.values()].reduce((a,b)=>a+b,0)||1;
  return {pt, vec, k, umbral:UMBRAL, corte, dmin, distK, descartados, distCercano:cercaADist,
          candidatos:ordenar(vAsig,'p'),        // principal
          cosenos:ordenar(vCos,'np').slice(0,5), // referencia 768D
          totalAsig, totalCos,
          distMax:vec[vec.length-1].d, distMin:vec[0].d};
}

function pintarTarjetaKNN(){
  const card=document.getElementById('knnCard'), body=document.getElementById('knnBody');
  if(!resKNN){
    card.style.display='block';
    body.innerHTML='<div class="sub">No hay tickets visibles con los filtros actuales.</div>';
    return;
  }
  card.style.display='block';
  const {pt,candidatos,cosenos,vec,totalAsig,totalCos,k,distMax,distMin,umbral,corte,dmin,distK,descartados,distCercano}=resKNN;
  if(!vec.length){
    document.getElementById('knnTit').textContent='K-NN · K='+k;
    body.innerHTML=`<div class="sug"><div class="big">Sin vecinos</div>
      <div class="why">Con umbral ${umbral.toFixed(2)} el corte queda en ${corte.toFixed(2)} u
        y el más cercano está a <b>${distCercano.toFixed(2)}</b>.</div>
      <div class="why">Bajá el umbral (0 = todos los K) o subí K.</div></div>`;
    return;
  }
  const top=candidatos[0], seg=candidatos[1];
  document.getElementById('knnTit').textContent='K-NN · K='+k;
  const pat=byId.get(top.id);
  const pct=top.w/totalAsig*100;
  const segPct=seg?seg.w/totalAsig*100:0;
  const ventaja=seg&&seg.w?top.w/seg.w:99;
  const coinc=vec.filter(v=>v.p.np===top.id).length;
  const cosTop=cosenos[0]||null, patCos=cosTop?byId.get(cosTop.id):null;
  body.innerHTML=`
    <div class="sug">
      <div class="big"><span class="dot" style="width:11px;height:11px;background:${pat?col(pat):'#999'}"></span>
        ${pat?'#'+pat.id:'#'+top.id}</div>
      <div style="font-size:11.5px;margin-top:2px">${pat?pat.categoria:'—'}</div>
      <div class="why">${pat?pat.problema.slice(0,110):''}${pat&&pat.problema.length>110?'…':''}</div>
      <div class="why"><b>${top.votos}/${vec.length}</b> vecinos con este patrón asignado · peso ${pct.toFixed(0)}%
        ${seg?` · 2º #${seg.id} (${segPct.toFixed(0)}%)`:''}</div>
      <div class="why">separación 1º/2º: ${ventaja>=9?'—':ventaja.toFixed(1)+'×'} ·
        coinciden por coseno: ${coinc}/${vec.length}</div>
    </div>
    <div class="sub" style="margin:0 0 4px">Vecinos por patrón asignado</div>
    <table><tr><th>Patrón</th><th>vecinos</th><th>peso</th></tr>
      ${candidatos.slice(0,5).map(c=>{const pp=byId.get(c.id);
        return `<tr><td><b style="color:${pp?col(pp):'#111'}">#${c.id}</b> ${pp?pp.categoria:''}</td>
        <td class="mono">${c.votos}/${vec.length}</td>
        <td class="mono">${(c.w/totalAsig*100).toFixed(0)}%</td></tr>`;}).join('')}
    </table>
    <div class="sub" style="margin:8px 0 4px">Lo que diría el matcher por vector (coseno 768D)</div>
    <table><tr><th>Patrón</th><th>peso</th><th>cos</th></tr>
      ${cosenos.map(c=>{const pp=byId.get(c.id);
        return `<tr><td><b>#${c.id}</b> ${pp?pp.categoria:''}</td>
        <td class="mono">${(c.w/totalCos*100).toFixed(0)}%</td>
        <td class="mono">${c.cos!=null?c.cos.toFixed(3):'—'}</td></tr>`;}).join('')}
    </table>
    <div class="sub" style="margin:8px 0 4px">Los ${vec.length} vecinos (distancia en el mapa)</div>
    <table><tr><th>Ticket</th><th>asignado</th><th>más cercano</th><th>d</th></tr>
      ${vec.map(v=>`<tr><td class="mono">${v.p.i}</td><td class="mono">#${v.p.p}</td>
        <td class="mono">#${v.p.np}</td><td class="mono">${v.d.toFixed(2)}</td></tr>`).join('')}
    </table>
    <div class="sub" style="margin-top:6px">d: ${distMin.toFixed(2)} a ${distMax.toFixed(2)} unidades t-SNE ·
      x=${pt.x.toFixed(1)} · y=${pt.y.toFixed(1)}</div>
    <div class="sub" style="margin-top:2px">umbral ${umbral.toFixed(2)} (similitud; 1 = el más cercano) ·
      ${vec.length} de ${vec.length+descartados} conectados hasta ${corte.toFixed(2)} u
      (distancias ${dmin.toFixed(2)}–${distK.toFixed(2)})</div>`;
}

// ---------- geometria / dibujo ----------
function fit(){
  const r=cv.getBoundingClientRect(); W=r.width; H=r.height;
  cv.width=W*dpr; cv.height=H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  resize(); draw();
}
function bounds(){
  let xs=[],ys=[];
  D.puntos.forEach(p=>{ if(visPat.has(p.p)){xs.push(p.x);ys.push(p.y);} });
  if(puntoKNN){xs.push(puntoKNN.x);ys.push(puntoKNN.y);}
  if(!xs.length){xs=[0,1];ys=[0,1];}
  const mnx=Math.min(...xs),mxx=Math.max(...xs),mny=Math.min(...ys),mxy=Math.max(...ys);
  return {cx:(mnx+mxx)/2,cy:(mny+mxy)/2,wx:(mxx-mnx)||1,wy:(mxy-mny)||1};
}
function resize(){
  const b=bounds(), pad=54;
  esc=Math.min((W-pad*2)/b.wx,(H-pad*2)/b.wy);
  ox=W/2-b.cx*esc; oy=H/2+b.cy*esc;
}
const tx=x=>ox+x*esc, ty=y=>oy-y*esc;
const invX=px=>(px-ox)/esc, invY=py=>(oy-py)/esc;
function niceStep(range,objetivo){
  const raw=Math.max(range,1e-9)/objetivo, mag=Math.pow(10,Math.floor(Math.log10(raw))), n=raw/mag;
  return (n<=1?1:n<=2?2:n<=5?5:10)*mag;
}
function fmtTick(v,step){
  const dec=step>=1?0:(step>=0.1?1:(step>=0.01?2:3));
  return (Math.abs(v)<1e-9?0:v).toFixed(dec);
}
function drawGrid(){
  const x0=invX(0), x1=invX(W), y1=invY(0), y0=invY(H);
  const sx=niceStep(x1-x0,9), sy=niceStep(y1-y0,7);
  ctx.font='10px Segoe UI'; ctx.textBaseline='alphabetic';
  for(let x=Math.ceil(x0/sx)*sx; x<=x1; x+=sx){
    const px=tx(x);
    ctx.strokeStyle=(Math.abs(x)<1e-9)?'#cbd5e1':'#e8edf5'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,H); ctx.stroke();
    ctx.fillStyle='#94a3b8'; ctx.textAlign='center'; ctx.fillText(fmtTick(x,sx),px,H-4);
  }
  for(let y=Math.ceil(y0/sy)*sy; y<=y1; y+=sy){
    const py=ty(y);
    ctx.strokeStyle=(Math.abs(y)<1e-9)?'#cbd5e1':'#e8edf5'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(0,py); ctx.lineTo(W,py); ctx.stroke();
    ctx.fillStyle='#94a3b8'; ctx.textAlign='right'; ctx.fillText(fmtTick(y,sy),W-6,py-3);
  }
  ctx.strokeStyle='#cbd5e1'; ctx.lineWidth=1; ctx.strokeRect(0.5,0.5,W-1,H-1);
  ctx.fillStyle='#64748b'; ctx.textAlign='left'; ctx.font='600 10.5px Segoe UI';
  ctx.fillText('t-SNE 1',8,H-4);
  ctx.save(); ctx.translate(12,16); ctx.rotate(Math.PI/2); ctx.fillText('t-SNE 2',0,0); ctx.restore();
}
function draw(){
  if(!W||!H) return;
  ctx.clearRect(0,0,W,H); ctx.fillStyle='#fff'; ctx.fillRect(0,0,W,H);
  drawGrid();
  ctx.globalAlpha=0.45;
  const r=puntoPx;
  D.puntos.forEach(p=>{
    if(!visPat.has(p.p)) return;
    const pat=byId.get(p.p); if(!pat) return;
    ctx.fillStyle=col(pat);
    ctx.fillRect(tx(p.x)-r/2, ty(p.y)-r/2, r, r);
  });
  ctx.globalAlpha=1;
  if(hover && hover.tipo==='punto'){
    const p=hover.obj; ctx.fillStyle='#111827';
    ctx.beginPath(); ctx.arc(tx(p.x),ty(p.y),Math.max(4.5,r*1.8),0,6.2832); ctx.fill();
  }
  // K-NN: vecinos + punto
  if(puntoKNN && resKNN){
    const px=tx(puntoKNN.x), py=ty(puntoKNN.y);
    if(resKNN.distK && resKNN.distK>0){
      ctx.save();
      ctx.strokeStyle='rgba(148,163,184,.55)'; ctx.lineWidth=1; ctx.setLineDash([3,4]);
      ctx.beginPath(); ctx.arc(px,py,resKNN.distK*esc,0,6.2832); ctx.stroke();
      ctx.restore();
      ctx.font='10px Segoe UI'; ctx.fillStyle='#94a3b8'; ctx.textAlign='left';
      ctx.fillText('K° '+resKNN.distK.toFixed(2)+' u', px + resKNN.distK*esc*0.72, py - resKNN.distK*esc*0.72);
    }
    if(resKNN.corte>0){
      ctx.save();
      ctx.strokeStyle='rgba(37,99,235,.55)'; ctx.lineWidth=1.4; ctx.setLineDash([6,5]);
      ctx.beginPath(); ctx.arc(px,py,resKNN.corte*esc,0,6.2832); ctx.stroke();
      ctx.restore();
      ctx.font='600 10px Segoe UI'; ctx.fillStyle='rgba(37,99,235,.9)'; ctx.textAlign='left';
      ctx.fillText('umbral '+resKNN.corte.toFixed(2)+' u', px + resKNN.corte*esc + 6, py - 6);
    }
    ctx.strokeStyle='rgba(17,24,39,.45)'; ctx.lineWidth=1.1; ctx.setLineDash([3,3]);
    resKNN.vec.forEach(v=>{
      ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(tx(v.p.x),ty(v.p.y)); ctx.stroke();
    });
    ctx.setLineDash([]);
    resKNN.vec.forEach(v=>{
      const pat=byId.get(v.p.np) || byId.get(v.p.p);
      ctx.strokeStyle=pat?col(pat):'#111827'; ctx.lineWidth=1.8;
      ctx.beginPath(); ctx.arc(tx(v.p.x),ty(v.p.y),5.5,0,6.2832); ctx.stroke();
    });
    const top=resKNN.candidatos[0];
    if(!top) return;
    const pt=byId.get(top.id);
    ctx.save(); ctx.translate(px,py); ctx.beginPath();
    for(let i=0;i<10;i++){const rr=i%2?5:12,a=-Math.PI/2+i*Math.PI/5;ctx.lineTo(Math.cos(a)*rr,Math.sin(a)*rr);}
    ctx.closePath(); ctx.fillStyle=pt?col(pt):'#111827'; ctx.fill();
    ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke(); ctx.restore();
    ctx.font='700 12px Segoe UI'; ctx.textAlign='left'; ctx.fillStyle='#111827';
    const et='→ #'+top.id;
    const w=ctx.measureText(et).width;
    ctx.fillStyle='rgba(255,255,255,.9)'; ctx.fillRect(px+15,py-11,w+6,16);
    ctx.fillStyle=pt?col(pt):'#111827'; ctx.fillText(et,px+18,py+2);
  }
}

function cerca(mx,my){
  let best=null,bd=Math.max(70, puntoPx*puntoPx*4);
  D.puntos.forEach(p=>{
    if(!visPat.has(p.p)) return;
    const dx=tx(p.x)-mx,dy=ty(p.y)-my,d=dx*dx+dy*dy;
    if(d<bd){bd=d;best={tipo:'punto',obj:p};}
  });
  return best;
}
cv.addEventListener('mousemove',e=>{
  const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  if(drag){
    if(!drag.moved && Math.hypot(mx-drag.x0,my-drag.y0)>4) drag.moved=true;
    if(drag.moved){
      ox-=drag.x-mx; oy-=drag.y-my; drag.x=mx; drag.y=my; draw(); return;
    }
  }
  const h=cerca(mx,my);
  const cambio=(!h&&hover)||(h&&!hover)||(h&&hover&&(h.obj!==hover.obj||h.tipo!==hover.tipo));
  hover=h;
  if(h&&h.tipo==='punto'){
    const p=h.obj, pat=byId.get(p.p), cer=byId.get(p.np);
    tip.innerHTML=`<b>Ticket ${p.i}</b><br>asignado: #${p.p} · ${pat?pat.categoria:''}<br>`+
      `<span style="opacity:.75">${pat?pat.problema.slice(0,95):''}…</span><br>`+
      `<span style="opacity:.6">más cercano por coseno: #${p.np} (${p.ns}) · x=${p.x} y=${p.y}</span>`;
    tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY+14)+'px'; tip.style.opacity=1;
  } else { tip.style.opacity=0; }
  if(cambio) draw();
});
cv.addEventListener('mouseleave',()=>{tip.style.opacity=0;if(hover){hover=null;draw();}});
cv.addEventListener('mousedown',e=>{
  const r=cv.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  drag={x0:mx,y0:my,x:mx,y:my,moved:false};
  cv.classList.add('drag');
});
window.addEventListener('mouseup',e=>{
  if(drag && !drag.moved){
    const r=cv.getBoundingClientRect();
    const mx=e.clientX-r.left, my=e.clientY-r.top;
    if(mx>=0 && my>=0 && mx<=W && my<=H){
      puntoKNN={x:invX(mx), y:invY(my)};
      resKNN=calcularKNN(puntoKNN);
      pintarTarjetaKNN();
      draw();
    }
  }
  drag=null; cv.classList.remove('drag');
});
cv.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  const k=e.deltaY<0?1.15:1/1.15;
  ox=mx-(mx-ox)*k; oy=my-(my-oy)*k; esc*=k; draw();
},{passive:false});
function zoom(k){const mx=W/2,my=H/2;ox=mx-(mx-ox)*k;oy=my-(my-oy)*k;esc*=k;draw();}
document.getElementById('zoomIn').onclick=()=>zoom(1.3);
document.getElementById('zoomOut').onclick=()=>zoom(1/1.3);
document.getElementById('reset').onclick=()=>{resize();draw();};
document.getElementById('zoomPunto').onclick=()=>{
  if(!puntoKNN) return;
  const ref=(resKNN && resKNN.distK) ? resKNN.distK : 2;
  esc=Math.min(W,H)*0.30/Math.max(ref,0.05);
  ox=W/2-puntoKNN.x*esc; oy=H/2+puntoKNN.y*esc;
  draw();
};
document.getElementById('quitarPunto').onclick=()=>{
  puntoKNN=null; resKNN=null;
  document.getElementById('knnCard').style.display='none';
  draw();
};
document.getElementById('knnClose').onclick=()=>{document.getElementById('knnCard').style.display='none';};
const kIn=document.getElementById('kIn');
function aplicarK(){
  let v=parseInt(kIn.value,10);
  if(!Number.isFinite(v) || v<1) v=1;
  const visibles=D.puntos.filter(p=>visPat.has(p.p)).length;
  v=Math.min(v, 60, Math.max(1,visibles));
  K=v; kIn.value=v;
  if(puntoKNN){ resKNN=calcularKNN(puntoKNN); pintarTarjetaKNN(); draw(); }
}
kIn.onchange=aplicarK;
const umbralIn=document.getElementById('umbralIn');
function aplicarUmbral(){
  let v=parseFloat(String(umbralIn.value).replace(',','.'));
  if(!Number.isFinite(v) || v<0) v=0;
  v=Math.min(v,1);
  UMBRAL=v; umbralIn.value=v;
  if(puntoKNN){ resKNN=calcularKNN(puntoKNN); pintarTarjetaKNN(); draw(); }
}
umbralIn.onchange=aplicarUmbral;
umbralIn.oninput=()=>{ const v=parseFloat(String(umbralIn.value).replace(',','.'));
  if(Number.isFinite(v)&&v>=0&&v<=1) aplicarUmbral(); };
kIn.oninput=()=>{ const v=parseInt(kIn.value,10); if(Number.isFinite(v)&&v>=1&&v<=60) aplicarK(); };
document.getElementById('color').onchange=e=>{
  colorMode=e.target.value;
  FACETS.forEach(f=>pintarLista(f));
  draw();
};
document.getElementById('limpiar').onclick=()=>{
  FACETS.forEach(f=>{sel[f.k]=new Set(cnt[f.k].keys());});
  if(document.getElementById('buscar')) document.getElementById('buscar').value='';
  FACETS.forEach(f=>pintarLista(f));
};

function pintarPunto(){
  document.getElementById('ptVal').textContent=puntoPx.toFixed(1).replace('.',',');
}
document.getElementById('ptPlus').onclick=()=>{puntoPx=Math.min(10,puntoPx+0.6);pintarPunto();draw();};
document.getElementById('ptMinus').onclick=()=>{puntoPx=Math.max(1,puntoPx-0.6);pintarPunto();draw();};
pintarPunto();

// metricas
const m=D.metricas;
document.getElementById('sub').textContent =
  `${m.n} tickets · ${m.patrones} patrones dibujados · K-NN contra ${m.patrones_total} patrones activos · PCA50 ${m.var50}% + t-SNE`;
document.getElementById('metricas').innerHTML=`
 <div class="kpi"><span>Puntos</span><b>${m.n}</b></div>
 <div class="kpi"><span>Patrones dibujados / total</span><b>${m.patrones} / ${m.patrones_total}</b></div>
 <div class="kpi"><span>Fidelidad de vecindad</span><b>${(m.trust*100).toFixed(1)}%</b></div>
 <div class="kpi"><span>Separación 2D (silhouette)</span><b>${m.sil2.toFixed(3)}</b></div>
 <div class="kpi"><span>Separación real 768D</span><b>${m.sil768.toFixed(3)}</b></div>
 <div class="kpi"><span>Patrón asignado = más cercano</span><b>${(m.coincide*100).toFixed(1)}%</b></div>`;

window.addEventListener('resize',fit);
fit();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patrones', type=int, default=0,
                    help='0 = todos los activos con tickets (default); N = top N por volumen')
    ap.add_argument('--tickets-por-patron', type=int, default=55)
    ap.add_argument('--desde', type=int, default=400000,
                    help='TicketId minimo a muestrear (400000 por defecto; 0 = todo el historial)')
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--offline', action='store_true',
                    help='regenera el HTML con la base cacheada, sin tocar la DB')
    ap.add_argument('--demo', action='store_true',
                    help='genera un mapa con datos sinteticos (no toca la DB ni usa datos reales)')
    args = ap.parse_args()

    if args.demo:
        construir_html(construir_demo(), 'demo.html')
        return

    params = (args.patrones, args.tickets_por_patron)

    if args.offline:
        base = cargar_base(params, args.desde)
        meta = cargar_meta_json()
        if base is None or meta is None:
            sys.exit('[offline] falta base.npz / base_meta.json; correr una vez sin --offline')
        base['meta'] = meta
        print(f"[offline] {len(base['tids'])} tickets · {len(meta)} patrones (cache)")
        construir_html(base)
        return

    conn = get_connection()
    cur = conn.cursor()
    try:
        base = None if args.refresh else cargar_base(params, args.desde)
        if base is None:
            base = construir_base(cur, args.patrones, args.tickets_por_patron, args.desde)
        else:
            print(f"[base] cache: {len(base['tids'])} tickets")

        # metadatos y vectores de patron siempre frescos desde la base
        ids_all, P_all, meta_all = _vectores_patrones(cur)
        base['P'], base['pmid'], base['meta'] = P_all, ids_all, meta_all
        print(f"[base] {len(ids_all)} patrones activos para K-NN")

        construir_html(base)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
