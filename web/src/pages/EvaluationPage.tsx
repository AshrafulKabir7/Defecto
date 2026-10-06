import { useEffect, useMemo, useState } from 'react';
import { asset, confusionAt, defectCounts, pct, rocPoints, useIndex, useManifest, type Item, type Manifest } from '../data';

export default function EvaluationPage() {
  const { data: index } = useIndex();
  const [category, setCategory] = useState<string | null>(null);
  useEffect(() => {
    if (!category && index?.length) setCategory(index[0].category);
  }, [index, category]);
  const { data: manifest } = useManifest(category);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Evaluation report</p>
          <h1 className="display">Measured, not promised</h1>
          <p className="lead">
            Every number here comes from a held-out test split that no training or tuning step has seen. Move the threshold to see
            the trade-off between missed defects and good parts thrown away.
          </p>
        </div>
        {index && (
          <div className="seg" role="group" aria-label="Product">
            {index.map((c) => (
              <button key={c.category} type="button" aria-pressed={c.category === category} onClick={() => setCategory(c.category)}>
                {c.title.split(':')[0]}
              </button>
            ))}
          </div>
        )}
      </div>

      {manifest && (manifest.model ? <Report key={manifest.category} manifest={manifest} /> : <Pending manifest={manifest} />)}
    </div>
  );
}

/* ───────────── Before training: real dataset facts + protocol ───────────── */

function Pending({ manifest }: { manifest: Manifest }) {
  const bad = manifest.items.filter((i) => i.label === 'bad');
  const defects = defectCounts(bad);
  const max = Math.max(1, ...defects.map(([, n]) => n));
  const c = manifest.counts;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="notice">
        <span aria-hidden="true">ℹ️</span>
        <div>
          <strong>No model results for {manifest.title.split(':')[0]} yet.</strong> This page fills in automatically after running{' '}
          <code>ml/patchcore_lite.py</code> and <code>ml/export_web.py</code>. Below is the real dataset the model
          will be judged on.
        </div>
      </div>

      <div className="grid-4">
        <Stat label="Training images" value={c.train_good} hint="Good parts only. No defect photos are used for training." />
        <Stat label="Validation images" value={c.val} hint="Used only to fit the score scale and thresholds." />
        <Stat label="Test: good parts" value={c.test_good} hint="Held out. Never used for training or tuning." />
        <Stat label="Test: defective parts" value={c.test_bad} hint="Held out, with pixel-level defect masks." />
      </div>

      <div className="grid-2">
        <div className="card">
          <p className="h3">Defects in the test split</p>
          <p className="small muted" style={{ margin: '4px 0 16px' }}>
            {manifest.title} · from the dataset&apos;s human annotations
          </p>
          <div className="bars">
            {defects.map(([name, n]) => (
              <div key={name} className="bar" style={{ gridTemplateColumns: '160px minmax(0,1fr) 28px' }}>
                <span className="bar__label" title={name}>
                  {name}
                </span>
                <span className="bar__track">
                  <span className="bar__fill" style={{ width: pct(n / max), display: 'block' }} />
                </span>
                <span className="num small">{n}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <p className="h3">What the report will contain</p>
          <ul className="small" style={{ margin: '12px 0 0', paddingLeft: 18, lineHeight: 1.7, color: 'var(--c-ink-soft)' }}>
            <li>Image-level AUROC and average precision on the test split</li>
            <li>Pixel-level AUROC and AUPRO (how well the heatmap finds the defect area)</li>
            <li>Missed-defect and false-reject rates at the chosen threshold, projected per 10,000 parts</li>
            <li>A REVIEW band that sends borderline parts to a person instead of guessing</li>
            <li>The model&apos;s actual misses and false alarms, shown as images</li>
            <li>Measured inference time on the named hardware</li>
          </ul>
        </div>
      </div>

      <div className="card">
        <p className="h3">Evaluation protocol</p>
        <div className="table-wrap" style={{ marginTop: 10 }}>
          <table className="table">
            <tbody>
              <tr>
                <th>Dataset</th>
                <td>{manifest.dataset}</td>
              </tr>
              <tr>
                <th>Splits</th>
                <td>VisA&apos;s official split. Its test set is divided 40 / 60 (stratified, fixed seed) into validation and test.</td>
              </tr>
              <tr>
                <th>Thresholds</th>
                <td>
                  FAIL threshold = the score that rejects at most 5% of good validation parts. REVIEW threshold = the highest value
                  that still sends 98% of validation defects to REVIEW or FAIL.
                </td>
              </tr>
              <tr>
                <th>Reported numbers</th>
                <td>Test split only. Results on public data are not a promise for a buyer&apos;s parts, which need their own pilot.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: number | string; hint?: string }) {
  return (
    <div className="card stat">
      <p className="stat__label">{label}</p>
      <p className="stat__value">{value}</p>
      {hint && <p className="stat__hint">{hint}</p>}
    </div>
  );
}

/* ───────────── After training: the full report ───────────── */

function Report({ manifest }: { manifest: Manifest }) {
  const model = manifest.model!;
  const scored = manifest.items.filter((i) => i.score != null);
  const [threshold, setThreshold] = useState(model.thresholds.fail);
  const c = confusionAt(scored, threshold);
  const roc = useMemo(() => rocPoints(scored), [scored]);
  const t = model.test;

  const misses = scored.filter((i) => i.label === 'bad' && i.score! < threshold).sort((a, b) => a.score! - b.score!);
  const falseAlarms = scored.filter((i) => i.label === 'good' && i.score! >= threshold).sort((a, b) => b.score! - a.score!);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="grid-4">
        <Stat label="Image AUROC" value={t.image_auroc.toFixed(3)} hint="Ranking of defective vs good parts (1.0 = perfect)." />
        <Stat label="Average precision" value={t.image_ap.toFixed(3)} hint="Precision averaged over all thresholds." />
        <Stat label="Pixel AUROC" value={t.pixel_auroc?.toFixed(3) ?? '–'} hint="How well the heatmap separates defect pixels." />
        <Stat label="AUPRO @ 30% FPR" value={t['pixel_aupro_0.3']?.toFixed(3) ?? '–'} hint="Per-defect-region overlap, small defects count equally." />
      </div>

      <div className="card">
        <div className="explorer">
          <div>
            <p className="h3">Threshold explorer</p>
            <p className="small muted" style={{ margin: '4px 0 16px' }}>
              Parts scoring at or above the threshold are rejected. Default = {model.thresholds.fail.toFixed(2)}, set so at most 5%
              of good validation parts are rejected; uncertain parts below it go to the REVIEW band.
            </p>
            <label className="small" htmlFor="thr">
              Threshold <b className="num">{threshold.toFixed(3)}</b>
            </label>
            <input
              id="thr"
              className="slider"
              type="range"
              min={0}
              max={1}
              step={0.005}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
            <div className="grid-2" style={{ marginTop: 14, gap: 10 }}>
              <div className="counter">
                <b style={{ color: 'var(--c-fail)' }}>{pct(c.escapeRate)}</b>
                <span>Missed defects (escape rate)</span>
              </div>
              <div className="counter">
                <b>{pct(c.falseRejectRate)}</b>
                <span>Good parts rejected</span>
              </div>
            </div>
            <div className="table-wrap" style={{ marginTop: 14 }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Per 10,000 parts, if 5% are defective</th>
                    <th>Parts</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Defects shipped to customers</td>
                    <td className="num">{Math.round(500 * c.escapeRate).toLocaleString()}</td>
                  </tr>
                  <tr>
                    <td>Good parts scrapped</td>
                    <td className="num">{Math.round(9500 * c.falseRejectRate).toLocaleString()}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="metric-note">Projection from test-split rates; the 5% defect rate is an illustrative assumption.</p>
          </div>

          <div>
            <RocChart points={roc} current={[c.falseRejectRate, c.recall]} />
            <div className="confusion" style={{ marginTop: 12 }}>
              <div className="hd" />
              <div className="hd">Rejected</div>
              <div className="hd">Passed</div>
              <div className="hd">Defective</div>
              <div>
                <b style={{ color: 'var(--c-pass)' }}>{c.tp}</b>caught
              </div>
              <div>
                <b style={{ color: 'var(--c-fail)' }}>{c.fn}</b>missed
              </div>
              <div className="hd">Good</div>
              <div>
                <b>{c.fp}</b>false alarm
              </div>
              <div>
                <b>{c.tn}</b>correct pass
              </div>
            </div>
          </div>
        </div>
      </div>

      <Gallery title="Missed defects at this threshold" items={misses} category={manifest.category} empty="No defective part slips through at this threshold." />
      <Gallery title="Good parts rejected at this threshold" items={falseAlarms} category={manifest.category} empty="No good part is rejected at this threshold." />

      <div className="card">
        <p className="h3">Model and run</p>
        <div className="table-wrap" style={{ marginTop: 10 }}>
          <table className="table">
            <tbody>
              <tr>
                <th>Model</th>
                <td>
                  {model.train.implementation ?? model.name}
                  {model.train.backbone ? ` · ${model.train.backbone}` : ''}
                  {model.train.memory_bank ? ` · memory bank of ${model.train.memory_bank[0].toLocaleString()} patch embeddings` : ''}
                  {' · '}
                  <a href={`#/inspect?c=${manifest.category}`}>run it in your browser</a>
                </td>
              </tr>
              <tr>
                <th>Trained on</th>
                <td>
                  {model.train.train_images} good images
                  {model.train.train_seconds ? ` in ${Math.round(model.train.train_seconds)} s` : ''}
                  {model.train.gpu ? ` on ${model.train.gpu}` : ' on CPU'}
                </td>
              </tr>
              <tr>
                <th>Inference time</th>
                <td>
                  {model.latency_ms.mean} ms mean, {model.latency_ms.p95} ms p95 on {model.latency_ms.device}. {model.latency_ms.note}
                </td>
              </tr>
              <tr>
                <th>Test split</th>
                <td>
                  {t.n_good} good, {t.n_bad} defective ({manifest.dataset})
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function RocChart({ points, current }: { points: [number, number][]; current: [number, number] }) {
  const W = 320, H = 240, P = 30;
  const x = (v: number) => P + v * (W - P - 10);
  const y = (v: number) => H - P - v * (H - P - 10);
  const d = points.map(([fx, ty], i) => `${i ? 'L' : 'M'}${x(fx).toFixed(1)},${y(ty).toFixed(1)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="ROC curve" style={{ width: '100%', height: 'auto', background: '#fff', borderRadius: 12, border: '1px solid var(--c-line)' }}>
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="#d9d9d9" strokeDasharray="4 4" />
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(0)} stroke="#bbb" />
      <line x1={x(0)} y1={y(0)} x2={x(0)} y2={y(1)} stroke="#bbb" />
      <path d={d} fill="none" stroke="#000" strokeWidth="2" />
      <circle cx={x(current[0])} cy={y(current[1])} r="6" fill="var(--c-accent)" stroke="#000" strokeWidth="1.5" />
      <text x={x(0.5)} y={H - 6} textAnchor="middle" fontSize="11" fill="#8a8a8a">
        Good parts rejected (FPR)
      </text>
      <text x={10} y={y(0.5)} textAnchor="middle" fontSize="11" fill="#8a8a8a" transform={`rotate(-90 10 ${y(0.5)})`}>
        Defects caught (TPR)
      </text>
    </svg>
  );
}

function Gallery({ title, items, category, empty }: { title: string; items: Item[]; category: string; empty: string }) {
  return (
    <div className="card">
      <p className="h3" style={{ marginBottom: 12 }}>
        {title} <span className="muted num">({items.length})</span>
      </p>
      {items.length ? (
        <div className="gallery">
          {items.slice(0, 12).map((it) => (
            <figure key={it.id}>
              <div className="frame">
                <img src={asset(category, 'img', it.id)} alt="" loading="lazy" />
                {it.heat && <img src={asset(category, 'heat', it.id)} alt="" loading="lazy" />}
              </div>
              <figcaption className="num">
                Score {it.score!.toFixed(3)}
                {it.defects.length ? ` · ${it.defects.join(', ')}` : ''}
              </figcaption>
            </figure>
          ))}
        </div>
      ) : (
        <p className="small muted" style={{ margin: 0 }}>
          {empty}
        </p>
      )}
    </div>
  );
}
