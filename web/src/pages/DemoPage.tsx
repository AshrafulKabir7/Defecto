import { useEffect, useMemo, useState } from 'react';
import { asset, defectCounts, pct, useIndex, useManifest, verdictOf, type Item, type Manifest, type Verdict } from '../data';

const SPEEDS = [
  { label: '1x', ms: 2600 },
  { label: '2x', ms: 1300 },
  { label: '4x', ms: 650 },
];

const VERDICT_TEXT: Record<Verdict, string> = { pass: 'PASS', review: 'REVIEW', fail: 'FAIL' };
const VERDICT_COLOR: Record<Verdict, string> = { pass: 'var(--c-pass)', review: 'var(--c-review)', fail: 'var(--c-fail)' };

/** Deterministic shuffle so the conveyor order is the same on every visit. */
function seededShuffle<T>(list: T[], seed: string): T[] {
  let h = 1779033703 ^ seed.length;
  for (let i = 0; i < seed.length; i++) h = Math.imul(h ^ seed.charCodeAt(i), 3432918353);
  const rand = () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    return ((h ^= h >>> 16) >>> 0) / 4294967296;
  };
  const out = [...list];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

export default function DemoPage() {
  const { data: index, error } = useIndex();
  const [category, setCategory] = useState<string | null>(null);
  useEffect(() => {
    if (!category && index?.length) setCategory(index[0].category);
  }, [index, category]);
  const { data: manifest } = useManifest(category);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Inspection demo</p>
          <h1 className="display">Watch parts come down the line</h1>
          <p className="lead">
            Real parts from the VisA dataset&apos;s held-out test split, replayed like a conveyor. Each part is checked the
            moment it arrives; defective ones light up where the flaw is.
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

      {error && <div className="notice">Could not load the demo data ({error}). Run <code>python ml/export_web.py</code>.</div>}
      {manifest && <Station key={manifest.category} manifest={manifest} />}
    </div>
  );
}

function Station({ manifest }: { manifest: Manifest }) {
  const model = manifest.model;
  const order = useMemo(() => seededShuffle(manifest.items, manifest.category), [manifest]);
  const [pos, setPos] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(0);
  const [showMask, setShowMask] = useState(!model);
  const [showHeat, setShowHeat] = useState(true);

  useEffect(() => {
    if (!playing) return;
    const id = window.setTimeout(() => setPos((p) => (p + 1) % order.length), SPEEDS[speed].ms);
    return () => window.clearTimeout(id);
  }, [playing, pos, speed, order.length]);

  // Preload the next part so the scan never shows a blank frame.
  useEffect(() => {
    const next = order[(pos + 1) % order.length];
    const img = new Image();
    img.src = asset(manifest.category, 'img', next.id);
  }, [pos, order, manifest.category]);

  const item = order[pos];
  const seen = order.slice(0, pos + 1);
  const verdict = model && item.score != null ? verdictOf(item.score, model.thresholds.fail, model.thresholds.review) : null;

  const tally = useMemo(() => {
    const t = { good: 0, bad: 0, pass: 0, review: 0, fail: 0, escapes: 0, falseRejects: 0 };
    for (const it of seen) {
      it.label === 'bad' ? t.bad++ : t.good++;
      if (model && it.score != null) {
        const v = verdictOf(it.score, model.thresholds.fail, model.thresholds.review);
        t[v]++;
        if (v === 'pass' && it.label === 'bad') t.escapes++;
        if (v === 'fail' && it.label === 'good') t.falseRejects++;
      }
    }
    return t;
  }, [seen, model]);
  const defects = defectCounts(seen);
  const maxDefect = Math.max(1, ...defects.map(([, n]) => n));

  const jump = (i: number) => {
    setPlaying(false);
    setPos(i);
  };

  return (
    <>
      {!model && (
        <div className="notice" style={{ marginBottom: 16 }}>
          <span aria-hidden="true">ℹ️</span>
          <div>
            <strong>Model results are not in yet.</strong> The parts and the red defect outlines are real: they come from the
            dataset&apos;s human annotations (ground truth). Anomaly scores, PASS / REVIEW / FAIL verdicts and heatmaps appear here
            after training: <code>ml/patchcore_lite.py</code> → <code>ml/export_web.py</code>.
          </div>
        </div>
      )}

      <div className="station">
        <section aria-label="Current part">
          <div className="viewer">
            <img src={asset(manifest.category, 'img', item.id)} alt={`${manifest.title}, test part ${item.id}`} />
            {item.mask && (
              <img className="overlay" src={asset(manifest.category, 'mask', item.id)} alt="" style={{ opacity: showMask ? 0.55 : 0 }} />
            )}
            {item.heat && (
              <img className="overlay" src={asset(manifest.category, 'heat', item.id)} alt="" style={{ opacity: showHeat ? 1 : 0 }} />
            )}
            <div key={item.id} className="scanline" aria-hidden="true" />
            <div className="viewer__badges">
              <span className="chip chip--sim" title="Parts are replayed from the dataset; there is no live camera.">
                Simulation · conveyor replay
              </span>
              {showMask && item.mask && <span className="chip chip--fail">Ground-truth defect</span>}
            </div>
            <div className="viewer__caption">
              <span>
                Part {pos + 1} of {order.length}
              </span>
              <span>{manifest.title}</span>
            </div>
          </div>

          <div className="controls">
            <button type="button" className="btn btn--light btn--icon" aria-label="Previous part" onClick={() => jump((pos - 1 + order.length) % order.length)}>
              ‹
            </button>
            <button type="button" className="btn btn--dark" onClick={() => setPlaying((p) => !p)} style={{ minWidth: 96 }}>
              {playing ? 'Pause' : 'Play'}
            </button>
            <button type="button" className="btn btn--light btn--icon" aria-label="Next part" onClick={() => jump((pos + 1) % order.length)}>
              ›
            </button>
            <div className="seg" role="group" aria-label="Speed">
              {SPEEDS.map((s, i) => (
                <button key={s.label} type="button" aria-pressed={speed === i} onClick={() => setSpeed(i)}>
                  {s.label}
                </button>
              ))}
            </div>
            <span className="spacer" />
            <label className="toggle">
              <input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} />
              Defect annotation
            </label>
            <label className={`toggle${model ? '' : ' is-disabled'}`} title={model ? undefined : 'Available after the model is evaluated'}>
              <input type="checkbox" checked={!!model && showHeat} disabled={!model} onChange={(e) => setShowHeat(e.target.checked)} />
              Model heatmap
            </label>
          </div>

          <div className="strip" aria-label="Inspected parts">
            {seen
              .map((it, i) => ({ it, i }))
              .slice(-40)
              .reverse()
              .map(({ it, i }) => (
                <button
                  key={it.id}
                  type="button"
                  className={`thumb${i === pos ? ' is-current' : ''}`}
                  onClick={() => jump(i)}
                  aria-label={`Part ${i + 1}: ${it.label === 'bad' ? 'defective' : 'good'}`}
                >
                  <img src={asset(manifest.category, 'img', it.id)} alt="" loading="lazy" />
                  <span className="thumb__flag" style={{ background: thumbColor(it, model) }} />
                </button>
              ))}
          </div>
        </section>

        <aside className="side" aria-label="Inspection result">
          <div className="card verdict">
            {verdict && model && item.score != null ? (
              <>
                <span className="small muted">Model verdict · {model.name}</span>
                <div className="verdict__big">
                  <span className="dot" style={{ background: VERDICT_COLOR[verdict] }} />
                  {VERDICT_TEXT[verdict]}
                </div>
                <span className="small muted num">Anomaly score {item.score.toFixed(3)}</span>
                <div className="meter" aria-hidden="true">
                  <div className="meter__fill" style={{ width: pct(Math.min(item.score, 1), 2), background: VERDICT_COLOR[verdict] }} />
                  <div className="meter__tick" style={{ left: pct(model.thresholds.review, 2) }} />
                  <div className="meter__tick" style={{ left: pct(model.thresholds.fail, 2) }} />
                </div>
                <span className="small muted num">
                  Review ≥ {model.thresholds.review.toFixed(2)} · Fail ≥ {model.thresholds.fail.toFixed(2)}
                </span>
                <a className="small" href={`#/inspect?c=${manifest.category}&id=${item.id}`} style={{ display: 'block', marginTop: 10, color: 'var(--c-accent-ink)' }}>
                  Re-run this part in your browser →
                </a>
              </>
            ) : (
              <>
                <span className="small muted">Model verdict</span>
                <div className="verdict__big" style={{ color: 'var(--c-faint)' }}>
                  <span className="dot" style={{ background: 'var(--c-line)' }} />
                  Pending
                </div>
                <span className="small muted">Appears once the model has been trained and evaluated.</span>
              </>
            )}
            <hr style={{ border: 0, borderTop: '1px solid var(--c-line)', margin: '14px 0 12px' }} />
            <span className="small muted">Dataset label (ground truth)</span>
            <p style={{ margin: '6px 0 0', display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {item.label === 'good' ? (
                <span className="chip chip--pass">Good part</span>
              ) : (
                <>
                  <span className="chip chip--fail">Defective</span>
                  {item.defects.map((d) => (
                    <span key={d} className="chip chip--neutral" style={{ textTransform: 'capitalize' }}>
                      {d}
                    </span>
                  ))}
                </>
              )}
            </p>
          </div>

          <div className="card">
            <p className="h3" style={{ fontSize: 15, marginBottom: 10 }}>
              This session
            </p>
            <div className="counters">
              <div className="counter">
                <b>{seen.length}</b>
                <span>Inspected</span>
              </div>
              {model ? (
                <>
                  <div className="counter">
                    <b style={{ color: 'var(--c-pass)' }}>{tally.pass}</b>
                    <span>Passed</span>
                  </div>
                  <div className="counter">
                    <b style={{ color: '#b7730a' }}>{tally.review}</b>
                    <span>Sent to review</span>
                  </div>
                  <div className="counter">
                    <b style={{ color: 'var(--c-fail)' }}>{tally.fail}</b>
                    <span>Rejected</span>
                  </div>
                  <div className="counter">
                    <b>{tally.escapes}</b>
                    <span>Missed defects</span>
                  </div>
                  <div className="counter">
                    <b>{tally.falseRejects}</b>
                    <span>Good parts rejected</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="counter">
                    <b style={{ color: 'var(--c-fail)' }}>{tally.bad}</b>
                    <span>Defective (label)</span>
                  </div>
                  <div className="counter">
                    <b style={{ color: 'var(--c-pass)' }}>{tally.good}</b>
                    <span>Good (label)</span>
                  </div>
                  <div className="counter">
                    <b>{seen.length ? pct(tally.good / seen.length, 0) : '–'}</b>
                    <span>Good share</span>
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="card">
            <p className="h3" style={{ fontSize: 15, marginBottom: 12 }}>
              Defect types seen
            </p>
            {defects.length ? (
              <div className="bars">
                {defects.slice(0, 6).map(([name, n]) => (
                  <div key={name} className="bar">
                    <span className="bar__label" title={name}>
                      {name}
                    </span>
                    <span className="bar__track">
                      <span className="bar__fill" style={{ width: pct(n / maxDefect, 1), display: 'block' }} />
                    </span>
                    <span className="num small">{n}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="small muted" style={{ margin: 0 }}>
                No defective parts yet.
              </p>
            )}
            <p className="small muted" style={{ margin: '12px 0 0' }}>
              From the dataset&apos;s annotations. One part can have several defect types.
            </p>
          </div>
        </aside>
      </div>
    </>
  );
}

function thumbColor(it: Item, model: Manifest['model']) {
  if (model && it.score != null) {
    return VERDICT_COLOR[verdictOf(it.score, model.thresholds.fail, model.thresholds.review)];
  }
  return it.label === 'bad' ? 'var(--c-fail)' : 'var(--c-pass)';
}
