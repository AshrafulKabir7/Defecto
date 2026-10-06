import { useEffect, useMemo, useRef, useState } from 'react';
import { asset, pct, useIndex, useManifest, verdictOf, type Item, type Verdict } from '../data';
import { heatmapDataUrl, inspect, loadBank, loadModel, type Backend, type BankMeta, type InspectionResult } from '../inference';

const VERDICT_TEXT: Record<Verdict, string> = { pass: 'PASS', review: 'REVIEW', fail: 'FAIL' };
const VERDICT_COLOR: Record<Verdict, string> = { pass: 'var(--c-pass)', review: 'var(--c-review)', fail: 'var(--c-fail)' };

type Source = { kind: 'dataset'; item: Item } | { kind: 'upload'; url: string; name: string };

function hashParams() {
  return new URLSearchParams(window.location.hash.split('?')[1] ?? '');
}

/** A fixed, balanced pick of held-out test parts: the first six good and six defective. */
function pickSamples(items: Item[]) {
  return [...items.filter((i) => i.label === 'bad').slice(0, 6), ...items.filter((i) => i.label === 'good').slice(0, 6)];
}

export default function InspectPage() {
  const { data: index } = useIndex();
  const [category, setCategory] = useState<string | null>(() => hashParams().get('c'));
  useEffect(() => {
    if (!category && index?.length) setCategory((index.find((c) => c.hasModel) ?? index[0]).category);
  }, [index, category]);
  const { data: manifest } = useManifest(category);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Live inspection</p>
          <h1 className="display">Run the model in your browser</h1>
          <p className="lead">
            The same anomaly detector that produced the evaluation report runs here on your device: ResNet18 features
            compared against a memory bank of good parts. Pick a held-out test part or upload a photo.
          </p>
        </div>
        {index && (
          <div className="seg" role="group" aria-label="Product">
            {index.map((c) => (
              <button
                key={c.category}
                type="button"
                aria-pressed={c.category === category}
                disabled={!c.hasModel}
                title={c.hasModel ? undefined : 'Not trained yet'}
                onClick={() => setCategory(c.category)}
              >
                {c.title.split(':')[0]}
              </button>
            ))}
          </div>
        )}
      </div>
      {manifest && category && <Inspector key={category} category={category} items={manifest.items} title={manifest.title} />}
    </div>
  );
}

function Inspector({ category, items, title }: { category: string; items: Item[]; title: string }) {
  const samples = useMemo(() => pickSamples(items), [items]);
  const [source, setSource] = useState<Source | null>(() => {
    const id = hashParams().get('id');
    const item = items.find((i) => i.id === id) ?? samples[0];
    return item ? { kind: 'dataset', item } : null;
  });
  const [meta, setMeta] = useState<BankMeta | null>(null);
  const [backend, setBackend] = useState<Backend | null>(null);
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [heat, setHeat] = useState<string | null>(null);
  const [status, setStatus] = useState<'loading' | 'running' | 'done' | 'error'>('loading');
  const [error, setError] = useState('');
  const [showHeat, setShowHeat] = useState(true);
  const [showMask, setShowMask] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const runId = useRef(0);

  useEffect(() => {
    let live = true;
    Promise.all([loadModel(), loadBank(category)])
      .then(([model, bank]) => {
        if (!live) return;
        setBackend(model.backend);
        setMeta(bank.meta);
      })
      .catch((e: unknown) => {
        if (!live) return;
        setStatus('error');
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      live = false;
    };
  }, [category]);

  const imageUrl = source ? (source.kind === 'dataset' ? asset(category, 'img', source.item.id) : source.url) : null;

  // The image may finish loading before the model; inspect as soon as both are ready.
  useEffect(() => {
    if (meta && imgRef.current?.complete) void run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta]);

  async function run() {
    const image = imgRef.current;
    if (!image || !image.complete || !image.naturalWidth) return;
    const id = ++runId.current;
    setStatus('running');
    setError('');
    try {
      // Dataset parts use the exported model input, so the browser sees the evaluation's pixels.
      let modelInput = image;
      if (source?.kind === 'dataset' && source.item.input) {
        modelInput = new Image();
        modelInput.src = asset(category, 'input', source.item.id);
        await modelInput.decode();
      }
      const next = await inspect(category, modelInput);
      if (id !== runId.current) return;
      setResult(next);
      setHeat(heatmapDataUrl(next.map, image.naturalWidth, image.naturalHeight));
      setStatus('done');
    } catch (e) {
      if (id !== runId.current) return;
      setStatus('error');
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function choose(next: Source) {
    if (source?.kind === 'upload' && next.kind !== 'upload') URL.revokeObjectURL(source.url);
    setResult(null);
    setHeat(null);
    setSource(next);
  }

  const verdict = result && meta ? verdictOf(result.score, meta.fail, meta.review) : null;
  const datasetItem = source?.kind === 'dataset' ? source.item : null;
  const pythonVerdict = datasetItem?.score != null && meta ? verdictOf(datasetItem.score, meta.fail, meta.review) : null;

  return (
    <>
      <div className="station">
        <section aria-label="Inspected part">
          <div className="viewer">
            {imageUrl && (
              <img
                ref={imgRef}
                src={imageUrl}
                alt={datasetItem ? `${title}, test part ${datasetItem.id}` : 'Uploaded part'}
                onLoad={() => meta && void run()}
              />
            )}
            {heat && <img className="overlay" src={heat} alt="" style={{ opacity: showHeat ? 1 : 0 }} />}
            {datasetItem?.mask && (
              <img className="overlay" src={asset(category, 'mask', datasetItem.id)} alt="" style={{ opacity: showMask ? 0.55 : 0 }} />
            )}
            {status === 'running' && <div className="scanline" aria-hidden="true" />}
            <div className="viewer__badges">
              <span className="chip chip--sim">
                {backend ? `Running on ${backend === 'webgpu' ? 'WebGPU (GPU)' : 'WebAssembly (CPU)'}` : 'Loading model…'}
              </span>
            </div>
            <div className="viewer__caption">
              <span>{datasetItem ? `Test part ${datasetItem.id}` : source?.kind === 'upload' ? source.name : ''}</span>
              <span>{title}</span>
            </div>
          </div>

          <div className="controls">
            <button type="button" className="btn btn--dark" disabled={!meta || status === 'running'} onClick={() => void run()}>
              {status === 'running' ? 'Inspecting…' : 'Inspect again'}
            </button>
            <button type="button" className="btn btn--light" onClick={() => fileRef.current?.click()}>
              Upload a photo
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              hidden
              aria-label="Upload a part photo"
              onChange={(e) => {
                const file = e.target.files?.[0];
                e.target.value = '';
                if (file) choose({ kind: 'upload', url: URL.createObjectURL(file), name: file.name });
              }}
            />
            <span className="spacer" />
            <label className="toggle">
              <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} />
              Model heatmap
            </label>
            <label className={`toggle${datasetItem?.mask ? '' : ' is-disabled'}`}>
              <input type="checkbox" checked={showMask && !!datasetItem?.mask} disabled={!datasetItem?.mask} onChange={(e) => setShowMask(e.target.checked)} />
              Ground truth
            </label>
          </div>

          <div className="strip" aria-label="Held-out test parts">
            {samples.map((it) => (
              <button
                key={it.id}
                type="button"
                className={`thumb${datasetItem?.id === it.id ? ' is-current' : ''}`}
                onClick={() => choose({ kind: 'dataset', item: it })}
                aria-label={`Test part ${it.id}: ${it.label === 'bad' ? 'defective' : 'good'}`}
              >
                <img src={asset(category, 'img', it.id)} alt="" loading="lazy" />
                <span className="thumb__flag" style={{ background: it.label === 'bad' ? 'var(--c-fail)' : 'var(--c-pass)' }} />
              </button>
            ))}
          </div>
          {source?.kind === 'upload' && (
            <div className="notice" style={{ marginTop: 14 }}>
              <span aria-hidden="true">ℹ️</span>
              <div>
                This model only knows what good <strong>{title.split(':')[0]}</strong> parts look like under the VisA camera
                setup. Other objects, angles, backgrounds or lighting will score as anomalous: that is how anomaly detection
                is meant to behave. A production model is trained on photos from your own station.
              </div>
            </div>
          )}
        </section>

        <aside className="side" aria-label="Inspection result">
          <div className="card verdict" aria-live="polite">
            {status === 'error' ? (
              <>
                <span className="small muted">Model verdict</span>
                <div className="verdict__big" style={{ color: 'var(--c-fail)', fontSize: 22 }}>
                  Could not run
                </div>
                <span className="small muted">{error}</span>
              </>
            ) : verdict && result && meta ? (
              <>
                <span className="small muted">Model verdict · computed in this browser</span>
                <div className="verdict__big">
                  <span className="dot" style={{ background: VERDICT_COLOR[verdict] }} />
                  {VERDICT_TEXT[verdict]}
                </div>
                <span className="small muted num">Anomaly score {result.score.toFixed(3)}</span>
                <div className="meter" aria-hidden="true">
                  <div className="meter__fill" style={{ width: pct(Math.min(result.score, 1), 2), background: VERDICT_COLOR[verdict] }} />
                  <div className="meter__tick" style={{ left: pct(meta.review, 2) }} />
                  <div className="meter__tick" style={{ left: pct(meta.fail, 2) }} />
                </div>
                <span className="small muted num">
                  Review ≥ {meta.review.toFixed(2)} · Fail ≥ {meta.fail.toFixed(2)} · {Math.round(result.ms)} ms
                </span>
              </>
            ) : (
              <>
                <span className="small muted">Model verdict</span>
                <div className="verdict__big" style={{ color: 'var(--c-faint)' }}>
                  <span className="dot" style={{ background: 'var(--c-line)' }} />
                  {status === 'loading' ? 'Loading' : 'Inspecting'}
                </div>
                <span className="small muted">First run downloads the model (11 MB) and this product&apos;s memory bank.</span>
              </>
            )}
            {datasetItem && (
              <>
                <hr style={{ border: 0, borderTop: '1px solid var(--c-line)', margin: '14px 0 12px' }} />
                <span className="small muted">Dataset label (ground truth)</span>
                <p style={{ margin: '6px 0 0', display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {datasetItem.label === 'good' ? (
                    <span className="chip chip--pass">Good part</span>
                  ) : (
                    <>
                      <span className="chip chip--fail">Defective</span>
                      {datasetItem.defects.map((d) => (
                        <span key={d} className="chip chip--neutral" style={{ textTransform: 'capitalize' }}>
                          {d}
                        </span>
                      ))}
                    </>
                  )}
                </p>
              </>
            )}
          </div>

          {datasetItem?.score != null && pythonVerdict && (
            <div className="card">
              <p className="h3" style={{ fontSize: 15, margin: '0 0 8px' }}>
                Matches the evaluation?
              </p>
              <p className="small muted" style={{ margin: 0 }}>
                Offline evaluation (PyTorch): score <b className="num">{datasetItem.score.toFixed(3)}</b>,{' '}
                {VERDICT_TEXT[pythonVerdict]}.
                {result && (
                  <>
                    {' '}
                    This browser: <b className="num">{result.score.toFixed(3)}</b>, from the identical 256 px model input.
                  </>
                )}
              </p>
            </div>
          )}

          {meta && (
            <div className="card">
              <p className="h3" style={{ fontSize: 15, margin: '0 0 10px' }}>
                Model
              </p>
              <div className="counters">
                <div className="counter">
                  <b>{meta.rows.toLocaleString()}</b>
                  <span>Good-part patch memories</span>
                </div>
                <div className="counter">
                  <b>256 px</b>
                  <span>Input size</span>
                </div>
              </div>
              <p className="small muted" style={{ margin: '12px 0 0' }}>
                PatchCore-style detector: ResNet18 layer2+3 features, nearest-neighbour distance to good parts. Trained only
                on good parts; thresholds fitted on the validation split. <a href="#/evaluation">See the evaluation</a>.
              </p>
            </div>
          )}
        </aside>
      </div>
    </>
  );
}
