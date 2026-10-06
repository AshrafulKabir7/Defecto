import { useEffect, useState } from 'react';

/** Shapes written by ml/export_web.py. */
export interface Item {
  id: string;
  label: 'good' | 'bad';
  defects: string[];
  mask?: boolean;
  heat?: boolean;
  /** Exact 256x256 model input exported for in-browser inference. */
  input?: boolean;
  /** Normalised model score (0..1); present only once the model has been evaluated. */
  score?: number;
}

export interface ModelInfo {
  name: string;
  thresholds: { fail: number; review: number; review_recall_target: number };
  latency_ms: { device: string; mean: number; p95: number; note: string };
  train: {
    train_images?: number;
    train_seconds?: number;
    gpu?: string | null;
    backbone?: string;
    max_steps?: number;
    memory_bank?: [number, number];
    implementation?: string;
  };
  test: {
    image_auroc: number;
    image_ap: number;
    pixel_auroc?: number;
    'pixel_aupro_0.3'?: number;
    n_good: number;
    n_bad: number;
  };
}

export interface Manifest {
  category: string;
  title: string;
  blurb: string;
  dataset: string;
  counts: { train_good: number; val: number; test_good: number; test_bad: number };
  model: ModelInfo | null;
  items: Item[];
}

export interface CategoryIndex {
  category: string;
  title: string;
  hasModel: boolean;
  model: string | null;
  n: number;
}

const EXTENSION = { img: 'jpg', mask: 'png', heat: 'png', input: 'webp' } as const;

export const asset = (category: string, kind: keyof typeof EXTENSION, id: string) =>
  `${import.meta.env.BASE_URL}demo/${category}/${kind}/${id}.${EXTENSION[kind]}`;

const cache = new Map<string, Promise<unknown>>();
function getJson<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(`${import.meta.env.BASE_URL}${path}`).then((r) => {
        if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
        return r.json();
      }),
    );
  }
  return cache.get(path) as Promise<T>;
}

export function useIndex() {
  const [data, setData] = useState<CategoryIndex[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    getJson<{ categories: CategoryIndex[] }>('demo/index.json')
      .then((d) => setData(d.categories))
      .catch((e) => setError(String(e)));
  }, []);
  return { data, error };
}

export function useManifest(category: string | null) {
  const [data, setData] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!category) return;
    let live = true;
    setData(null);
    getJson<Manifest>(`demo/${category}/manifest.json`)
      .then((d) => live && setData(d))
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [category]);
  return { data, error };
}

/* ── Decisions and metrics (computed from per-image scores) ── */

export type Verdict = 'pass' | 'review' | 'fail';

export function verdictOf(score: number, failT: number, reviewT: number): Verdict {
  if (score >= failT) return 'fail';
  if (score >= reviewT) return 'review';
  return 'pass';
}

export interface Confusion {
  tp: number;
  fn: number;
  fp: number;
  tn: number;
  recall: number;
  precision: number;
  escapeRate: number;
  falseRejectRate: number;
}

export function confusionAt(items: Item[], threshold: number): Confusion {
  let tp = 0, fn = 0, fp = 0, tn = 0;
  for (const it of items) {
    if (it.score == null) continue;
    const flagged = it.score >= threshold;
    if (it.label === 'bad') flagged ? tp++ : fn++;
    else flagged ? fp++ : tn++;
  }
  return {
    tp, fn, fp, tn,
    recall: tp + fn ? tp / (tp + fn) : 0,
    precision: tp + fp ? tp / (tp + fp) : 0,
    escapeRate: tp + fn ? fn / (tp + fn) : 0,
    falseRejectRate: fp + tn ? fp / (fp + tn) : 0,
  };
}

/** ROC points [fpr, tpr] sweeping every distinct score. */
export function rocPoints(items: Item[]): [number, number][] {
  const scored = items.filter((i) => i.score != null);
  const thresholds = [Infinity, ...Array.from(new Set(scored.map((i) => i.score!))).sort((a, b) => b - a), -Infinity];
  return thresholds.map((t) => {
    const c = confusionAt(scored, t);
    return [c.falseRejectRate, c.recall];
  });
}

export function defectCounts(items: Item[]): [string, number][] {
  const counts = new Map<string, number>();
  for (const it of items) for (const d of it.defects) counts.set(d, (counts.get(d) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
