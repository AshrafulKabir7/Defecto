/**
 * In-browser anomaly detection with the PatchCore-lite graph from ml/patchcore_lite.py.
 *
 * One ONNX graph (ResNet18 features + nearest-neighbour scoring + smoothing) is shared by
 * every product; each product brings its own memory bank of good-part patch embeddings and
 * the normalisation fitted on its validation split. The browser therefore runs the same
 * computation that produced the published evaluation numbers.
 */
import type { InferenceSession, Tensor } from 'onnxruntime-web';

type Ort = typeof import('onnxruntime-web/wasm');
export type Backend = 'webgpu' | 'wasm';

export interface BankMeta {
  category: string;
  rows: number;
  dim: number;
  raw_threshold: number;
  raw_min: number;
  raw_max: number;
  fail: number;
  review: number;
}

export interface InspectionResult {
  /** Normalised anomaly score; FAIL threshold is 0.5. */
  score: number;
  /** Normalised 256x256 anomaly map, row-major, values 0..1. */
  map: Float32Array;
  ms: number;
}

const SIZE = 256;
const base = import.meta.env.BASE_URL;

let sessionPromise: Promise<{ ort: Ort; session: InferenceSession; backend: Backend }> | null = null;
const banks = new Map<string, Promise<{ meta: BankMeta; tensor: Tensor }>>();

async function createSession() {
  const model = `${base}models/patchcore-resnet18.onnx`;
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (gpu) {
    try {
      if (await gpu.requestAdapter()) {
        const ort = (await import('onnxruntime-web/webgpu')) as unknown as Ort;
        // Production builds bundle the binary next to the runtime; the dev server serves a copy.
        if (import.meta.env.DEV) ort.env.wasm.wasmPaths = { wasm: new URL(`${base}ort/ort-wasm-simd-threaded.asyncify.wasm`, location.href).href };
        ort.env.wasm.numThreads = 1;
        ort.env.logLevel = 'error';
        const session = await ort.InferenceSession.create(model, { executionProviders: ['webgpu'] });
        return { ort, session, backend: 'webgpu' as const };
      }
    } catch (error) {
      console.warn('WebGPU unavailable, using WebAssembly.', error);
    }
  }
  const ort = await import('onnxruntime-web/wasm');
  if (import.meta.env.DEV) ort.env.wasm.wasmPaths = { wasm: new URL(`${base}ort/ort-wasm-simd-threaded.wasm`, location.href).href };
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'error';
  const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
  return { ort, session, backend: 'wasm' as const };
}

export function loadModel() {
  sessionPromise ??= createSession().catch((error) => {
    sessionPromise = null;
    throw error;
  });
  return sessionPromise;
}

export function loadBank(category: string) {
  if (!banks.has(category)) {
    banks.set(
      category,
      (async () => {
        const [{ ort }, meta, buffer] = await Promise.all([
          loadModel(),
          fetch(`${base}models/${category}.bank.json`).then((r) => {
            if (!r.ok) throw new Error(`No trained model for ${category} yet.`);
            return r.json() as Promise<BankMeta>;
          }),
          fetch(`${base}models/${category}.bank.bin`).then((r) => {
            if (!r.ok) throw new Error(`No memory bank for ${category} yet.`);
            return r.arrayBuffer();
          }),
        ]);
        const data = new Float32Array(buffer);
        if (data.length !== meta.rows * meta.dim) throw new Error('Memory bank size does not match its metadata.');
        return { meta, tensor: new ort.Tensor('float32', data, [meta.rows, meta.dim]) };
      })().catch((error) => {
        banks.delete(category);
        throw error;
      }),
    );
  }
  return banks.get(category)!;
}

/**
 * 256x256 area-average (box) resize, matching PIL's Image.BOX used in training, then CHW
 * floats in 0..1. Each output pixel averages the source pixels it covers, with fractional
 * weights at the edges. A 256x256 input (the exported model inputs) passes through as-is.
 */
function toTensorData(image: HTMLImageElement) {
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  // Very large photos are first halved by the browser to keep memory bounded.
  const scale = Math.min(1, 3000 / Math.max(width, height));
  const w = Math.max(SIZE, Math.round(width * scale));
  const h = Math.max(SIZE, Math.round(height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(image, 0, 0, w, h);
  const src = ctx.getImageData(0, 0, w, h).data;

  const weights = (inSize: number) => {
    const ratio = inSize / SIZE;
    return Array.from({ length: SIZE }, (_, o) => {
      const start = o * ratio;
      const end = start + ratio;
      const taps: [number, number][] = [];
      for (let i = Math.floor(start); i < Math.ceil(end) && i < inSize; i++) {
        const overlap = Math.min(end, i + 1) - Math.max(start, i);
        if (overlap > 1e-9) taps.push([i, overlap / ratio]);
      }
      return taps;
    });
  };
  const xs = weights(w);
  const ys = weights(h);

  const plane = SIZE * SIZE;
  const data = new Float32Array(3 * plane);
  const row = new Float32Array(SIZE * 3);
  for (let oy = 0; oy < SIZE; oy++) {
    row.fill(0);
    for (const [iy, wy] of ys[oy]) {
      const base = iy * w * 4;
      for (let ox = 0; ox < SIZE; ox++) {
        let r = 0, g = 0, b = 0;
        for (const [ix, wx] of xs[ox]) {
          const p = base + ix * 4;
          r += src[p] * wx;
          g += src[p + 1] * wx;
          b += src[p + 2] * wx;
        }
        row[ox * 3] += r * wy;
        row[ox * 3 + 1] += g * wy;
        row[ox * 3 + 2] += b * wy;
      }
    }
    for (let ox = 0; ox < SIZE; ox++) {
      const i = oy * SIZE + ox;
      // PIL rounds the resized image to 8 bits before the model sees it.
      data[i] = Math.round(row[ox * 3]) / 255;
      data[i + plane] = Math.round(row[ox * 3 + 1]) / 255;
      data[i + 2 * plane] = Math.round(row[ox * 3 + 2]) / 255;
    }
  }
  return data;
}

export async function inspect(category: string, image: HTMLImageElement): Promise<InspectionResult> {
  const [{ ort, session }, { meta, tensor: bank }] = await Promise.all([loadModel(), loadBank(category)]);
  const input = new ort.Tensor('float32', toTensorData(image), [1, 3, SIZE, SIZE]);
  const started = performance.now();
  const outputs = await session.run({ image: input, bank });
  const ms = performance.now() - started;
  try {
    const span = Math.max(meta.raw_max - meta.raw_min, 1e-6);
    const normalise = (raw: number) => Math.min(1, Math.max(0, (raw - meta.raw_threshold) / span + 0.5));
    const rawMap = outputs.anomaly_map.data as Float32Array;
    const map = new Float32Array(rawMap.length);
    for (let i = 0; i < rawMap.length; i++) map[i] = normalise(rawMap[i]);
    return { score: normalise((outputs.score.data as Float32Array)[0]), map, ms };
  } finally {
    input.dispose();
    Object.values(outputs).forEach((t) => t.dispose());
  }
}

/** Same transparent-to-hot colour stops as ml/export_web.py, so live and replayed heatmaps match. */
const STOPS: [number, number, number, number, number][] = [
  [0, 0, 0, 0, 0],
  [0.35, 56, 198, 236, 0],
  [0.55, 56, 198, 236, 110],
  [0.75, 246, 168, 37, 170],
  [1, 229, 72, 77, 215],
];

/** Colour-maps a 256x256 map and stretches it back to the photo's aspect ratio. */
export function heatmapDataUrl(map: Float32Array, width: number, height: number): string {
  const canvas = document.createElement('canvas');
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext('2d')!;
  const image = ctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < map.length; i++) {
    const v = map[i];
    let s = 1;
    while (s < STOPS.length - 1 && v > STOPS[s][0]) s++;
    const [a0, ...c0] = STOPS[s - 1];
    const [a1, ...c1] = STOPS[s];
    const t = Math.min(1, Math.max(0, (v - a0) / (a1 - a0 || 1)));
    for (let c = 0; c < 4; c++) image.data[i * 4 + c] = c0[c] + (c1[c] - c0[c]) * t;
  }
  ctx.putImageData(image, 0, 0);
  const out = document.createElement('canvas');
  out.width = 640;
  out.height = Math.max(1, Math.round((640 * height) / width));
  const outCtx = out.getContext('2d')!;
  outCtx.imageSmoothingQuality = 'high';
  outCtx.drawImage(canvas, 0, 0, out.width, out.height);
  return out.toDataURL('image/png');
}
