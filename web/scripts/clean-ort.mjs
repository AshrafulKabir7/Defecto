// Production builds get ONNX Runtime's binaries from Vite (hashed assets), so the dev-only
// copies in public/ort are removed before building to avoid shipping them twice.
import { rm } from 'node:fs/promises';

await rm('public/ort', { recursive: true, force: true });
