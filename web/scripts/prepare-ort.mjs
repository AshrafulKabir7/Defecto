// Copies the two ONNX Runtime Web binaries the inspector uses into public/ort:
// the plain WebAssembly build, and the asyncify build behind the WebGPU provider.
import { copyFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

const source = path.resolve('node_modules/onnxruntime-web/dist');
await mkdir('public/ort', { recursive: true });
for (const file of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.asyncify.wasm']) {
  await copyFile(path.join(source, file), path.join('public/ort', file));
}
console.log('ONNX Runtime Web binaries copied to public/ort.');
