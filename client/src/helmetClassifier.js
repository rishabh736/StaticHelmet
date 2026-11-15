let session = null;
const SIZE = 128;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const canvas = document.createElement('canvas');
canvas.width = canvas.height = SIZE;
const ctx = canvas.getContext('2d', {willReadFrequently:true});

export async function loadHelmetClassifier(url='/helmet_classifier.onnx') {
  if (session) return true;
  const ort = window.ort;
  if (!ort) throw new Error('ONNX Runtime Web is not available');
  const response = await fetch(url);
  if (!response.ok) throw new Error('Camera classifier ONNX file is missing');
  const bytes = new Uint8Array(await response.arrayBuffer());
  session = await ort.InferenceSession.create(bytes, {executionProviders:['wasm'], graphOptimizationLevel:'all'});
  return true;
}

// Face box is expanded to include the helmet/head above the face and shoulders below it.
// This classifier is used only for live webcam frames, not uploaded photos/videos.
export async function classifyCameraFace(src, face, W, H) {
  if (!session) throw new Error('Camera classifier is not loaded');
  const cx = face.x + face.w/2;
  const rw = face.w * 3.0;
  const rh = face.h * 4.35;
  const x = Math.max(0, cx-rw/2);
  const y = Math.max(0, face.y-face.h*1.55);
  const right = Math.min(W, cx+rw/2);
  const bottom = Math.min(H, face.y+face.h*2.8);
  ctx.fillStyle = '#777'; ctx.fillRect(0,0,SIZE,SIZE);
  ctx.drawImage(src, x, y, Math.max(1,right-x), Math.max(1,bottom-y), 0,0,SIZE,SIZE);
  const pixels = ctx.getImageData(0,0,SIZE,SIZE).data;
  const plane = SIZE*SIZE, input = new Float32Array(3*plane);
  for (let i=0;i<plane;i++) {
    input[i] = (pixels[i*4]/255-MEAN[0])/STD[0];
    input[plane+i] = (pixels[i*4+1]/255-MEAN[1])/STD[1];
    input[2*plane+i] = (pixels[i*4+2]/255-MEAN[2])/STD[2];
  }
  const out = await session.run({[session.inputNames[0]]:new window.ort.Tensor('float32',input,[1,3,SIZE,SIZE])});
  const logits = out[session.outputNames[0]].data;
  const m = Math.max(logits[0],logits[1]);
  const a = Math.exp(logits[0]-m), b = Math.exp(logits[1]-m), sum=a+b;
  const helmet=a/sum, noHelmet=b/sum;
  const cls = helmet >= noHelmet ? 0 : 1;
  return {x, y, w:right-x, h:bottom-y, c:cls, s:Math.max(helmet,noHelmet), s0:helmet, s1:noHelmet};
}
