from pathlib import Path
import sys
import torch
from train_classifier import HelmetNet
ROOT = Path(__file__).resolve().parent
ckpt_path = ROOT / 'helmet_classifier.pt'
if not ckpt_path.exists():
    raise SystemExit('Checkpoint missing. Run: python training/train_classifier.py')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model = HelmetNet().eval()
model.load_state_dict(ckpt['state_dict'])
out = ROOT.parent / 'client' / 'public' / 'helmet_classifier.onnx'
out.parent.mkdir(parents=True, exist_ok=True)
dummy = torch.zeros(1,3,128,128)
torch.onnx.export(model, dummy, out, input_names=['images'], output_names=['logits'],
                  opset_version=17, do_constant_folding=True, dynamo=False)
print('Exported:', out)
