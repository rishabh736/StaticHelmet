# Rider Kavach – webcam-only helmet classifier

This is a SECONDARY binary image classifier trained from the supplied 74 photographs. It is used only in the live webcam path. The existing `best.onnx` YOLO model and uploaded photo/video analysis path are left intact.

- Class `0`: helmet
- Class `1`: no_helmet
- Input: 128 x 128 RGB, ImageNet normalization
- Checkpoint: `training/helmet_classifier.pt`
- Browser model target: `client/public/helmet_classifier.onnx`

## Windows setup

From the project root, activate your Python environment and install:

```powershell
python -m pip install torch torchvision pillow onnx
```

To retrain using the 74 labeled images:

```powershell
python training/train_classifier.py
```

To export the checkpoint for browser inference:

```powershell
python training/export_onnx.py
```

Then run the frontend normally. Live webcam detection will try the new classifier. If its ONNX file is absent or face detection fails, it falls back to the existing YOLO + face-fusion path. Uploaded photos and videos continue using the original YOLO model.

## Dataset limitation

The 74 photos are a small, same-person/same-room set with very similar backgrounds and camera framing. This is useful for a controlled close-up demo but is not enough to claim general road-scene accuracy. Validate with different people, helmets, angles, distances, lighting and outdoor backgrounds before relying on the result. A face detector must also find the face; full-face helmets, masks, occlusion or profile views can reduce reliability.
