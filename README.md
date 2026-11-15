# Rider Kavach: Smart Helmet Compliance Monitoring

    npm run setup     (first time)
    npm run dev       then open http://localhost:5173

Login (created on first run): admin / admin123 and officer / officer123.
To choose your own passwords, delete server/data/users.json and start the server with
ADMIN_PASSWORD and OFFICER_PASSWORD set. Only admins can delete data.
Production: npm run build, then npm start (http://localhost:5050).

## Hosting on Render (one service for frontend + backend)
Build command: npm i && npm i --prefix server && npm i --include=dev --prefix client && npm run build
Start command: npm start
Environment: DATA_DIR=/var/data, ADMIN_PASSWORD, OFFICER_PASSWORD (set before first start)
Disk: mount at /var/data (needs a paid instance). render.yaml in this folder does all of this.

## Frontend and backend on different hosts
Set VITE_API_URL=https://your-backend-url (no trailing slash) on the FRONTEND host and redeploy it.
vercel.json and public/_redirects make page refreshes work on Vercel / Netlify.

## Webcam-only helmet adaptation (added)

The supplied 74-photo dataset is included under `training/helmet_dataset/images`. A separate small CNN checkpoint is included at `training/helmet_classifier.pt`. The live webcam path now tries this classifier on MediaPipe face/head crops; uploaded images and videos continue to use the original `client/public/best.onnx` YOLO path.

Before the webcam classifier can run in the browser, export its checkpoint once on your computer:

```powershell
python -m pip install torch torchvision pillow onnx
python training/export_onnx.py
```

This creates `client/public/helmet_classifier.onnx`. If that file is missing, the webcam path automatically falls back to the original YOLO + face-fusion logic.

The classifier is an experimental adaptation to the supplied close-up photos, not a general road-traffic model. Its small same-person dataset needs testing with independent riders and outdoor camera footage.
