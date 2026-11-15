from pathlib import Path
import json, random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / 'helmet_dataset' / 'images'
LABELS = json.loads((ROOT / 'labels.json').read_text(encoding='utf-8'))
CLASSES = ['helmet', 'no_helmet']  # label 0 = helmet, label 1 = no helmet
torch.set_num_threads(min(4, torch.get_num_threads()))

class HelmetDataset(Dataset):
    def __init__(self, augment=False):
        self.items = [(IMAGE_DIR / name, int(label)) for name, label in LABELS.items()]
        ops = ([transforms.RandomResizedCrop(128, scale=(.75, 1.0), ratio=(.85, 1.15)),
                transforms.RandomHorizontalFlip(), transforms.ColorJitter(.18, .18, .15, .04)]
               if augment else [transforms.Resize((128, 128))])
        self.transform = transforms.Compose(ops + [transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label = self.items[i]
        im = Image.open(path).convert('RGB')
        w, h = im.size
        # Training photos are portrait close-ups; focus on head and upper torso.
        im = im.crop((int(w*.12), 0, int(w*.88), int(h*.62)))
        return self.transform(im), label

class HelmetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,96,3,padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.25), nn.Linear(96,2))
    def forward(self, x): return self.head(self.features(x))

if __name__ == '__main__':
    random.seed(42); torch.manual_seed(42)
    ds = HelmetDataset(augment=True)
    n_val = max(10, round(len(ds)*.2))
    train_ds, val_indices = random_split(ds, [len(ds)-n_val,n_val], generator=torch.Generator().manual_seed(42))
    # Recreate deterministic validation transforms using the same selected indices.
    val_base = HelmetDataset(augment=False)
    val_ds = torch.utils.data.Subset(val_base, val_indices.indices)
    train_loader = DataLoader(train_ds, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=12)
    model = HelmetNet(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
    criterion = nn.CrossEntropyLoss(); best = -1
    for epoch in range(25):
        model.train(); total = 0
        for x,y in train_loader:
            optimizer.zero_grad(); loss=criterion(model(x),y); loss.backward(); optimizer.step(); total += loss.item()*len(y)
        model.eval(); correct=seen=0
        with torch.no_grad():
            for x,y in val_loader:
                pred=model(x).argmax(1); correct += (pred==y).sum().item(); seen += len(y)
        acc=correct/max(seen,1)
        print(f'Epoch {epoch+1:02d}/25 loss={total/len(train_ds):.4f} val_accuracy={acc:.3f}')
        if acc > best:
            best=acc
            torch.save({'state_dict':model.cpu().eval().state_dict(),'classes':CLASSES,'input_size':128,
                        'mean':[.485,.456,.406],'std':[.229,.224,.225]}, ROOT/'helmet_classifier.pt')
        model.train()
    print(f'Best holdout accuracy: {best:.3f} (small, same-session dataset; not a real-world accuracy estimate)')
