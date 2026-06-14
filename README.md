# COVID-19 X-Ray Detection

A three-class chest X-ray classifier that distinguishes COVID-19, bacterial pneumonia, and healthy cases. ResNet18 is fine-tuned via transfer learning on a balanced dataset of 5,200 images, with Weights & Biases used for experiment tracking and per-prediction logging.

---

## Architecture

```
data/
  ├── COVID/        (~1,600 images, 256×256)
  ├── NORMAL/       (~1,800 images, 256×256)
  └── PNEUMONIA/    (~1,800 images, 256×256)
          │
          ▼  ChestXRayDataset (custom torch.utils.data.Dataset)
          │  Resize → 224×224, ToTensor, ImageNet normalisation
          │
          ▼  random_split     (80% train / 20% test)
          │
          ▼  ResNet18 (pretrained on ImageNet)
          │  Final FC replaced: Linear(512 → 3)
          │
          ▼  CrossEntropyLoss + Adam (lr=0.001)
          │
          ▼  10 epochs, batch_size=6
          │
          ▼  WandB — loss curves, softmax scores, prediction table
```

---

## Core Technical Stack

Python, PyTorch, torchvision (ResNet18), Weights & Biases, PIL, Matplotlib

---

## Key Methodologies

- **Transfer learning from ImageNet** — ResNet18's convolutional weights provide general visual features (edges, textures, shapes) that transfer well to grayscale medical images. Only the final fully-connected layer is replaced and trained from scratch; this converges faster and avoids overfitting on a dataset of ~5,200 images.

- **ImageNet normalisation on X-rays** — mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` applied to three-channel inputs. X-rays are naturally single-channel but are replicated to three channels to match the pretrained backbone's input expectation without architectural changes.

- **CrossEntropyLoss for multi-class** — chosen over binary cross-entropy because the three classes (COVID, Normal, Pneumonia) are mutually exclusive; CrossEntropyLoss applies a softmax internally and is numerically stable for three-way classification.

- **Per-prediction WandB table** — at inference time, softmax probabilities for all three classes are logged alongside the raw image, true label, and predicted label, enabling visual inspection of misclassifications without re-running the notebook.

---

## Production Metrics & Validation

- Dataset: 5,200 balanced chest X-ray images across three classes, sourced from the [Covid19-Pneumonia-Normal Chest X-Ray dataset](https://data.mendeley.com/datasets/dvntn9yhd2/1) (Mendeley Data).
- Training tracked in WandB project `XRay-COVID`: per-iteration loss, per-epoch accuracy, and a full prediction table on the test set.
- 80/20 stratified split; `torch.manual_seed(0)` ensures reproducibility.

---

## Local Replication

Prerequisites: Python 3.8+, CUDA-capable GPU recommended (CPU works for inference).

```bash
git clone https://github.com/tharrmeehan/COVID19-XRay-Detection.git
cd COVID19-XRay-Detection

pip install torch torchvision wandb pillow matplotlib

# Download dataset into data/ with subdirectories COVID/, NORMAL/, PNEUMONIA/
# then open and run:
jupyter notebook notebook.ipynb
```

WandB logging is optional — remove the `wandb.init(...)` and `wandb.log(...)` calls to run fully offline.
