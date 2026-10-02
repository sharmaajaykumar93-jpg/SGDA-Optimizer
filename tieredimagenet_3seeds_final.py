import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import math
import os
import random
import csv
from PIL import Image

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
# NeurIPS Style
# ------------------------------------------------------------
plt.style.use("default")
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,

    # Axis label font size
    "axes.labelsize": 18,

    # Title font size
    "axes.titlesize": 18,

    # Tick label font size
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,

    # Legend font size
    "legend.fontsize": 17,

    "font.family": "serif",
    "grid.color": "#DDDDDD",
    "axes.grid": True,
    "grid.linewidth": 0.7,
    "lines.linewidth": 2.0,
})
# ------------------------------------------------------------
# tieredImageNet / Medium ImageNet Dataset Loader
# ------------------------------------------------------------
# This experiment uses the union of the published tieredImageNet train/val/test
# class lists (608 ImageNet WordNet classes) as one ordinary classification
# label space. Training images are read directly from the ILSVRC2012 training
# split and validation images from the ILSVRC2012 validation split.
#
# No duplicate resized tieredImageNet image tree is created. Images are resized
# to 84 x 84 on the fly.
#
# Override either path with an environment variable if needed.
IMAGENET_ROOT = os.environ.get(
    "IMAGENET_ROOT",
    "/datasets/ImageNet",
)
TIERED_SPLIT_DIR = os.environ.get(
    "TIERED_SPLIT_DIR",
    "/home/user/tiered-imagenet-tools/tiered_imagenet_split",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_tiered_imagenet_class_ids(split_dir=TIERED_SPLIT_DIR):
    """
    Read the published tieredImageNet class lists and return the 608 unique
    ImageNet WordNet IDs in deterministic order.
    """
    class_ids = []
    seen = set()

    for split in ("train", "val", "test"):
        csv_path = os.path.join(split_dir, f"{split}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"tieredImageNet split file not found: {csv_path}"
            )

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                wnid = row[0].strip()
                if wnid and wnid not in seen:
                    seen.add(wnid)
                    class_ids.append(wnid)

    if len(class_ids) != 608:
        raise RuntimeError(
            f"Expected 608 unique tieredImageNet classes, found {len(class_ids)}."
        )

    return class_ids


class TieredImageNetDataset(Dataset):
    IMAGE_EXTENSIONS = (
        ".jpg", ".jpeg", ".png", ".bmp", ".ppm",
        ".pgm", ".pbm", ".pnm", ".webp",
    )

    def __init__(self, image_split_root, class_ids, transform=None):
        self.image_split_root = os.path.abspath(image_split_root)
        self.transform = transform
        self.classes = list(class_ids)
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.classes)
        }

        if not os.path.isdir(self.image_split_root):
            raise FileNotFoundError(
                f"ImageNet split directory not found: {self.image_split_root}"
            )

        self.samples = []
        missing_classes = []

        for class_name in self.classes:
            class_dir = os.path.join(self.image_split_root, class_name)

            if not os.path.isdir(class_dir):
                missing_classes.append(class_name)
                continue

            label = self.class_to_idx[class_name]

            for filename in sorted(os.listdir(class_dir)):
                path = os.path.join(class_dir, filename)

                if (
                    os.path.isfile(path)
                    and filename.lower().endswith(self.IMAGE_EXTENSIONS)
                ):
                    self.samples.append((path, label))

        if missing_classes:
            preview = ", ".join(missing_classes[:10])
            raise FileNotFoundError(
                f"{len(missing_classes)} tieredImageNet classes are missing "
                f"from {self.image_split_root}. First missing classes: {preview}"
            )

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.image_split_root} "
                "for the 608 tieredImageNet classes."
            )

        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_tieredimagenet(batch=128, num_workers=4):
    transform_train = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.RandomCrop(84, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    class_ids = load_tiered_imagenet_class_ids()

    train = TieredImageNetDataset(
        os.path.join(IMAGENET_ROOT, "train"),
        class_ids=class_ids,
        transform=transform_train,
    )

    val = TieredImageNetDataset(
        os.path.join(IMAGENET_ROOT, "validation"),
        class_ids=class_ids,
        transform=transform_val,
    )

    print(
        f"tieredImageNet classification subset: "
        f"{len(class_ids)} classes | "
        f"{len(train)} train images | "
        f"{len(val)} validation images"
    )

    return (
        DataLoader(
            train,
            batch_size=batch,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        DataLoader(
            val,
            batch_size=batch,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    )

# ------------------------------------------------------------
# ResNet-18 for tieredImageNet / Medium ImageNet
# ------------------------------------------------------------
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet18(weights=None)

        # 84 x 84 input: use 3x3 convolution
        self.model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Remove maxpool for 84 x 84 images
        self.model.maxpool = nn.Identity()

        # tieredImageNet classification subset: 608 classes
        self.model.fc = nn.Linear(512, 608)

    def forward(self, x):
        return self.model(x)

# ------------------------------------------------------------
# SGDA — Your Optimizer (UNCHANGED)
# ------------------------------------------------------------
class SGDA(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, beta1=0.1, beta2=0.01, lam=0.1, eps=1e-6):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, lam=lam, eps=eps, step=0)
        super().__init__(params, defaults)

        self.grad_norms = []
        self.diff_norms = []
        self.step_sizes = []
        self.mom_norms = []
        self.grad_norms_epoch = []
        self.diff_norms_epoch = []
        self.step_sizes_epoch = []
        self.mom_norms_epoch = []

    @torch.no_grad()
    def record_epoch_stats(self):
        if len(self.grad_norms_epoch) > 0:
            self.grad_norms.append(np.mean(self.grad_norms_epoch))
            self.diff_norms.append(np.mean(self.diff_norms_epoch))
            self.step_sizes.append(np.mean(self.step_sizes_epoch))
            self.mom_norms.append(np.mean(self.mom_norms_epoch))
        self.grad_norms_epoch = []
        self.diff_norms_epoch = []
        self.step_sizes_epoch = []
        self.mom_norms_epoch = []

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:

            group["step"] += 1
            t = group["step"]

            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lam = group["lam"]
            eps = group["eps"]

            beta2_t = beta2 * (lam ** (t - 1))

            for p in group["params"]:

                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["g_prev"] = torch.zeros_like(g)

                m = state["m"]
                g_prev = state["g_prev"]

                diff = g - g_prev
                diff_norm = diff.norm().item() + eps
                delta_g = diff / (diff_norm)

                m.mul_(beta1).add_(delta_g)

                eta_t = lr / diff_norm
                z = g + beta2_t * m
                p.add_(-eta_t * z)

                # Logging
                self.grad_norms_epoch.append(g.norm().item())
                self.diff_norms_epoch.append(diff_norm)
                self.mom_norms_epoch.append(m.norm().item())
                self.step_sizes_epoch.append(eta_t)

                state["g_prev"] = g.clone()


# ---------------------SGDMD---------------------------------------
class SGDMD(torch.optim.Optimizer):
    """
    Naming matches
      lr        -> α
      momentum  -> β1
      beta      -> β2
    """
    def __init__(self, params, lr=0.1, momentum=0.9, beta=0.1, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                st["step"] = 0
                st["m"] = torch.zeros_like(p.data)       # difference momentum
                st["g_prev"] = torch.zeros_like(p.data)  # previous gradient

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["momentum"]
            beta2 = group["beta"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                st = self.state[p]
                st["step"] += 1
                t = st["step"]

                # α_t = lr / sqrt(t), t=60 epouchs
                lr_t = lr / math.sqrt(t)

                # g_t
                g = p.grad.data
                if wd != 0:
                    g.add_(p.data, alpha=wd)

                # Δg_t = g_t - g_{t-1}
                delta_g = g - st["g_prev"]

                # m_t = β1 m_{t-1} + Δg_t
                st["m"].mul_(beta1).add_(delta_g)

                # z_t = g_t + β2 m_t
                z = g + beta2 * st["m"]

                # θ update
                p.data.add_(z, alpha=-lr_t)

                # store previous gradient
                st["g_prev"] = g.clone()


# SGDAMD (Algorithm 1 – DSP 2025) corrected to match the image
# ------------------------------------------------------------
class SGDAMD(torch.optim.Optimizer):
    def __init__(self, params,
                 lr=0.1,
                 beta=0.999,     # Momentum parameter (paper default)
                 eta=0.6,        # Adaptive differential output clamping factor
                 K1=1e-4,        # Difference coefficient
                 K2=0.01,        # Difference coefficient
                 eps=1e-5,       # Small constant
                 mu=0.5,         # Correction parameter
                 weight_decay=0):

        defaults = dict(lr=lr, beta=beta, eta=eta,
                        K1=K1, K2=K2, eps=eps, mu=mu,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        for g in self.param_groups:
            for p in g["params"]:
                st = self.state[p]
                st["g_prev"] = torch.zeros_like(p.data)  # g_{j-1}
                st["m_prev"] = torch.zeros_like(p.data)  # m_{j-1}

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr   = group["lr"]
            beta = group["beta"]
            eta  = group["eta"]
            K1   = group["K1"]
            K2   = group["K2"]
            eps  = group["eps"]
            mu   = group["mu"]
            wd   = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # g_j
                g = p.grad.data
                if wd != 0:
                    g = g.add(p.data, alpha=wd)

                st = self.state[p]
                g_prev = st["g_prev"]
                m_prev = st["m_prev"]

                # Δg_j = g_j - g_{j-1}
                Dg = g - g_prev

                # norms (scalars)
                D_norm = torch.norm(Dg).item()
                g_norm = torch.norm(g).item()
                gprev_norm = torch.norm(g_prev).item()

                # R ← η ||g_j||
                R = eta * g_norm

                # γ ← 0 if ||Δg_j|| > R else 1
                gamma = 1.0 if D_norm <= R else 0.0

                # s_j ← γ Δg_j + (1-γ) * (Δg_j / (||Δg_j|| + ε)) * R
                s = gamma * Dg + (1.0 - gamma) * (Dg * (R / (D_norm + eps)))

                # β*_j ← min( ||g_j||^2 / ( μ||g_{j-1}||^2 + ||g_j||^2 ), β )
                g2 = g_norm * g_norm
                gprev2 = gprev_norm * gprev_norm
                beta_star = g2 / (mu * gprev2 + g2 + eps)
                beta_star = min(beta_star, beta)

                # K_D ← K1 + (K2 - K1) * σ(β*_j - β), σ(x)=1/(1+e^{-x})
                sig = 1.0 / (1.0 + math.exp(-(beta_star - beta)))
                K_D = K1 + (K2 - K1) * sig

                # m_j ← β*_j m_{j-1} + g_j + K_D s_j
                m = beta_star * m_prev + g + (K_D * s)

                # θ_{j+1} ← θ_j − α m_j
                p.data.add_(m, alpha=-lr)

                # save state
                st["g_prev"] = g.clone()
                st["m_prev"] = m.clone()


# ------------------------------------------------------------
# 3D Plots for SGDA
# ------------------------------------------------------------
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def plot_all_3d(opt):
    if len(opt.grad_norms) < 1:
        print("Not enough data for 3D diagnostics.")
        return

    diff_raw = np.array(opt.diff_norms)
    mom_raw  = np.array(opt.mom_norms)
    grad_raw = np.array(opt.grad_norms)
    step_raw = np.array(opt.step_sizes)

    diff = np.round(diff_raw, 3)
    mom  = np.round(mom_raw, 3)
    grad = np.round(grad_raw, 3)
    step = np.round(step_raw, 3)

    epochs = np.arange(1, len(diff) + 1)

    def style(ax):
        ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.9))
        ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.9))
        ax.zaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.9))

        ax.grid(True, alpha=0.5)
        ax.view_init(elev=26, azim=-60)

        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.zaxis.set_major_locator(MaxNLocator(6))

        fmt = FormatStrFormatter('%.3f')
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.zaxis.set_major_formatter(fmt)

    # ---------------- 1. Surface Plot ---------------- #
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    style(ax)

    X, Y = np.meshgrid(epochs, diff)
    Z = mom.reshape(-1, 1) * np.ones_like(X)

    surf = ax.plot_surface(
        X, Y, Z,
        cmap="viridis",
        edgecolor="none",
        alpha=0.92
    )

    fig.colorbar(surf, shrink=0.55)

    # ax.set_title("Surface: Gradient Diff. vs Momentum")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Diff.")
    ax.set_zlabel("Momentum")

    plt.tight_layout()
    plt.show()

    # ---------------- 2. 3D Line Plot ---------------- #
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    style(ax)

    ax.plot(epochs, grad, mom, linewidth=3)

    # ax.set_title("3D Line: Gradient vs Momentum")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient")
    ax.set_zlabel("Momentum")

    plt.tight_layout()
    plt.show()

    # ---------------- 3. Scatter Plot ---------------- #
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    style(ax)

    scat = ax.scatter(
        step,
        diff,
        mom,
        c=step_raw,
        s=70,
        alpha=0.9
    )

    fig.colorbar(scat, shrink=0.55)

    # ax.set_title("Scatter: Learning Rate vs Gradient Diff. vs Momentum")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Gradient Diff.")
    ax.set_zlabel("Momentum")

    plt.tight_layout()
    plt.show()

    # ---------------- 4. Phase Space Plot ---------------- #
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    style(ax)

    ax.plot(grad, diff, mom, linewidth=3)

    ax.scatter(
        grad,
        diff,
        mom,
        c=np.arange(len(grad)),
        s=40
    )

    # ax.set_title("Phase Space: Gradient vs Gradient Diff. vs Momentum")
    ax.set_xlabel("Gradient")
    ax.set_ylabel("Gradient Diff.")
    ax.set_zlabel("Momentum")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# (1) Learning Rate vs Epoch for SGDA
# ------------------------------------------------------------
def plot_lr_sgdnew(opt):
    if not hasattr(opt, "step_sizes") or len(opt.step_sizes) == 0:
        print("No LR data for SGDA.")
        return

    lr = opt.step_sizes
    epochs = np.arange(1, len(lr) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, lr, linewidth=3)

    # plt.title("SGDA — Learning Rate vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.show()


# ------------------------------------------------------------
# (2) Gradient Norm Trajectory
# ------------------------------------------------------------
def plot_gradnorm_sgdnew(opt):
    if not hasattr(opt, "grad_norms") or len(opt.grad_norms) == 0:
        print("No grad norm data for SGDA.")
        return

    gnorm = opt.grad_norms
    epochs = np.arange(1, len(gnorm) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, gnorm, linewidth=3)

    # plt.title("SGDA — Gradient Norm Trajectory")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.grid(True)
    plt.show()

# ------------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------------
def train_model(name, opt_fn, epochs=70, seed=42):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_tieredimagenet()
    model = ResNet18().to(device)
    optimizer = opt_fn(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    train_acc, test_acc = [], []
    train_loss, test_loss = [], []

    final_preds, final_labels = [], []

    for ep in range(epochs):
        model.train()
        correct = total = 0
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)

        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / len(train_loader))

        if isinstance(optimizer, SGDA):
            optimizer.record_epoch_stats()

        # Validation
        model.eval()
        correct = total = 0
        running_loss = 0.0
        epoch_preds, epoch_labels = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                running_loss += loss.item()

                preds = out.argmax(1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)

                epoch_preds.extend(preds.cpu().numpy())
                epoch_labels.extend(y.cpu().numpy())

        test_acc.append(100 * correct / total)
        test_loss.append(running_loss / len(test_loader))

        # Keep only the final epoch predictions for final metrics.
        final_preds = epoch_preds
        final_labels = epoch_labels

        print(
            f"{name} | Seed {seed} | Epoch {ep+1}/{epochs} | "
            f"Train={train_acc[-1]:.2f}% | Val={test_acc[-1]:.2f}%"
        )

    precision, recall, f1, _ = precision_recall_fscore_support(
        final_labels, final_preds, average="macro", zero_division=0
    )

    metrics = {
        "cm": confusion_matrix(final_labels, final_preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return train_acc, test_acc, train_loss, test_loss, metrics, optimizer


# ------------------------------------------------------------
# THREE-SEED COMPARISON PIPELINE
# ------------------------------------------------------------
def compare_all_tieredimagenet(epochs=70, seeds=(42, 123, 2026)):
    output_dir = "results_tieredimagenet_3seeds"
    os.makedirs(output_dir, exist_ok=True)

    # Same optimizer hyperparameter settings used for Tiny ImageNet:
    # SGDA:   lr=0.1, beta1=0.1, beta2=0.01, lam=0.1, eps=1e-6
    # SGDM:   lr=0.01, momentum=0.85, weight_decay=0
    # Adam:   lr=0.01, PyTorch default betas=(0.9, 0.999), eps=1e-8
    # RAdam:  lr=0.001, PyTorch default betas=(0.9, 0.999), eps=1e-8
    # SGDMD:  lr=0.1, momentum=0.9, beta=0.1, weight_decay=0
    # SGDAMD: lr=0.1, beta=0.999, eta=0.6, K1=1e-4, K2=0.01,
    #         eps=1e-5, mu=0.5, weight_decay=0
    optimizers = {
        "SGDA": lambda p: SGDA(
            p, lr=0.1, beta1=0.1, beta2=0.01, lam=0.1, eps=1e-6
        ),
        "SGDM": lambda p: torch.optim.SGD(
            p, lr=0.01, momentum=0.85, weight_decay=0.0
        ),
        "Adam": lambda p: torch.optim.Adam(
            p, lr=0.01, weight_decay=0.0
        ),
        "RAdam": lambda p: torch.optim.RAdam(
            p, lr=0.001, weight_decay=0.0
        ),
        "SGDMD": lambda p: SGDMD(
            p, lr=0.1, momentum=0.9, beta=0.1, weight_decay=0.0
        ),
        "SGDAMD": lambda p: SGDAMD(
            p,
            lr=0.1,
            beta=0.999,
            eta=0.6,
            K1=1e-4,
            K2=0.01,
            eps=1e-5,
            mu=0.5,
            weight_decay=0.0,
        ),
    }

    all_results = {}
    all_csv_rows = []

    for name, opt_fn in optimizers.items():
        print("\n" + "=" * 70)
        print(f"Running {name} with seeds {seeds}")
        print("=" * 70)

        seed_results = []

        for seed in seeds:
            print(f"\n--- {name}: seed {seed} ---")
            result = train_model(name, opt_fn, epochs=epochs, seed=seed)
            seed_results.append(result)

            # Store this seed's epoch-by-epoch results in one common CSV table.
            for ep in range(epochs):
                all_csv_rows.append({
                    "optimizer": name,
                    "seed": seed,
                    "epoch": ep + 1,
                    "train_loss": result[2][ep],
                    "val_loss": result[3][ep],
                    "train_acc": result[0][ep],
                    "val_acc": result[1][ep],
                    "macro_precision": (result[4]["precision"] * 100 if ep == epochs - 1 else ""),
                    "macro_recall": (result[4]["recall"] * 100 if ep == epochs - 1 else ""),
                    "macro_f1": (result[4]["f1"] * 100 if ep == epochs - 1 else ""),
                })

        all_results[name] = seed_results

    # Save all optimizers, seeds, and epochs in one CSV file.
    raw_csv_path = os.path.join(output_dir, "all_seed_results.csv")
    with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "optimizer", "seed", "epoch",
                "train_loss", "val_loss",
                "train_acc", "val_acc",
                "macro_precision", "macro_recall", "macro_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\nAll seed results saved to: {raw_csv_path}")

    # Aggregate epoch-wise curves.
    mean_results, std_results = {}, {}

    for name, runs in all_results.items():
        train_acc_runs = np.asarray([r[0] for r in runs])
        val_acc_runs = np.asarray([r[1] for r in runs])
        train_loss_runs = np.asarray([r[2] for r in runs])
        val_loss_runs = np.asarray([r[3] for r in runs])

        mean_results[name] = {
            "train_acc": train_acc_runs.mean(axis=0),
            "val_acc": val_acc_runs.mean(axis=0),
            "train_loss": train_loss_runs.mean(axis=0),
            "val_loss": val_loss_runs.mean(axis=0),
        }
        std_results[name] = {
            "train_acc": train_acc_runs.std(axis=0, ddof=1),
            "val_acc": val_acc_runs.std(axis=0, ddof=1),
            "train_loss": train_loss_runs.std(axis=0, ddof=1),
            "val_loss": val_loss_runs.std(axis=0, ddof=1),
        }

    def plot_metric(metric, ylabel, filename):
        plt.figure(figsize=(8, 6))
        epochs_x = np.arange(1, epochs + 1)

        for name in optimizers:
            mean = mean_results[name][metric]
            std = std_results[name][metric]
            line, = plt.plot(epochs_x, mean, label=name, linewidth=2)
            plt.fill_between(
                epochs_x, mean - std, mean + std,
                alpha=0.15, color=line.get_color()
            )

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    plot_metric("train_acc", "Accuracy (%)", "train_accuracy_mean_std.png")
    plot_metric("val_acc", "Accuracy (%)", "validation_accuracy_mean_std.png")
    plot_metric("train_loss", "Loss", "train_loss_mean_std.png")
    plot_metric("val_loss", "Loss", "validation_loss_mean_std.png")

    # Final-epoch mean +/- sample SD table.
    summary_rows = []

    print("\n" + "=" * 100)
    print("TIEREDIMAGENET FINAL RESULTS: MEAN ± STD OVER 3 SEEDS")
    print("=" * 100)

    for name, runs in all_results.items():
        train_loss = np.asarray([r[2][-1] for r in runs])
        val_loss = np.asarray([r[3][-1] for r in runs])
        train_acc = np.asarray([r[0][-1] for r in runs])
        val_acc = np.asarray([r[1][-1] for r in runs])
        precision = np.asarray([r[4]["precision"] for r in runs]) * 100.0
        recall = np.asarray([r[4]["recall"] for r in runs]) * 100.0
        f1 = np.asarray([r[4]["f1"] for r in runs]) * 100.0

        row = {
            "optimizer": name,
            "train_loss_mean": train_loss.mean(),
            "train_loss_std": train_loss.std(ddof=1),
            "val_loss_mean": val_loss.mean(),
            "val_loss_std": val_loss.std(ddof=1),
            "train_acc_mean": train_acc.mean(),
            "train_acc_std": train_acc.std(ddof=1),
            "val_acc_mean": val_acc.mean(),
            "val_acc_std": val_acc.std(ddof=1),
            "macro_precision_mean": precision.mean(),
            "macro_precision_std": precision.std(ddof=1),
            "macro_recall_mean": recall.mean(),
            "macro_recall_std": recall.std(ddof=1),
            "macro_f1_mean": f1.mean(),
            "macro_f1_std": f1.std(ddof=1),
        }
        summary_rows.append(row)

        print(f"\n{name}")
        print(f"Train Loss : {row['train_loss_mean']:.6f} ± {row['train_loss_std']:.6f}")
        print(f"Val Loss   : {row['val_loss_mean']:.6f} ± {row['val_loss_std']:.6f}")
        print(f"Train Acc  : {row['train_acc_mean']:.3f} ± {row['train_acc_std']:.3f}")
        print(f"Val Acc    : {row['val_acc_mean']:.3f} ± {row['val_acc_std']:.3f}")
        print(f"Precision  : {row['macro_precision_mean']:.3f} ± {row['macro_precision_std']:.3f}")
        print(f"Recall     : {row['macro_recall_mean']:.3f} ± {row['macro_recall_std']:.3f}")
        print(f"F1         : {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}")

    csv_path = os.path.join(output_dir, "final_summary_mean_std.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Macro Precision / Recall / F1: mean +/- SD.
    names = list(optimizers.keys())
    precision_mean = np.asarray([
        np.mean([r[4]["precision"] for r in all_results[n]]) * 100 for n in names
    ])
    precision_std = np.asarray([
        np.std([r[4]["precision"] for r in all_results[n]], ddof=1) * 100 for n in names
    ])
    recall_mean = np.asarray([
        np.mean([r[4]["recall"] for r in all_results[n]]) * 100 for n in names
    ])
    recall_std = np.asarray([
        np.std([r[4]["recall"] for r in all_results[n]], ddof=1) * 100 for n in names
    ])
    f1_mean = np.asarray([
        np.mean([r[4]["f1"] for r in all_results[n]]) * 100 for n in names
    ])
    f1_std = np.asarray([
        np.std([r[4]["f1"] for r in all_results[n]], ddof=1) * 100 for n in names
    ])

    x = np.arange(len(names))
    w = 0.25
    plt.figure(figsize=(12, 6))
    plt.bar(x - w, precision_mean, width=w, yerr=precision_std, capsize=4, label="Precision")
    plt.bar(x, recall_mean, width=w, yerr=recall_std, capsize=4, label="Recall")
    plt.bar(x + w, f1_mean, width=w, yerr=f1_std, capsize=4, label="F1")
    plt.xticks(x, names)
    plt.ylabel("Score (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "macro_metrics_mean_std.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.show()
    plt.close()

    # SGDA diagnostics: use seed 42 as a representative run.
    sgda_opt = all_results["SGDA"][0][5]
    print(f"\n=== SGDA diagnostics (representative seed {seeds[0]}) ===")
    plot_all_3d(sgda_opt)
    plot_lr_sgdnew(sgda_opt)
    plot_gradnorm_sgdnew(sgda_opt)

    print(f"\nSaved all outputs to: {os.path.abspath(output_dir)}")
    return all_results, mean_results, std_results


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    compare_all_tieredimagenet(epochs=70, seeds=(42, 123, 2026))
