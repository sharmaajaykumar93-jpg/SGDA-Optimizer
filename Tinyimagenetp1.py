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
from PIL import Image

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
# Tiny ImageNet Dataset Loader
# ------------------------------------------------------------
class TinyImageNetValDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform

        val_dir = os.path.join(root, "val")
        image_dir = os.path.join(val_dir, "images")
        annotation_file = os.path.join(val_dir, "val_annotations.txt")
        wnids_file = os.path.join(root, "wnids.txt")

        with open(wnids_file, "r") as f:
            classes = [line.strip() for line in f]

        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.samples = []

        with open(annotation_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                image_name = parts[0]
                class_name = parts[1]

                image_path = os.path.join(image_dir, image_name)
                label = self.class_to_idx[class_name]

                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_tinyimagenet(batch=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                             (0.2770, 0.2691, 0.2821)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                             (0.2770, 0.2691, 0.2821)),
    ])

    root = r"C:\Users\english\PycharmProjects\PythonProject1\data\tiny-imagenet-200"

    train = datasets.ImageFolder(
        os.path.join(root, "train"),
        transform=transform_train
    )

    test = datasets.ImageFolder(
        os.path.join(root, "val"),
        transform=transform_test
    )

    test = datasets.ImageFolder(
        os.path.join(root, "val"),
        transform=transform_test
    )

    return (
        DataLoader(train, batch_size=batch, shuffle=True),
        DataLoader(test, batch_size=batch, shuffle=False)
    )
# ------------------------------------------------------------
# ResNet-18 for Tiny ImageNet
# ------------------------------------------------------------
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet18(weights=None)

        # Tiny ImageNet: use 3×3 convolution
        self.model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Remove maxpool for 64×64 images
        self.model.maxpool = nn.Identity()

        # Tiny ImageNet: 200 classes
        self.model.fc = nn.Linear(512, 200)

    def forward(self, x):
        return self.model(x)

# ------------------------------------------------------------
# SGDA
# ------------------------------------------------------------
class SGDA(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, beta1=0.1, beta2=0.01, lam=0.1, eps=1e-6):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, lam=lam, eps=eps, step=0)
        super().__init__(params, defaults)

        self.grad_norms = []
        self.diff_norms = []
        self.step_sizes = []
        self.mom_norms = []

    @torch.no_grad()
    def record_epoch_stats(self):
        if len(self.grad_norms_epoch) > 0:
            self.grad_norms.append(np.mean(self.grad_norms_epoch))
            self.diff_norms.append(np.mean(self.diff_norms_epoch))
            self.step_sizes.append(np.mean(self.step_sizes_epoch))
            self.mom_norms.append(np.mean(self.mom_norms_epoch))

    @torch.no_grad()
    def step(self):

        self.grad_norms_epoch = []
        self.diff_norms_epoch = []
        self.step_sizes_epoch = []
        self.mom_norms_epoch = []

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
                lr_t = lr/  math.sqrt(t)

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


# SGDAMD
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
                beta_star = g2 * (mu * gprev2 + g2 + eps)
                beta_star = min(beta_star, beta)

                # K_D ← K1 + (K2 - K1) * σ(β*_j - β), σ(x)=1/(1+e^{-x})
                sig = 1.0 / (1.0 + math.exp(-(beta_star - beta)))
                K_D = K1 + (K2 - K1) * sig

                # m_j ← β*_j m_{j-1} + g_j + K_D s_j
                m = beta_star * m_prev + g + (K_D * s)

                # θ_{j+1} ← θ_j − α m_j
                p.data.add_(m, alpha=-0.05)

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
def train_model(name, opt_fn, epochs=30):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_tinyimagenet()
    model = ResNet18().to(device)
    optimizer = opt_fn(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    train_acc, test_acc = [], []
    train_loss, test_loss = [], []

    all_preds, all_labels = [], []

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

        # Test
        model.eval()
        correct = total = 0
        running_loss = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                running_loss += loss.item()

                preds = out.argmax(1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        test_acc.append(100 * correct / total)
        test_loss.append(running_loss / len(test_loader))

        print(f"{name} | Epoch {ep+1} | Train={train_acc[-1]:.2f}% | Test={test_acc[-1]:.2f}%")

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds,
        average="macro",
        zero_division=0
    )

    metrics = {
        "cm": confusion_matrix(all_labels, all_preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return train_acc, test_acc, train_loss, test_loss, metrics, optimizer

# ------------------------------------------------------------
# Confusion Matrix Plot
# ------------------------------------------------------------
def plot_confusion_grid(confusions):

    methods = list(confusions.keys())
    n = len(methods)
    rows = int(np.ceil(n / 2))

    fig, axes = plt.subplots(rows, 2, figsize=(12, 4*rows))

    for idx, name in enumerate(methods):
        ax = axes[idx//2][idx%2]
        cm = confusions[name]
        ax.imshow(cm, cmap="Blues")

        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, f"{cm[r,c]}", ha="center", va="center")

        #ax.set_title(name)
        ax.set_xlabel("Predicted", fontsize=20)
        ax.set_ylabel("True", fontsize=20)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# MAIN Comparison Pipeline
# ------------------------------------------------------------
def compare_all_tinyimagenet(epochs=30):

    results = {}

    print("\n=== Running SGDA ===")
    results["SGDA"] = train_model("SGDA", lambda p: SGDA(p), epochs)

    print("\n=== Running SGD ===")
    results["SGDM"] = train_model("SGDM", lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.85), epochs)

    print("\n=== Running Adam ===")
    results["Adam"] = train_model("Adam", lambda p: torch.optim.Adam(p, lr=0.01), epochs)

    print("\n=== Running RAdam ===")
    results["RAdam"] = train_model("RAdam", lambda p: torch.optim.RAdam(p, lr=0.001), epochs)

    print("\n=== Running SGDMD (CG) ===")
    results["SGDMD"] = train_model("SGDMD", lambda p: SGDMD(p, lr=0.1), epochs)

    print("\n=== Running SGDAMD (Adaptive CG-MD) ===")
    results["SGDAMD"] = train_model("SGDAMD", lambda p: SGDAMD(p, lr=0.1), epochs)



    # Plots
    names = list(results.keys())
    epochs_x = range(1, epochs+1)

    def plot_metric(idx, title, ylabel):
        plt.figure()
        for n in names:
            plt.plot(epochs_x, results[n][idx], label=n)
        #plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    plot_metric(0, "Tiny ImageNet: Train Accuracy", "Accuracy (%)")
    plot_metric(1, "Tiny ImageNet: Validation Accuracy", "Accuracy (%)")
    plot_metric(2, "Tiny ImageNet: Train Loss", "Loss")
    plot_metric(3, "Tiny ImageNet: Validation Loss", "Loss")

    # Confusion matrices
    confusions = {n: results[n][4]["cm"] for n in names}
    plot_confusion_grid(confusions)

    # F1 / Precision / Recall
    precision = [results[n][4]["precision"] for n in names]
    recall    = [results[n][4]["recall"]    for n in names]
    f1        = [results[n][4]["f1"]        for n in names]

    x = np.arange(len(names))
    w = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - w, precision, width=w, label="Precision")
    plt.bar(x,     recall,    width=w, label="Recall")
    plt.bar(x + w, f1,        width=w, label="F1")
    plt.xticks(x, names)
    #plt.title("Tiny ImageNet: Precision / Recall / F1")
    plt.legend()
    plt.show()


# ---------- 3D Diagnostics ----------
    for name in ["SGDA"]:
        print(f"\n=== 3D Diagnostics for {name} ===")
        plot_all_3d(results[name][5])
        # =====================================================
        # ADVANCED PLOTS — FOR SGD(New) ONLY
        # =====================================================
    print("\n===== Advanced Diagnostics for SGD(New) =====\n")

    sgdnew_results = results["SGDA"]
    sgdnew_opt = sgdnew_results[5]

    # 1. LR Curve
    plot_lr_sgdnew(sgdnew_opt)

    # 2. Gradient Norms
    plot_gradnorm_sgdnew(sgdnew_opt)
# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    compare_all_tinyimagenet(epochs=30)
