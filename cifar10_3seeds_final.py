import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import math
import random
import csv
from pathlib import Path

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
# CIFAR-10 Dataset Loader
# ------------------------------------------------------------
def get_cifar10(batch=128, seed=42):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train = datasets.CIFAR10("./cifar10", train=True, download=True, transform=transform_train)
    test  = datasets.CIFAR10("./cifar10", train=False, download=True, transform=transform_test)

    g = torch.Generator()
    g.manual_seed(seed)

    return (
        DataLoader(train, batch_size=batch, shuffle=True, generator=g),
        DataLoader(test, batch_size=batch, shuffle=False)
    )
# ------------------------------------------------------------
# ResNet-18 for CIFAR-10
# ------------------------------------------------------------
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)

        # CIFAR-10: replace first 7×7 conv → 3×3 conv
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # CIFAR-10: remove maxpool
        self.model.maxpool = nn.Identity()

        # CIFAR-10: adjust classifier to 10 classes
        self.model.fc = nn.Linear(512, 10)

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
# REPRODUCIBILITY
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------------
def train_model(name, opt_fn, epochs=80, seed=42):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_cifar10(seed=seed)
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
        final_preds, final_labels = epoch_preds, epoch_labels

        print(
            f"{name} | Seed {seed} | Epoch {ep+1:03d}/{epochs} | "
            f"Train={train_acc[-1]:.2f}% | Test={test_acc[-1]:.2f}%"
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
# MULTI-SEED COMPARISON
# ------------------------------------------------------------
def compare_all_cifar10(epochs=80, seeds=(42, 123, 2026), output_dir="results_3seeds"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizers = {
        "SGDA":   lambda p: SGDA(p),
        "SGDM":   lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.85),
        "Adam":   lambda p: torch.optim.Adam(p, lr=0.01),
        "RAdam":  lambda p: torch.optim.RAdam(p, lr=0.001),
        "SGDMD":  lambda p: SGDMD(p, lr=0.1),
        "SGDAMD": lambda p: SGDAMD(p, lr=0.1),
    }

    all_results = {}
    all_csv_rows = []

    for name, opt_fn in optimizers.items():
        print(f"\n{'=' * 70}\nRunning {name} with seeds {list(seeds)}\n{'=' * 70}")
        runs = []

        for seed in seeds:
            result = train_model(name, opt_fn, epochs=epochs, seed=seed)
            runs.append(result)

            # Store this seed's epoch-by-epoch results in one common CSV table.
            for ep in range(epochs):
                all_csv_rows.append({
                    "optimizer": name,
                    "seed": seed,
                    "epoch": ep + 1,
                    "train_loss": result[2][ep],
                    "test_loss": result[3][ep],
                    "train_acc": result[0][ep],
                    "test_acc": result[1][ep],
                    "macro_precision": (result[4]["precision"] * 100 if ep == epochs - 1 else ""),
                    "macro_recall": (result[4]["recall"] * 100 if ep == epochs - 1 else ""),
                    "macro_f1": (result[4]["f1"] * 100 if ep == epochs - 1 else ""),
                })

        all_results[name] = runs

    # Save all optimizers, seeds, and epochs in one CSV file.
    raw_csv_path = output_dir / "all_seed_results.csv"
    with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "optimizer", "seed", "epoch",
                "train_loss", "test_loss",
                "train_acc", "test_acc",
                "macro_precision", "macro_recall", "macro_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\nAll seed results saved to: {raw_csv_path}")

    # Aggregate epoch-wise curves.
    mean_results, std_results = {}, {}
    for name, runs in all_results.items():
        arrays = {
            "train_acc": np.asarray([r[0] for r in runs]),
            "test_acc": np.asarray([r[1] for r in runs]),
            "train_loss": np.asarray([r[2] for r in runs]),
            "test_loss": np.asarray([r[3] for r in runs]),
        }
        mean_results[name] = {k: v.mean(axis=0) for k, v in arrays.items()}
        std_results[name] = {k: v.std(axis=0, ddof=1) for k, v in arrays.items()}

    # Final-epoch summary table.
    summary_rows = []
    for name, runs in all_results.items():
        values = {
            "train_loss": np.asarray([r[2][-1] for r in runs]),
            "test_loss": np.asarray([r[3][-1] for r in runs]),
            "train_acc": np.asarray([r[0][-1] for r in runs]),
            "test_acc": np.asarray([r[1][-1] for r in runs]),
            "precision": 100 * np.asarray([r[4]["precision"] for r in runs]),
            "recall": 100 * np.asarray([r[4]["recall"] for r in runs]),
            "f1": 100 * np.asarray([r[4]["f1"] for r in runs]),
        }
        row = {"optimizer": name}
        for key, vals in values.items():
            row[f"{key}_mean"] = vals.mean()
            row[f"{key}_std"] = vals.std(ddof=1)
        summary_rows.append(row)

    csv_path = output_dir / "final_summary_mean_std.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 100)
    print("FINAL RESULTS: MEAN ± SAMPLE STD OVER 3 SEEDS")
    print("=" * 100)
    for row in summary_rows:
        print(f"\n{row['optimizer']}")
        for key, label in [
            ("train_loss", "Train Loss"), ("test_loss", "Test Loss"),
            ("train_acc", "Train Acc (%)"), ("test_acc", "Test Acc (%)"),
            ("precision", "Macro Precision (%)"), ("recall", "Macro Recall (%)"),
            ("f1", "Macro F1 (%)")]:
            print(f"  {label:20s}: {row[key + '_mean']:.4f} ± {row[key + '_std']:.4f}")

    # Mean ± std plots.
    epochs_x = np.arange(1, epochs + 1)
    def plot_metric(metric, ylabel, filename):
        plt.figure(figsize=(8, 6))
        for name in optimizers:
            mean = mean_results[name][metric]
            std = std_results[name][metric]
            line, = plt.plot(epochs_x, mean, label=name, linewidth=2)
            plt.fill_between(epochs_x, mean - std, mean + std,
                             alpha=0.15, color=line.get_color())
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.show()

    plot_metric("train_acc", "Accuracy (%)", "train_accuracy_mean_std.png")
    plot_metric("test_acc", "Accuracy (%)", "test_accuracy_mean_std.png")
    plot_metric("train_loss", "Loss", "train_loss_mean_std.png")
    plot_metric("test_loss", "Loss", "test_loss_mean_std.png")

    # Aggregate final confusion matrices across seeds (sum of counts).
    confusions = {
        name: np.sum([r[4]["cm"] for r in runs], axis=0)
        for name, runs in all_results.items()
    }
    plot_confusion_grid(confusions)

    # Final macro metrics: mean ± std.
    names = list(optimizers.keys())
    x = np.arange(len(names))
    w = 0.25
    precision_mean = [100*np.mean([r[4]["precision"] for r in all_results[n]]) for n in names]
    precision_std  = [100*np.std([r[4]["precision"] for r in all_results[n]], ddof=1) for n in names]
    recall_mean    = [100*np.mean([r[4]["recall"] for r in all_results[n]]) for n in names]
    recall_std     = [100*np.std([r[4]["recall"] for r in all_results[n]], ddof=1) for n in names]
    f1_mean        = [100*np.mean([r[4]["f1"] for r in all_results[n]]) for n in names]
    f1_std         = [100*np.std([r[4]["f1"] for r in all_results[n]], ddof=1) for n in names]

    plt.figure(figsize=(12, 6))
    plt.bar(x-w, precision_mean, width=w, yerr=precision_std, capsize=4, label="Precision")
    plt.bar(x, recall_mean, width=w, yerr=recall_std, capsize=4, label="Recall")
    plt.bar(x+w, f1_mean, width=w, yerr=f1_std, capsize=4, label="F1")
    plt.xticks(x, names)
    plt.ylabel("Score (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "macro_metrics_mean_std.png", dpi=300, bbox_inches="tight")
    plt.show()

    # SGDA diagnostics: use the first seed's optimizer object.
    sgda_opt = all_results["SGDA"][0][5]
    plot_all_3d(sgda_opt)
    plot_lr_sgdnew(sgda_opt)
    plot_gradnorm_sgdnew(sgda_opt)

    print(f"\nSaved per-seed results, plots, and summary CSV to: {output_dir.resolve()}")
    return all_results, mean_results, std_results, summary_rows


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    compare_all_cifar10(epochs=80, seeds=(42, 123, 2026))
