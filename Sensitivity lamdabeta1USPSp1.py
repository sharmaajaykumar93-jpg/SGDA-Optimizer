import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import ssl; ssl._create_default_https_context = ssl._create_unverified_context

plt.style.use("default")
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "font.family": "serif",
    "grid.color": "#DDDDDD",
    "axes.grid": True,
    "grid.linewidth": 0.7,
    "lines.linewidth": 2.0,
})

# ------------------------------------------------------------
# USPS Dataset
# ------------------------------------------------------------
def get_usps(batch=128):
    transform = transforms.ToTensor()
    train_set = datasets.USPS("./usps", train=True, download=True, transform=transform)
    test_set  = datasets.USPS("./usps", train=False, download=True, transform=transform)
    return (
        DataLoader(train_set, batch_size=batch, shuffle=True),
        DataLoader(test_set, batch_size=batch, shuffle=False),
    )

# ------------------------------------------------------------
# LeNet-5
# ------------------------------------------------------------
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.Tanh(),
            nn.AvgPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 1 * 1, 120), nn.Tanh(),
            nn.Linear(120, 84), nn.Tanh(),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ------------------------------------------------------------
# SGD(A)
# ------------------------------------------------------------
class SGDMD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, beta1=0.001, beta2=0.1, lam=0.001, eps=1e-3):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, lam=lam, eps=eps, step=0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            group["step"] += 1
            t = group["step"]

            lr = group["lr"]
            beta1, beta2, lam, eps = group["beta1"], group["beta2"], group["lam"], group["eps"]
            beta2_t = beta2 * (lam ** (t - 1))

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["m"] = torch.zeros_like(g)
                    state["g_prev"] = torch.zeros_like(g)

                m = state["m"]
                g_prev = state["g_prev"]

                diff = g - g_prev
                diff_norm = diff.norm().item() + eps
                delta_g = diff / diff_norm

                m.mul_(beta1).add_(delta_g)

                eta_t = lr / diff_norm
                z = g + beta2_t * m
                p.add_(-eta_t * z)

                state["g_prev"] = g.clone()

# ------------------------------------------------------------
# Training Loop (now parameterized by beta2 and lam)
# ------------------------------------------------------------
def train_once(lr=0.1, beta1=0.001, beta2=0.1, lam=0.001, eps=1e-3, epochs = 50):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_usps()

    model = LeNet5().to(device)
    optimizer = SGDMD(model.parameters(), lr=lr, beta1=beta1, beta2=beta2, lam=lam, eps=eps)
    criterion = nn.CrossEntropyLoss()

    train_acc, train_loss = [], []
    test_acc, test_loss = [], []

    for ep in range(epochs):
        # -------- TRAIN --------
        model.train()
        correct, total = 0, 0
        run_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

        train_loss.append(run_loss / len(train_loader))
        train_acc.append(100 * correct / total)

        # -------- TEST --------
        model.eval()
        correct, total = 0, 0
        run_loss = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)

                run_loss += loss.item()
                _, pred = out.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)

        test_loss.append(run_loss / len(test_loader))
        test_acc.append(100 * correct / total)

        print(
            f"LR={lr} | beta2={beta2} | lam={lam} | Epoch {ep+1:02d} "
            f"| Train Acc={train_acc[-1]:.2f}% | Test Acc={test_acc[-1]:.2f}%"
        )

    return train_acc, train_loss, test_acc, test_loss

# ------------------------------------------------------------
# Sensitivity Analysis — Single Heatmap (Final Test Accuracy)
# ------------------------------------------------------------
def sensitivity_analysis_heatmap():

    lr = 0.1
    epochs = 50

    beta2_list = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]
    lam_list   = [0.001, 0.01, 0.1, 0.2, 0.5, 0.9]

    B, L = len(beta2_list), len(lam_list)

    # rows = beta2, cols = lam
    final_test_acc = np.zeros((B, L))

    for i, beta2 in enumerate(beta2_list):
        for j, lam in enumerate(lam_list):

            print(f"\nRunning beta2={beta2}, lam={lam}")

            _, _, test_acc, _ = train_once(
                lr=lr, beta2=beta2, lam=lam, epochs=epochs
            )

            final_test_acc[i, j] = test_acc[-1]

    # ---------- HEATMAP ----------
    plt.figure()

    im = plt.imshow(final_test_acc, aspect="auto", origin="upper")
    plt.colorbar(im, label="Test Accuracy (%)")

    plt.xticks(range(L), lam_list)
    plt.yticks(range(B), beta2_list)

    plt.xlabel("$\\lambda$")
    plt.ylabel("$\\beta_2$")
    # plt.title(f"SGDA Performance Across $\\beta_2$ and $\\lambda$ Values (lr = {lr})")
    # annotate cells
    for i in range(B):
        for j in range(L):
            val = final_test_acc[i, j]
            plt.text(j, i, f"{val:.2f}%", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sensitivity_analysis_heatmap()
