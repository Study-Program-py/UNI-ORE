import numpy as np
import torch

def sinkhorn_log(cost, a, b, eps=0.05, n_iter=50):
    log_a = (a / a.sum()).clamp_min(1e-12).log()
    log_b = (b / b.sum()).clamp_min(1e-12).log()
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    C = cost / eps
    for _ in range(n_iter):
        f = log_a - torch.logsumexp(g[None, :] - C, dim=1)
        g = log_b - torch.logsumexp(f[:, None] - C, dim=0)
    T = torch.exp(f[:, None] + g[None, :] - C)
    return T

def sinkhorn_numpy(cost, a, b, eps=0.05, n_iter=100):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / a.sum()
    b = b / b.sum()
    log_a, log_b = np.log(a + 1e-12), np.log(b + 1e-12)
    f = np.zeros_like(log_a)
    g = np.zeros_like(log_b)
    C = cost / eps
    from scipy.special import logsumexp
    for _ in range(n_iter):
        f = log_a - logsumexp(g[None, :] - C, axis=1)
        g = log_b - logsumexp(f[:, None] - C, axis=0)
    return np.exp(f[:, None] + g[None, :] - C)

class OTPrototypeLoss(torch.nn.Module):

    def __init__(self, n_proto, dim, eps=0.05, sinkhorn_iter=30, momentum=0.9):
        super().__init__()
        self.n_proto = n_proto
        self.eps = eps
        self.sinkhorn_iter = sinkhorn_iter
        self.momentum = momentum
        self.register_buffer('protos', torch.randn(n_proto, dim) * 0.01)
        self.register_buffer('log_diag', torch.zeros(n_proto, dim))
        self.register_buffer('initialized', torch.zeros(1))

    @torch.no_grad()
    def update_protos(self, feats, labels, n_proto=None):
        n_proto = n_proto or self.n_proto
        lab = labels.clamp(0, n_proto - 1)
        for t in lab.unique():
            sel = feats[lab == t]
            if sel.numel() == 0:
                continue
            c = sel.mean(dim=0)
            if self.initialized.item() < 0.5:
                self.protos[t] = c
            else:
                self.protos[t] = (self.momentum * self.protos[t]
                                  + (1 - self.momentum) * c)
        if self.initialized.item() < 0.5:
            self.initialized.fill_(1.0)

    def forward(self, feats, batch_labels, active_mask):
        sel = feats[active_mask]
        if sel.shape[0] < 4:
            return feats.new_zeros(())
        lab = batch_labels[active_mask].clamp(0, self.n_proto - 1)
        with torch.no_grad():
            self.update_protos(sel.detach(), lab)
        C = ((sel[:, None, :] - self.protos[None, :, :]) ** 2
             * torch.exp(self.log_diag)[None]).sum(dim=2)
        a = torch.full((sel.shape[0],), 1.0 / sel.shape[0], device=feats.device)
        b = torch.full((self.n_proto,), 1.0 / self.n_proto, device=feats.device)
        hist = torch.bincount(lab, minlength=self.n_proto).float()
        b = 0.5 * b + 0.5 * hist / hist.sum().clamp_min(1.0)
        T = sinkhorn_log(C.detach(), a, b, eps=self.eps,
                         n_iter=self.sinkhorn_iter)
        if not torch.isfinite(T).all() or not torch.isfinite(C).all():
            print('OT_DBG sel_finite={} proto_finite={} C_finite={} T_finite={} sel_norm_range=({:.4f},{:.4f})'.format(
                torch.isfinite(sel).all().item(),
                torch.isfinite(self.protos).all().item(),
                torch.isfinite(C).all().item(),
                torch.isfinite(T).all().item(),
                float(sel.norm(dim=1).min()), float(sel.norm(dim=1).max())), flush=True)
        return (T * C).sum() / sel.shape[0]

    @torch.no_grad()
    def update_metric(self, feats, labels):
        lab = labels.clamp(0, self.n_proto - 1)
        for t in lab.unique():
            sel = feats[lab == t]
            if sel.shape[0] < 4:
                continue
            var = sel.var(dim=0).clamp_min(1e-4)
            target = (-var.log()) * 0.1  # gentle log-variance target
            self.log_diag[t] = 0.9 * self.log_diag[t] + 0.1 * target


def sinkhorn_kmeans(X, k, n_iter=10, sinkhorn_iter=50, eps=0.05, seed=0, b=None):
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    centers = [X[rng.randint(N)]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(X[rng.choice(N, p=d2 / d2.sum())])
    C = np.array(centers)
    labels = np.zeros(N, dtype=np.int64)
    a = np.full(N, 1.0 / N)
    b = np.full(k, 1.0 / k) if b is None else np.asarray(b, dtype=np.float64)
    b = b / b.sum()
    for it in range(n_iter):
        cost = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        T = sinkhorn_numpy(cost, a, b, eps=eps, n_iter=sinkhorn_iter)
        labels = T.argmax(axis=1)
        if (T.sum(axis=0) > 1e-9).all():
            M = T.T @ X
            C = M / T.sum(axis=0)[:, None]
        starved = np.where(T.sum(axis=0) <= 1e-9)[0]
        for s in starved:
            far = np.argmax(((X - C[labels]) ** 2).sum(-1))
            C[s] = X[far]
    return labels, C
