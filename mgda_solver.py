import torch

def mgda_weights(grads, max_iter=25):
    G = torch.stack([g.reshape(-1) for g in grads])
    norms = G.norm(dim=1)
    M = G.shape[0]
    if M == 1 or (norms < 1e-12).any():
        return [1.0 / M] * M
    Gn = G
    alpha = torch.ones(M, device=G.device) / M
    for _ in range(max_iter):
        g = alpha @ Gn
        ips = Gn @ g
        idx = int(torch.argmin(ips).item())
        gamma = 2.0 / (_ + 2)
        alpha = (1 - gamma) * alpha
        alpha[idx] += gamma
        if (ips - ips.min()).abs().max() < 1e-10:
            break
    return [float(a) for a in alpha]

class MGDABalancer:

    def __init__(self, ema_beta=0.0):
        self.ema_beta = ema_beta
        self.ema = None
        self.last_alpha = None

    def weights(self, feats, losses, retain_graph=True):
        grads = []
        for L in losses:
            g = torch.autograd.grad(L, feats, retain_graph=retain_graph)[0]
            grads.append(g.detach().reshape(-1))
        alpha = mgda_weights(grads)
        if self.ema_beta > 0:
            a = torch.tensor(alpha)
            if self.ema is None:
                self.ema = a
            else:
                self.ema = self.ema_beta * self.ema + (1 - self.ema_beta) * a
            alpha = [float(x) for x in (self.ema / self.ema.sum())]
        self.last_alpha = alpha
        return alpha
