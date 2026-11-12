import argparse
import json
import os
from collections import defaultdict
import numpy as np
from scipy import special

def log_dirichlet_constituent(alpha_vec):
    return special.gammaln(alpha_vec.sum()) - special.gammaln(alpha_vec).sum()

class TruncatedHDPVB:
    def __init__(self, n_components=16, gamma=2.0, alpha=10.0, max_iter=300,
                 tol=1e-4, seed=0):
        self.T = n_components
        self.gamma = gamma
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.RandomState(seed)

    def fit(self, X):
        N, D = X.shape
        T = self.T
        assign = self.rng.randint(0, min(T, 8), size=N)
        self.mu = X[self.rng.choice(N, T, replace=False)].copy()
        self.sigma2 = np.tile(X.var(axis=0), (T, 1)) + 1e-6
        self.beta_hat = np.ones((T, 2))
        resp = np.zeros((N, T))
        resp[np.arange(N), assign] = 1.0
        elbo_old = -np.inf
        self.elbo_trace = []
        for it in range(self.max_iter):
            Nk = resp.sum(axis=0) + 1e-10
            cum = np.cumsum(Nk)
            rem = np.maximum(N - cum + Nk, 0.0)
            self.beta_hat[:, 0] = 1.0 + Nk
            self.beta_hat[:, 1] = self.gamma + (N - cum + Nk)
            Ebeta = self.beta_hat[:, 0] / self.beta_hat.sum(axis=1)
            rest = np.concatenate([[1.0], np.cumprod(1 - Ebeta[:-1])])
            Epi = Ebeta * rest
            Epi = Epi / Epi.sum()
            self.Epi = Epi
            for t in range(T):
                r = resp[:, t:t + 1]
                nk = Nk[t]
                self.mu[t] = (r * X).sum(axis=0) / nk
                diff = X - self.mu[t]
                self.sigma2[t] = (resp[:, t:t + 1] * diff ** 2).sum(axis=0) / nk + 1e-6
            Elog_beta = special.digamma(self.beta_hat[:, 0]) - special.digamma(
                self.beta_hat.sum(axis=1))
            Elog_rest = np.concatenate([
                [0.0], np.cumsum(Elog_beta[:-1])])
            Elog_pi = Elog_beta + Elog_rest
            diff = X[:, None, :] - self.mu[None, :, :]
            prec = 1.0 / self.sigma2
            maha = (diff ** 2 * prec[None]).sum(axis=2)
            logdet = np.log(self.sigma2).sum(axis=1)
            loglik = -0.5 * (D * np.log(2 * np.pi) + logdet[None] + maha)
            log_rho = Elog_pi[None] + loglik
            log_norm = special.logsumexp(log_rho, axis=1, keepdims=True)
            resp = np.exp(log_rho - log_norm)
            elbo = float(log_norm.sum())
            a0 = self.beta_hat[:, 0]
            b0 = self.beta_hat[:, 1]
            kl_sb = np.sum(
                log_dirichlet_constituent(np.stack([a0, b0], axis=1))
                - (a0 - 1) * Elog_beta - (b0 - 1) * (
                    special.digamma(b0) - special.digamma(a0 + b0))
                + (a0 - 1.0) * 0.0 + (self.gamma - 1.0) * (
                    special.digamma(b0) - special.digamma(a0 + b0)))
            elbo += -kl_sb
            self.elbo_trace.append(elbo)
            if abs(elbo - elbo_old) < self.tol * abs(elbo_old):
                break
            elbo_old = elbo
        self.resp = resp
        self.Nk = resp.sum(axis=0)
        self.labels = resp.argmax(axis=1)
        self.confidence = resp.max(axis=1)
        self.entropy = -np.sum(resp * np.log(resp + 1e-12), axis=1)
        active = (self.Epi * len(X) >= max(3.0, 0.005 * len(X))).sum()
        self.K_active = int(active)
        self.n_iter = len(self.elbo_trace)
        return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work_dir', required=True)
    ap.add_argument('--prefix', required=True, choices=['retacred', 'tacred', 'fewrel'])
    ap.add_argument('--known_num', type=int, required=True)
    ap.add_argument('--novel_num', type=int, default=6)
    ap.add_argument('--ld', type=int, default=100)
    ap.add_argument('--outlier_ratio', type=float, default=0.05)
    ap.add_argument('--posterior', type=float, default=0.95)
    ap.add_argument('--T', type=int, default=16, help='HDP truncation level')
    ap.add_argument('--gamma', type=float, default=2.0)
    ap.add_argument('--alpha', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    p, wd, K = args.prefix, args.work_dir, args.known_num
    from scipy import spatial
    def find_top_kw_sim(vector, total_rel_num):
        all_dict = {}
        for i in range(total_rel_num):
            cur = np.zeros(total_rel_num)
            cur[i] = 1
            all_dict[i] = 1 - spatial.distance.cosine(cur, vector)
        return dict(sorted(all_dict.items(), key=lambda kv: kv[1], reverse=True))
    sentences = json.load(open(os.path.join(wd, '{}_test_sentence_half.json'.format(p))))
    labels = json.load(open(os.path.join(wd, '{}_test_label_half.json'.format(p))))
    kept = json.load(open(os.path.join(wd, '{}_test_half_rel_representations_kept.json'.format(p))))
    s = [sentences[i] for i in kept]
    label = [labels[i] for i in kept]
    d = np.array(json.load(open(os.path.join(
        wd, '{}_test_S_vectors_half_ld{}.json'.format(p, args.ld)))))
    assert len(d) == len(s)
    print('D_u:', len(s))

    fst = [next(iter(find_top_kw_sim(d[i], K).values())) for i in range(len(d))]
    threshold = sorted(fst)[int(len(d) * args.outlier_ratio) + 1]
    idx_out = [i for i in range(len(d)) if fst[i] < threshold]
    o_remain = d[idx_out]
    o_label = [label[i] for i in idx_out]
    o_sen = [s[i] for i in idx_out]
    print('outliers:', len(o_remain))

    hdp = TruncatedHDPVB(n_components=args.T, gamma=args.gamma, alpha=args.alpha,
                         seed=args.seed).fit(o_remain)
    print('HDP-SVI: active components {} (T={}), {} iters, ELBO {:.2f}->{:.2f}'.format(
        hdp.K_active, args.T, hdp.n_iter, hdp.elbo_trace[0], hdp.elbo_trace[-1]))

    id2rel = {v: k for k, v in json.load(open(os.path.join(wd, 'rel2id.json'))).items()}
    conf = hdp.confidence
    lab = hdp.labels
    ent = hdp.entropy
    log2K = np.log(args.T)

    report = {'threshold': float(threshold), 'outliers': len(o_remain),
              'T': args.T, 'gamma': args.gamma, 'alpha': args.alpha,
              'K_novel_est': hdp.K_active, 'n_iter': hdp.n_iter,
              'elbo': [float(hdp.elbo_trace[0]), float(hdp.elbo_trace[-1])],
              'posterior': args.posterior, 'clusters': []}
    weak_sen, weak_lab, weak_w = [], [], []
    novel_maj, purity_list = [], []
    for c in range(args.T):
        members = [j for j in range(len(o_remain)) if lab[j] == c]
        if not members or hdp.Epi[c] * len(o_remain) < max(3.0, 0.005 * len(o_remain)):
            continue
        m_lab = [o_label[j] for j in members]
        uniq, cnt = np.unique(m_lab, return_counts=True)
        majority = uniq[np.argmax(cnt)]
        purity = float(cnt.max() / len(members))
        belongs2novel = majority in range(K, K + args.novel_num)
        if belongs2novel:
            novel_maj.append(majority)
        kept_members = [j for j in members if conf[j] >= args.posterior]
        rel_dist = defaultdict(int)
        for l in m_lab:
            rel_dist[id2rel[l]] += 1
        print('comp{}: mass={:.4f} #={} pure={:.3f} maj={}({}) novel={} kept={}'.format(
            c, hdp.Epi[c], len(members), purity, majority, id2rel[majority],
            belongs2novel, len(kept_members)))
        report['clusters'].append({
            'component': c, 'mass': float(hdp.Epi[c]), 'size': len(members),
            'weak_label': K + c, 'majority_label': id2rel[majority],
            'majority_is_novel': bool(belongs2novel), 'purity': round(purity, 3),
            'mean_confidence': float(np.mean([conf[j] for j in members])),
            'relation_dist': dict(sorted(rel_dist.items(), key=lambda kv: kv[1], reverse=True))})
        for j in kept_members:
            weak_sen.append(o_sen[j])
            weak_lab.append(K + c)
            weak_w.append(float(1.0 - ent[j] / log2K))
        purity_list.append([len(members), purity])

    print('# found novel relations:', len(set(novel_maj)))
    if purity_list:
        wcp = float(sum(n * q for n, q in purity_list) /
                    sum(n for n, _ in purity_list))
        print('weighted cluster purity: {:.3f}'.format(wcp))
        report['weighted_cluster_purity'] = round(wcp, 3)
    report['found_novel_relations'] = len(set(novel_maj))

    active_components = [cl['component'] for cl in report['clusters']]
    remap = {comp: min(k, args.novel_num - 1)
             for k, comp in enumerate(active_components)}
    weak_lab = [K + remap[l - K] for l in weak_lab]
    for cl in report['clusters']:
        cl['weak_label'] = K + remap[cl['component']]

    tr_sen = json.load(open(os.path.join(wd, '{}_train_sentence_half.json'.format(p))))
    tr_lab = json.load(open(os.path.join(wd, '{}_train_label_half.json'.format(p))))
    kept_tr = json.load(open(os.path.join(wd, '{}_train_half_rel_representations_kept.json'.format(p))))
    tr_sen = [tr_sen[i] for i in kept_tr]
    tr_lab = [tr_lab[i] for i in kept_tr]
    tr_attr = json.load(open(os.path.join(
        wd, '{}_train_S_vectors_half_ld{}.json'.format(p, args.ld))))
    assert len(tr_attr) == len(tr_sen)
    by_label = defaultdict(list)
    for i, l in enumerate(tr_lab):
        by_label[l].append([tr_sen[i], tr_attr[i]])
    new_sen, new_lab = [], []
    for key in sorted(by_label):
        one_hot = np.zeros(K)
        one_hot[key] = 1
        sims = {k2: 1 - spatial.distance.cosine(one_hot, it[1])
                for k2, it in enumerate(by_label[key])}
        for k2 in dict(sorted(sims.items(), key=lambda kv: kv[1], reverse=True)):
            new_sen.append(by_label[key][k2][0])
            new_lab.append(key)
    new_sen.extend(weak_sen)
    new_lab.extend(weak_lab)

    for name, obj in [('{}_train_sentence_half_novel_added_hdpsvi.json'.format(p), new_sen),
                      ('{}_train_label_half_novel_added_hdpsvi.json'.format(p), new_lab),
                      ('{}_train_weak_weights_hdpsvi.json'.format(p), weak_w),
                      (P + '_hdpsvi_report.json', report)]:
        with open(os.path.join(wd, name), 'w') as f:
            json.dump(obj, f)
    print('written {}_train_*_novel_added_hdpsvi.json ({} instances, {} weak)'.format(
        p, len(new_sen), len(weak_sen)))


if __name__ == '__main__':
    main()
