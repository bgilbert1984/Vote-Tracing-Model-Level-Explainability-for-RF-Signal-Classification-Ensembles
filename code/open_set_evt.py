
import numpy as np

def softmax(logits, axis=-1):
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)

def oscr_curve_from_scores(s_known_maxprob, yhat_known, y_true_known, s_unknown_maxprob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.99, 40)
    fpr_u, ccr = [], []
    for t in thresholds:
        fpr = float(np.mean(s_unknown_maxprob >= t)) if len(s_unknown_maxprob) else 0.0
        accept = (s_known_maxprob >= t)
        correct = (yhat_known == y_true_known) & accept
        ccr.append(float(np.mean(correct)) if len(y_true_known) else 0.0)
        fpr_u.append(fpr)
    return np.array(fpr_u), np.array(ccr), np.array(thresholds)

def auc_trapz(x, y):
    order = np.argsort(x)
    xs = x[order]; ys = y[order]
    if len(xs) < 2:
        return 0.0
    return float(np.trapz(ys, xs))

def _fit_weibull_tail(x, tail_frac=0.2):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0.0)]
    if len(x) < 10:
        return 1.0, float(np.mean(x)) if len(x) else 1.0
    x_sorted = np.sort(x)
    n = len(x_sorted)
    start = max(0, int((1.0 - tail_frac) * n))
    xt = x_sorted[start:]
    ranks = (np.arange(len(xt)) + 1.0) / (len(xt) + 1.0)
    eps = 1e-9
    y = np.log(-np.log(1.0 - np.clip(ranks, eps, 1.0 - eps)))
    X = np.log(xt + eps)
    A = np.vstack([X, np.ones_like(X)]).T
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    k = float(sol[0]); b = float(sol[1])
    if k <= 0 or not np.isfinite(k):
        k = 1.0
    lam = float(np.exp(-b / k))
    lam = lam if np.isfinite(lam) and lam > 0 else float(np.mean(xt))
    return k, lam

def weibull_tail_cdf(x, k, lam):
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    return 1.0 - np.exp(- (x / lam) ** k)

def evt_accept_unknown_scores(smax_known, smax_unknown, alphas=None, tail_frac=0.2):
    if alphas is None:
        alphas = np.linspace(0.01, 0.99, 40)
    u_known = 1.0 - np.asarray(smax_known, dtype=np.float64)
    u_unknown = 1.0 - np.asarray(smax_unknown, dtype=np.float64)
    k, lam = _fit_weibull_tail(u_known, tail_frac=tail_frac)
    t_known = weibull_tail_cdf(u_known, k, lam)
    t_unknown = weibull_tail_cdf(u_unknown, k, lam)
    return t_known, t_unknown, k, lam, np.asarray(alphas, dtype=np.float64)

def oscr_from_evt(yhat_known, y_true_known, t_known, t_unknown, alphas):
    fpr_u, ccr = [], []
    for a in alphas:
        accept_known = (t_known <= a)
        correct = (yhat_known == y_true_known) & accept_known
        ccr.append(float(np.mean(correct)) if len(y_true_known) else 0.0)
        fpr = float(np.mean(t_unknown <= a)) if len(t_unknown) else 0.0
        fpr_u.append(fpr)
    return np.array(fpr_u), np.array(ccr)
