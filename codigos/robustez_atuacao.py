"""
Robustez do controle estacionario a erros de atuacao (apoio ao Cap. 5, V2).

Calcula as amplitudes otimas da V1 (beta = 0) e, SEM recalcula-las, aplica
pequenos erros aleatorios de amplitude e fase as fontes de controle,
avaliando a atenuacao global esperada. Mostra que as atenuacoes muito altas de
baixa frequencia dependem da reproducao quase exata do campo: elas colapsam com
erros minimos, enquanto as atenuacoes proximas do limite espacial sao robustas.

Formulacao identica a de v1_anc_freq.py. Gera figuras/v2_robustez.png.
Rode com: python3 robustez_atuacao.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import hankel1
import os

# As figuras são salvas em ../figuras, relativo a este arquivo
os.chdir(os.path.dirname(os.path.abspath(__file__)))

c0, w = 343.0, 2.0
rng = np.random.default_rng(42)

def huygens_sources(freq, theta_deg, N_ap=300):
    k = 2*np.pi*freq/c0
    th = np.radians(theta_deg)
    y = np.linspace(-w/2, w/2, N_ap)
    q = -2j*k*np.cos(th)*np.exp(1j*k*np.sin(th)*y)*(w/N_ap)
    return y, q, k

def green(k, xe, ye, xs, ys):
    R = np.sqrt((xe[...,None]-xs)**2 + (ye[...,None]-ys)**2) + 1e-10
    return (1j/4.0)*hankel1(0, k*R)

def ctrl_positions(M):
    return np.linspace(-w/2 + w/(2*M), w/2 - w/(2*M), M) if M > 1 else np.array([0.0])

def setup(freq, theta_deg, M, r=10.0, N=181, N_ap=300):
    y_ap, q_ap, k = huygens_sources(freq, theta_deg, N_ap)
    phi = np.linspace(-np.pi/2, np.pi/2, N)
    xe, ye = r*np.cos(phi), r*np.sin(phi)
    d = green(k, xe, ye, np.zeros_like(y_ap), y_ap) @ q_ap
    yc = ctrl_positions(M)
    A = green(k, xe, ye, np.zeros_like(yc), yc)
    qc, *_ = np.linalg.lstsq(A, -d, rcond=None)
    return dict(d=d, A=A, qc=qc, yc=yc, xe=xe, ye=ye, k=k)

def atten(d, e):
    return 10*np.log10(np.sum(np.abs(d)**2)/(np.sum(np.abs(e)**2)+1e-30))

def perturbed_atten(S, sig_amp, sig_phase_rad, n_trials=600):
    """Atenuacao global esperada: media da potencia residual sobre n_trials repeticoes."""
    d, A, qc = S["d"], S["A"], S["qc"]
    M = len(qc)
    Pe = 0.0
    for _ in range(n_trials):
        a  = rng.normal(0, sig_amp, M)
        ph = rng.normal(0, sig_phase_rad, M)
        qt = qc*(1+a)*np.exp(1j*ph)
        e = d + A @ qt
        Pe += np.sum(np.abs(e)**2)
    Pe /= n_trials
    return 10*np.log10(np.sum(np.abs(d)**2)/(Pe+1e-30))

kw_to_f = lambda kw: kw*c0/(2*np.pi*w)
kws = [1, 8, 26]
labels = {1:  r"$kw=1$  (atenuação ideal $\approx$ 155 dB)",
          8:  r"$kw=8$  (atenuação ideal $\approx$ 50 dB)",
          26: r"$kw=26$ (atenuação ideal $\approx$ 12 dB)"}
S = {kw: setup(kw_to_f(kw), 0.0, 5) for kw in kws}

eps = np.logspace(-3, -0.5, 25)          # sigma de 0.1% a ~32%
colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(kws)))

fig, ax = plt.subplots(figsize=(7.5, 5.2))
print("kw   ideal  |  aten. media (dB) em erro de 1% / 3% / 10%")
for kw, cor in zip(kws, colors):
    curve = np.array([perturbed_atten(S[kw], e, e) for e in eps])
    ax.semilogx(eps*100, curve, "-", color=cor, lw=2.2, label=labels[kw])
    ide = atten(S[kw]["d"], S[kw]["d"] + S[kw]["A"] @ S[kw]["qc"])
    g = lambda t: curve[np.argmin(np.abs(eps-t))]
    print(f"{kw:3d} {ide:6.1f}  |  {g(0.01):6.1f} / {g(0.03):6.1f} / {g(0.10):6.1f}")

ax.axhline(10, color="gray", ls="--", lw=1.2)
ax.text(eps[-1]*100, 11, "critério 10 dB", color="gray", ha="right", fontsize=9)
ax.set_xlabel(r"Nível de erro de atuação  $\sigma$  (amplitude em %, fase em $\sigma/100$ rad)", fontsize=11)
ax.set_ylabel("Atenuação global esperada (dB)", fontsize=11)
ax.set_ylim(0, 60)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
os.makedirs("../figuras", exist_ok=True)
out = "../figuras/v2_robustez.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("Figura salva:", out)


# ---------------------------------------------------------------------------
# Tabela de apoio ao texto (Cap. 4, Subsecao "Modelo do erro de atuacao", e
# Cap. 5, Subsecao 5.3.5). Valores EXATOS de erro (1, 3 e 10 %), com semente
# propria, para que os numeros citados no texto sejam reprodutiveis.
# Nao altera a figura acima.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
print("\n[Tabela do texto] M = 5, incidencia normal, 600 repeticoes por ponto")
print("kw   ideal | amplitude+fase: 1% / 3% / 10%")
for kw in kws:
    ide = atten(S[kw]["d"], S[kw]["d"] + S[kw]["A"] @ S[kw]["qc"])
    a = [perturbed_atten(S[kw], e, e) for e in (0.01, 0.03, 0.10)]
    print(f"{kw:3d} {ide:7.1f} | {a[0]:6.1f} / {a[1]:6.1f} / {a[2]:6.1f}")
