"""
V1 — Reprodução das configurações de fontes de Lam et al. (2015) com o modelo semianalítico.

Lam et al. (2015) usam elementos finitos (COMSOL, 2D, camada perfeitamente casada) e
testam configurações de 1, 2, 3 e 5 fontes secundárias entre q1..q5, em
y = -1; -0,5; 0; 0,5; 1 m, para ângulos de incidência 0, 30, 45 e 60 graus.
Aqui as mesmas configurações são resolvidas com o núcleo da V1 (fontes de Huygens +
função de Green 2D, arco de 181 pontos a 10 m, mínimos quadrados sem regularização).

Observação sobre a comparação: nas figuras de Lam et al. (2015) as curvas de atenuação são
poligonais cujos vértices parecem corresponder a 100, 200, 500 e 1000 Hz (kw ≈ 3,7; 7,3;
18,3; 36,6, com kw = 1 em 27,3 Hz), isto é, o cruzamento em 10 dB lido nelas é uma
interpolação linear entre poucas frequências. Por isso o script calcula kw_max de duas formas:
(i) em malha fina (1 pt a cada ~0,05 em kw) e (ii) nas mesmas frequências, com interpolação
linear, como nas figuras de Lam.

Saídas: figuras/v1_lam_reproducao_theta0.png, figuras/v1_lam_reproducao_m5.png e as tabelas de kw_max no terminal.
Rode com: python3 v1_reproducao_lam2015.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import hankel1

# As figuras são salvas em ../figuras, relativo a este arquivo
os.chdir(os.path.dirname(os.path.abspath(__file__)))

c0, w = 343.0, 2.0
Q = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}      # posições de q1..q5 (m)

CONFIGS = {                                           # nome: (índices das fontes, rótulo)
    "M=1 (q3)":                  [3],
    "M=2, d=2 m (q1,q5)":        [1, 5],
    "M=2, d=1 m (q2,q4)":        [2, 4],
    "M=3, d=1 m (q1,q3,q5)":     [1, 3, 5],
    "M=3, d=0,5 m (q2,q3,q4)":   [2, 3, 4],
    "M=5, d=0,5 m (q1..q5)":     [1, 2, 3, 4, 5],
}
ANGULOS = (0, 30, 45, 60)
kws = np.linspace(0.5, 36.6, 722)                     # mesma faixa de Lam (kw < 36,6)


def kw_to_f(kw):
    return kw * c0 / (2 * np.pi * w)


def huygens_sources(freq, theta_deg, N_ap=300):
    k = 2 * np.pi * freq / c0
    th = np.radians(theta_deg)
    y = np.linspace(-w / 2, w / 2, N_ap)
    q = -2j * k * np.cos(th) * np.exp(1j * k * np.sin(th) * y) * (w / N_ap)
    return y, q, k


def green(k, xe, ye, xs, ys):
    R = np.sqrt((xe[..., None] - xs) ** 2 + (ye[..., None] - ys) ** 2) + 1e-10
    return (1j / 4.0) * hankel1(0, k * R)


phi = np.linspace(-np.pi / 2, np.pi / 2, 181)
XE, YE = 10 * np.cos(phi), 10 * np.sin(phi)


def atenuacao(kw, theta, yc):
    y_ap, q_ap, k = huygens_sources(kw_to_f(kw), theta)
    d = green(k, XE, YE, np.zeros_like(y_ap), y_ap) @ q_ap
    G = green(k, XE, YE, np.zeros_like(yc), yc)
    q, *_ = np.linalg.lstsq(G, -d, rcond=None)
    return 10 * np.log10(np.sum(abs(d) ** 2) / np.sum(abs(d + G @ q) ** 2))


def curva(theta, idx):
    yc = np.array([Q[i] for i in idx])
    return np.array([atenuacao(k, theta, yc) for k in kws])


KW_LAM = np.array([1.83, 3.66, 7.33, 18.3, 36.6])    # 50, 100, 200, 500, 1000 Hz


def kw_max_esparso(theta, yc, limiar=10.0):
    """kw_max obtido como nas figuras de Lam: interpolação linear entre KW_LAM."""
    a = np.array([atenuacao(k, theta, yc) for k in KW_LAM])
    for i in range(len(KW_LAM) - 1):
        if a[i] >= limiar > a[i + 1]:
            return KW_LAM[i] + (a[i] - limiar) / (a[i] - a[i + 1]) * (KW_LAM[i + 1] - KW_LAM[i])
    return np.nan


def kw_max(a, limiar=10.0):
    """Primeiro kw em que a atenuação cai abaixo do limiar (critério conservador)."""
    i = np.where(a < limiar)[0]
    return kws[i[0] - 1] if len(i) else kws[-1]


if __name__ == "__main__":
    curvas = {(n, th): curva(th, idx) for n, idx in CONFIGS.items() for th in ANGULOS}

    print("kw_max (primeiro cruzamento abaixo de 10 dB)")
    print(f"{'configuração':28s}" + "".join(f"{th:>8d}°" for th in ANGULOS))
    for n in CONFIGS:
        print(f"{n:28s}" + "".join(f"{kw_max(curvas[(n, th)]):9.1f}" for th in ANGULOS))

    print("\nkw_max como nas figuras de Lam (interpolação linear entre 50, 100, 200, 500, 1000 Hz)")
    print(f"{'configuração':28s}" + "".join(f"{th:>8d}°" for th in ANGULOS))
    for n, idx in CONFIGS.items():
        yc_ = np.array([Q[i] for i in idx])
        print(f"{n:28s}" + "".join(f"{kw_max_esparso(th, yc_):9.1f}" for th in ANGULOS))

    os.makedirs("../figuras", exist_ok=True)
    estilo = dict(figsize=(6.4, 5.5))

    # (a) incidência normal, todas as configurações de Lam (cf. Fig. 11 do artigo)
    fig, ax = plt.subplots(**estilo)
    cores = plt.cm.tab10(np.linspace(0, 1, 10))[[3, 0, 1, 2, 4, 5]]
    ordem = ["M=5, d=0,5 m (q1..q5)", "M=3, d=0,5 m (q2,q3,q4)", "M=3, d=1 m (q1,q3,q5)",
             "M=2, d=1 m (q2,q4)", "M=1 (q3)", "M=2, d=2 m (q1,q5)"]
    for n, cor in zip(ordem, cores):
        ax.plot(kws, curvas[(n, 0)], color=cor, lw=1.6, label=n)
    ax.axhline(10, color="k", ls="--", lw=1.0)
    ax.set_xlim(0, 40); ax.set_ylim(0, 40)
    ax.set_xlabel(r"$kw$"); ax.set_ylabel("Atenuação global (dB)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig("../figuras/v1_lam_reproducao_theta0.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # (b) cinco fontes, vários ângulos (cf. Fig. 10 do artigo); pontos = frequências de Lam
    fig, ax = plt.subplots(**estilo)
    n5 = "M=5, d=0,5 m (q1..q5)"
    yc5 = np.array([Q[i] for i in CONFIGS[n5]])
    cores5 = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    for th, cor in zip(ANGULOS, cores5):
        ax.plot(kws, curvas[(n5, th)], color=cor, lw=1.6, label=rf"$\theta = {th}^\circ$")
        ax.plot(KW_LAM, [atenuacao(k, th, yc5) for k in KW_LAM], "o", color=cor, ms=5)
    ax.plot([], [], "o", color="gray", ms=5, label="frequências de Lam")
    ax.axhline(10, color="k", ls="--", lw=1.0)
    ax.set_xlim(0, 40); ax.set_ylim(0, 40)
    ax.set_xlabel(r"$kw$"); ax.set_ylabel("Atenuação global (dB)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="upper right")
    plt.tight_layout()
    plt.savefig("../figuras/v1_lam_reproducao_m5.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Figuras salvas: ../figuras/v1_lam_reproducao_theta0.png e ..._m5.png")
