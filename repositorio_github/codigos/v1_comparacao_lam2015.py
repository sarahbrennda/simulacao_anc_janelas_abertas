"""
V1 — Comparação com Lam et al. (2015): efeito do arranjo de fontes e do ângulo de incidência.

Usa o mesmo núcleo da V1 (fontes de Huygens + função de Green 2D, arco de 181
pontos a 10 m, mínimos quadrados sem regularização) e calcula o maior kw com
atenuação global >= 10 dB para dois arranjos de M = 5 fontes:
  - arranjo deste trabalho: meio passo da borda, espaçamento w/M = 0,4 m;
  - arranjo de Lam et al. (2015): fontes em y = -1; -0,5; 0; 0,5; 1 m (d = 0,5 m).
Compara com a regra de aliasing espacial kd < 2*pi/(1 + sin(theta)) (Lam et al., 2018b).
"""
import numpy as np
from scipy.special import hankel1

c0, w = 343.0, 2.0


def kw_to_f(kw):
    return kw * c0 / (2 * np.pi * w)


def huygens_sources(freq, theta_deg, N_aperture=300):
    k = 2 * np.pi * freq / c0
    th = np.radians(theta_deg)
    y_ap = np.linspace(-w / 2, w / 2, N_aperture)
    q = -2j * k * np.cos(th) * np.exp(1j * k * np.sin(th) * y_ap) * (w / N_aperture)
    return y_ap, q, k


def field_at(k, xs, ys, x_src, y_src, q_src):
    R = np.sqrt((xs[..., None] - x_src) ** 2 + (ys[..., None] - y_src) ** 2) + 1e-10
    G = (1j / 4.0) * hankel1(0, k * R)
    return G @ q_src, G


phi = np.linspace(-np.pi / 2, np.pi / 2, 181)
XE, YE = 10 * np.cos(phi), 10 * np.sin(phi)


def atenuacao(kw, theta, y_c):
    y_ap, q_ap, k = huygens_sources(kw_to_f(kw), theta)
    d, _ = field_at(k, XE, YE, np.zeros_like(y_ap), y_ap, q_ap)
    _, G = field_at(k, XE, YE, np.zeros_like(y_c), y_c, np.ones(len(y_c)))
    q, *_ = np.linalg.lstsq(G, -d, rcond=None)
    return 10 * np.log10(np.sum(abs(d) ** 2) / np.sum(abs(d + G @ q) ** 2))


kws = np.linspace(0.5, 60, 1200)


def kw_max(theta, y_c, limiar=10.0):
    a = np.array([atenuacao(k, theta, y_c) for k in kws])
    i = np.where(a < limiar)[0]
    return kws[i[0] - 1]


arranjos = {
    'Este trabalho (d = 0,4 m)': np.linspace(-0.8, 0.8, 5),
    'Lam et al. 2015 (d = 0,5 m)': np.linspace(-1.0, 1.0, 5),
}
for nome, y_c in arranjos.items():
    d = y_c[1] - y_c[0]
    print(nome)
    for th in (0, 30, 45, 60):
        regra = 2 * np.pi * w / (d * (1 + np.sin(np.radians(th))))
        print(f'  theta = {th:2d} deg: kw_max = {kw_max(th, y_c):5.2f}   regra de aliasing: kw < {regra:5.2f}')
