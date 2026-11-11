"""
V0 — Mapas de NPS em dois painéis separados, com proporções e domínio equivalentes aos das
Figuras 2 e 3 de Lam et al. (2015), para comparação lado a lado no texto.

Mesmo modelo de anc_propagacao.py (integral de Rayleigh com aproximação de Kirchhoff,
discretizada em fontes de Huygens); muda apenas a apresentação:
  - domínio x in [-5,5; 10] m, y in [-10; 10] m (como nas figuras do artigo);
  - um arquivo por frequência, com a mesma proporção de página e a barra de cores à direita.
A escala de cores continua em 55-100 dB (onda incidente de amplitude unitária = 94 dB).

Saídas: ../figuras/v0_mapa_baixa.png e ../figuras/v0_mapa_alta.png
Rode com: python3 v0_mapas_formato_lam.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import hankel1

# As figuras são salvas em ../figuras, relativo a este arquivo
os.chdir(os.path.dirname(os.path.abspath(__file__)))

c0, w = 343.0, 2.0
X_RANGE, Y_RANGE = (-5.5, 10.0), (-10.0, 10.0)
VMIN, VMAX = 55, 100


def campo(freq, theta_deg=0.0, Nx=320, Ny=400, N_ap=300):
    """Campo de pressão complexo (mesma formulação de anc_propagacao.py)."""
    k = 2 * np.pi * freq / c0
    th = np.radians(theta_deg)
    kx, ky = k * np.cos(th), k * np.sin(th)
    y_ap = np.linspace(-w / 2, w / 2, N_ap)
    dy = w / N_ap
    q = -2j * k * np.cos(th) * np.exp(1j * ky * y_ap) * dy
    x = np.linspace(*X_RANGE, Nx)
    y = np.linspace(*Y_RANGE, Ny)
    XX, YY = np.meshgrid(x, y)
    R = np.sqrt(XX[..., None] ** 2 + (YY[..., None] - y_ap) ** 2) + 1e-10
    p_ap = np.tensordot((1j / 4.0) * hankel1(0, k * R), q, axes=([2], [0]))
    p_inc = np.exp(1j * (kx * XX + ky * YY))
    p_ref = np.exp(1j * (-kx * XX + ky * YY))
    p = np.where(XX >= 0, p_ap, p_inc + p_ref - p_ap)
    return XX, YY, p


def painel(freq, arquivo, fontes=False):
    XX, YY, p = campo(freq)
    spl = 20 * np.log10(np.abs(p) / 2e-5 + 1e-30)
    kw = 2 * np.pi * freq * w / c0
    fig = plt.figure(figsize=(6.3, 7.15))
    ax = fig.add_axes([0.11, 0.08, 0.74, 0.84])
    cax = fig.add_axes([0.875, 0.08, 0.025, 0.84])
    im = ax.pcolormesh(XX, YY, spl, cmap="jet", vmin=VMIN, vmax=VMAX, shading="auto")
    ax.add_patch(Rectangle((-0.2, w / 2), 0.2, 10 - w / 2, color="darkgray", zorder=5))
    ax.add_patch(Rectangle((-0.2, -10), 0.2, 10 - w / 2, color="darkgray", zorder=5))
    ax.plot([0, 0], [w / 2, 10], "k-", lw=2, zorder=6)
    ax.plot([0, 0], [-10, -w / 2], "k-", lw=2, zorder=6)
    ax.plot([0, 0], [-w / 2, w / 2], "w--", lw=1.4, alpha=0.7, zorder=6)
    arco = np.linspace(-np.pi / 2, np.pi / 2, 200)
    ax.plot(10 * np.cos(arco), 10 * np.sin(arco), "w:", lw=1.2, alpha=0.6)
    ax.annotate("", xy=(-1.8, 0), xytext=(-5.0, 0),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.8))
    ax.text(-3.4, 0.5, "Onda plana\nincidente", color="white", fontsize=8, ha="center", va="bottom")
    if fontes:
        for sy in (-1.0, -0.5, 0.0, 0.5, 1.0):
            ax.plot(0, sy, "w^", markersize=5, zorder=7)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("NPS (dB)")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"f = {freq:.0f} Hz  (kw = {kw:.2f})".replace(".", ","), fontsize=10)
    os.makedirs("../figuras", exist_ok=True)
    fig.savefig(arquivo, dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    painel(100.0, "../figuras/v0_mapa_baixa.png", fontes=True)
    painel(1000.0, "../figuras/v0_mapa_alta.png")
    print("Figuras salvas: ../figuras/v0_mapa_baixa.png e ../figuras/v0_mapa_alta.png")
