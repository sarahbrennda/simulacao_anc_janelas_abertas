"""
Simulação 2D da Propagação Sonora através de uma Janela Aberta
Baseado em: B. Lam et al. (2015) "The Physical Limits of Active Noise Control of Open Windows"

Reproduz os padrões de nível de pressão sonora (NPS) para ondas planas incidindo
em uma abertura retangular em uma parede rígida, em diferentes frequências.

Método: integral de Rayleigh com a aproximação de Kirchhoff, discretizada em
fontes elementares de Huygens (Seção 2.4 do TCC). É uma formulação semianalítica
para o lado protegido (x > 0); não é equivalente ao modelo por elementos finitos
do artigo, que resolve o espalhamento completo (ver Seção 4.4 do TCC).

Geometria (do artigo, Figura 1):
  - Parede rígida em x=0, de y=-10 a y=10, exceto abertura de w=2m centrada em y=0
  - Domínio: x de -10 a 10, y de -10 a 10
  - Onda plana incidente vinda da esquerda (x < 0) com ângulo theta
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# Pasta de saída das figuras (../figuras, relativa a este arquivo)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figuras')
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Parâmetros físicos ───────────────────────────────────────────────────────
c0   = 343.0   # velocidade do som no ar [m/s]
rho0 = 1.21    # densidade do ar [kg/m³]

# ─── Geometria (conforme artigo, Figura 1) ────────────────────────────────────
w = 2.0   # largura da janela [m]

# ─── Domínio de cálculo ───────────────────────────────────────────────────────
x_min, x_max = -10.0, 10.0
y_min, y_max = -10.0, 10.0
Nx, Ny = 400, 400

x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
XX, YY = np.meshgrid(x, y)   # shape (Ny, Nx)

# ─── Funções auxiliares ───────────────────────────────────────────────────────

def compute_field_huygens(freq, theta_deg, Nx=400, Ny=400,
                           x_range=(-10, 10), y_range=(-10, 10),
                           N_aperture=300, w=2.0):
    """
    Calcula o campo de pressão total usando o Princípio de Huygens.
    """
    from scipy.special import hankel1

    omega = 2 * np.pi * freq
    k = omega / c0
    lam = c0 / freq
    theta = np.radians(theta_deg)

    kx = k * np.cos(theta)
    ky = k * np.sin(theta)

    y_ap = np.linspace(-w/2, w/2, N_aperture)
    x_ap = np.zeros(N_aperture)
    dy_ap = w / N_aperture

    p_inc_aperture = np.exp(1j * (kx * x_ap + ky * y_ap))

    x_obs = np.linspace(x_range[0], x_range[1], Nx)
    y_obs = np.linspace(y_range[0], y_range[1], Ny)
    XX, YY = np.meshgrid(x_obs, y_obs)

    p_incident = np.exp(1j * (kx * XX + ky * YY))

    # Vetorizado: distância de cada ponto do grid a cada fonte de Huygens
    R = np.sqrt((XX[..., None] - x_ap[None, None, :])**2 +
                (YY[..., None] - y_ap[None, None, :])**2) + 1e-10
    # Intensidade das fontes de Huygens (integral de Rayleigh + aproximação de Kirchhoff):
    # q_i = -2ik cos(theta) p_inc(0, y_i) dy   (Eq. eq:huygens_discreto do Cap. 2)
    q = -2j * k * np.cos(theta) * p_inc_aperture * dy_ap
    G = (1j/4.0) * hankel1(0, k * R)
    p_aperture = np.tensordot(G, q, axes=([2], [0]))

    p_total = np.zeros((Ny, Nx), dtype=complex)
    mask_right = XX >= 0
    mask_left  = XX < 0

    p_reflected = np.exp(1j * (-kx * XX + ky * YY))

    p_total[mask_right] = p_aperture[mask_right]
    p_total[mask_left] = (p_incident[mask_left] +
                          p_reflected[mask_left] -
                          p_aperture[mask_left])

    return XX, YY, p_total, k, lam


def spl_from_pressure(p, p_ref=2e-5):
    """Converte pressão complexa para NPS em dB."""
    return 20 * np.log10(np.abs(p) / p_ref + 1e-30)


kw_100  = 2*np.pi*100*w/c0   # = 3.66
kw_1000 = 2*np.pi*1000*w/c0  # = 36.64

cases = [
    (kw_100,  0, r'Baixa frequência ($\lambda \gg w$, $kw=3.66$, $\theta=0°$)',  'fig2_low_freq'),
    (kw_1000, 0, r'Alta frequência ($w \gg \lambda$, $kw=36.64$, $\theta=0°$)', 'fig3_high_freq'),
]

print("Iniciando simulações...")
print(f"Parâmetros: w={w}m, c0={c0}m/s")
print()

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('Padrão de Propagação Sonora através de Abertura em Parede Rígida\n'
             '(Reprodução do artigo B. Lam et al., 2015)',
             fontsize=13, fontweight='bold')

for idx, (kw, theta_deg, titulo, label) in enumerate(cases):
    freq = kw * c0 / (2 * np.pi * w)
    print(f"Simulando: kw={kw:.2f}, θ={theta_deg}°, f={freq:.1f} Hz")

    XX, YY, p_total, k, lam = compute_field_huygens(
        freq=freq, theta_deg=theta_deg,
        Nx=300, Ny=300,
        x_range=(-10, 10), y_range=(-10, 10),
        N_aperture=300, w=w
    )

    spl = spl_from_pressure(p_total)
    ax = axes[idx]

    vmin, vmax = 55, 100
    im = ax.pcolormesh(XX, YY, spl, cmap='jet', vmin=vmin, vmax=vmax,
                       shading='auto')

    wall_thickness = 0.2
    ax.add_patch(Rectangle((-wall_thickness, w/2), wall_thickness, 10 - w/2,
                            color='darkgray', zorder=5))
    ax.add_patch(Rectangle((-wall_thickness, -10), wall_thickness, 10 - w/2,
                            color='darkgray', zorder=5))
    ax.plot([0, 0], [w/2, 10], 'k-', lw=2, zorder=6)
    ax.plot([0, 0], [-10, -w/2], 'k-', lw=2, zorder=6)
    ax.plot([0, 0], [-w/2, w/2], 'w--', lw=1.5, alpha=0.7, zorder=6,
            label=f'Abertura w={w}m')

    theta_arc = np.linspace(-np.pi/2, np.pi/2, 200)
    r_hemi = 10.0
    ax.plot(r_hemi * np.cos(theta_arc), r_hemi * np.sin(theta_arc),
            'w:', lw=1.2, alpha=0.6, label='Arco semicircular r = 10 m')

    ax.annotate('', xy=(-3, 0), xytext=(-7, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.text(-6.5, 0.5, 'Onda\nplana\nincidente', color='white', fontsize=8,
            ha='center', va='bottom')

    if idx == 0:
        source_y = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        for sy in source_y:
            ax.plot(0, sy, 'w^', markersize=6, zorder=7)

    plt.colorbar(im, ax=ax, label='NPS (dB)', shrink=0.85)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('x (m)', fontsize=10)
    ax.set_ylabel('y (m)', fontsize=10)
    ax.set_title(f'{titulo}\nf = {freq:.1f} Hz, λ = {lam:.2f} m', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(False)

    print(f"  → faixa de NPS: {spl.min():.1f} a {spl.max():.1f} dB")

plt.tight_layout()
output_path = os.path.join(FIG_DIR, 'anc_window_propagation.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"\nFigura salva em: {output_path}")

# ─── Figura adicional: múltiplos kw para comparação ───────────────────────────
print("\nGerando figura com múltiplos kw...")

kw_values = [1, 4, 8, 16]
theta_deg = 0

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Padrão de Radiação Sonora — Onda Plana θ=0° em Diferentes Frequências\n'
              '(Baseado em B. Lam et al., 2015 — Abertura w=2m)',
              fontsize=13, fontweight='bold')

for idx, kw in enumerate(kw_values):
    freq = kw * c0 / (2 * np.pi * w)
    lam  = c0 / freq
    ax = axes2.flat[idx]

    print(f"  kw={kw}, f={freq:.1f} Hz, λ={lam:.2f} m")

    XX, YY, p_total, k, _ = compute_field_huygens(
        freq=freq, theta_deg=theta_deg,
        Nx=300, Ny=300,
        x_range=(-10, 10), y_range=(-10, 10),
        N_aperture=300, w=w
    )

    spl = spl_from_pressure(p_total)
    vmin, vmax = 55, 100
    im = ax.pcolormesh(XX, YY, spl, cmap='jet', vmin=vmin, vmax=vmax,
                       shading='auto')

    ax.add_patch(Rectangle((-0.2, w/2), 0.2, 10-w/2, color='darkgray', zorder=5))
    ax.add_patch(Rectangle((-0.2, -10), 0.2, 10-w/2, color='darkgray', zorder=5))
    ax.plot([0,0], [w/2, 10], 'k-', lw=2, zorder=6)
    ax.plot([0,0], [-10, -w/2], 'k-', lw=2, zorder=6)

    theta_arc = np.linspace(-np.pi/2, np.pi/2, 200)
    ax.plot(10*np.cos(theta_arc), 10*np.sin(theta_arc), 'w:', lw=1.2, alpha=0.5)

    ax.annotate('', xy=(-2, 0), xytext=(-5, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    plt.colorbar(im, ax=ax, label='NPS (dB)', shrink=0.8)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('x (m)', fontsize=9)
    ax.set_ylabel('y (m)', fontsize=9)
    ax.set_title(f'kw = {kw}  |  f = {freq:.0f} Hz  |  λ = {lam:.2f} m',
                 fontsize=10, fontweight='bold')
    ax.set_aspect('equal')

plt.tight_layout()
output2 = os.path.join(FIG_DIR, 'anc_window_multifreq.png')
plt.savefig(output2, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Figura salva em: {output2}")