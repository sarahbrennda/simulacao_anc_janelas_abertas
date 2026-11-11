"""
V1 — Controle ótimo estacionário (domínio da frequência) em janela aberta.

Mesmo campo primário da V0 (integral de Rayleigh + aproximação de Kirchhoff,
fontes de Huygens). M fontes secundárias de linha no plano da abertura têm suas
amplitudes determinadas por mínimos quadrados (sem regularização), minimizando
a potência no arco de avaliação (181 pontos, r = 10 m). Seções 4.6 e 5.2 do TCC.

Saídas: ../figuras/v1_anc_mapa.png, ../figuras/v1_anc_limites.png e a tabela
de kw_max (atenuação >= 10 dB) no terminal.
Rode com: python3 v1_anc_freq.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import hankel1
import os

c0 = 343.0
w  = 2.0

# ─── Núcleo: campos no domínio da frequência ─────────────────────────────────

def huygens_sources(freq, theta_deg, N_aperture=300):
    """Fontes de Huygens equivalentes da abertura (ruído primário)."""
    k = 2*np.pi*freq/c0
    th = np.radians(theta_deg)
    y_ap = np.linspace(-w/2, w/2, N_aperture)
    dy = w/N_aperture
    q = -2j*k*np.cos(th)*np.exp(1j*k*np.sin(th)*y_ap)*dy   # Eq. (19) do TCC
    return y_ap, q, k

def field_at(k, xs, ys, x_src, y_src, q_src):
    """Campo em pontos (xs,ys) devido a fontes de linha em (x_src,y_src)."""
    R = np.sqrt((xs[..., None]-x_src)**2 + (ys[..., None]-y_src)**2) + 1e-10
    G = (1j/4.0)*hankel1(0, k*R)
    return G @ q_src, G

def anc_optimal(freq, theta_deg, N_ctrl, r_eval=10.0, N_eval=181):
    """Resolve o problema de mínimos quadrados no arco de avaliação.
    Retorna atenuação [dB] e as amplitudes ótimas das fontes de controle."""
    y_ap, q_ap, k = huygens_sources(freq, theta_deg)

    # Pontos de avaliação no arco semicircular (lado protegido)
    phi = np.linspace(-np.pi/2, np.pi/2, N_eval)
    xe, ye = r_eval*np.cos(phi), r_eval*np.sin(phi)

    # Campo primário no arco
    p_prim, _ = field_at(k, xe, ye, np.zeros_like(y_ap), y_ap, q_ap)

    # Fontes de controle na abertura
    # fontes a meio passo da borda: espaçamento w/M (mesmo arranjo de Lam et al., 2018; 2020)
    if N_ctrl > 1:
        y_c = np.linspace(-w/2 + w/(2*N_ctrl), w/2 - w/(2*N_ctrl), N_ctrl)
    else:
        y_c = np.array([0.0])
    _, A = field_at(k, xe, ye, np.zeros_like(y_c), y_c, np.ones(N_ctrl))

    # Mínimos quadrados: min ||p_prim + A q_c||²
    q_c, *_ = np.linalg.lstsq(A, -p_prim, rcond=None)

    p_res = p_prim + A @ q_c
    atten = 10*np.log10(np.sum(np.abs(p_prim)**2)/(np.sum(np.abs(p_res)**2)+1e-30))
    return atten, q_c, y_c, k

def full_field(freq, theta_deg, q_c=None, y_c=None, Ng=300):
    """Campo de NPS no lado protegido (x>0), com ou sem controle."""
    y_ap, q_ap, k = huygens_sources(freq, theta_deg)
    xg = np.linspace(0.05, 10, Ng)
    yg = np.linspace(-10, 10, Ng)
    XX, YY = np.meshgrid(xg, yg)
    p, _ = field_at(k, XX, YY, np.zeros_like(y_ap), y_ap, q_ap)
    if q_c is not None:
        p_ctrl, _ = field_at(k, XX, YY, np.zeros_like(y_c), y_c, q_c)
        p = p + p_ctrl
    spl = 20*np.log10(np.abs(p)/2e-5 + 1e-30)
    return XX, YY, spl

# Pasta de saída das figuras (../figuras, relativa a este arquivo)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figuras')
os.makedirs(output_dir, exist_ok=True)

# ─── 1) Caso demonstrativo: mapa antes/depois ────────────────────────────────
freq_demo, theta_demo, N_demo = 218.4, 0, 5   # kw=8, 5 fontes (como no artigo)
kw_demo = 2*np.pi*freq_demo*w/c0

atten_demo, q_c, y_c, k = anc_optimal(freq_demo, theta_demo, N_demo)
print(f"Demo: f={freq_demo:.1f} Hz (kw={kw_demo:.1f}), N={N_demo} fontes → atenuação = {atten_demo:.1f} dB")

XX, YY, spl_off = full_field(freq_demo, theta_demo)
_,  _,  spl_on  = full_field(freq_demo, theta_demo, q_c, y_c)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(f'ANC Estacionário na Abertura — f={freq_demo:.0f} Hz (kw={kw_demo:.1f}), '
             f'M = {N_demo} fontes de controle\nAtenuação global no arco: {atten_demo:.1f} dB',
             fontsize=12, fontweight='bold')

for ax, spl, title in [(axes[0], spl_off, 'ANC desligado'),
                       (axes[1], spl_on,  'ANC ligado (amplitudes ótimas)')]:
    im = ax.pcolormesh(XX, YY, spl, cmap='jet', vmin=30, vmax=90, shading='auto')
    ax.add_patch(Rectangle((-0.15, w/2), 0.15, 10-w/2, color='darkgray', zorder=5))
    ax.add_patch(Rectangle((-0.15, -10), 0.15, 10-w/2, color='darkgray', zorder=5))
    phi = np.linspace(-np.pi/2, np.pi/2, 200)
    ax.plot(10*np.cos(phi), 10*np.sin(phi), 'w:', lw=1, alpha=0.6)
    ax.plot(np.zeros_like(y_c), y_c, 'w^', ms=7, zorder=6, label='Fontes de controle')
    plt.colorbar(im, ax=ax, label='NPS (dB)', shrink=0.85)
    ax.set(xlim=(-0.5, 10), ylim=(-10, 10), xlabel='x (m)', ylabel='y (m)', title=title)
    ax.set_aspect('equal')
axes[1].legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(f'{output_dir}/v1_anc_mapa.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figura salva: v1_anc_mapa.png")

# ─── 2) Limite teórico: atenuação vs. kw para vários N ───────────────────────
kw_range = np.linspace(0.5, 60, 600)          # (mantido)
N_list = [2, 3, 5, 8]                          # (mantido)
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(N_list)))
LIMIAR = 10.0                                  # critério — Lam et al. (2015)

plt.figure(figsize=(10, 6.5))
for N, cor in zip(N_list, colors):
    att = np.array([anc_optimal(kw*c0/(2*np.pi*w), 0, N)[0] for kw in kw_range])
    plt.plot(kw_range, att, '-', color=cor, lw=2, label=f'M = {N} fontes')
    plt.axvline(N*np.pi, color=cor, ls=':', alpha=0.5)

plt.axhline(LIMIAR, color='gray', ls='--', alpha=0.7)
plt.text(58, LIMIAR+0.6, f'{LIMIAR:.0f} dB', color='gray',
         fontsize=9, ha='right')                                # ← (2) era x=38
plt.xlabel('Frequência normalizada  kw', fontsize=11)
plt.ylabel('Atenuação global no arco (dB)', fontsize=11)
plt.title('Limite Teórico do ANC em Janela Aberta — Caso Estacionário\n'
          'Onda plana θ=0° | linhas pontilhadas: kw = Mπ '
          '(espaçamento λ/2)', fontsize=12, fontweight='bold')  # ← (3) era N·π
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.xlim(0, 60)
plt.ylim(-2, 120)                                               # ← (4) era 80
plt.tight_layout()
plt.savefig(f'{output_dir}/v1_anc_limites.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figura salva: v1_anc_limites.png")

# Tabela-resumo: frequência máxima controlável (atenuação ≥ 10 dB)
print("\nFrequência máxima com atenuação ≥ 10 dB (w=2m):")
resultados = {}
for N, cor in zip(N_list, colors):
    att = np.array([anc_optimal(kw*c0/(2*np.pi*w), 0, N)[0] for kw in kw_range])
    resultados[N] = att

    abaixo = np.where(att < 10)[0]
    if len(abaixo) == 0:
        kw_max = kw_range[-1]
        print(f"  N={N}: atenuação ≥ 10 dB em toda a faixa — ESTENDER kw_range")
        continue
    if abaixo[0] == 0:
        print(f"  N={N}: nunca atinge 10 dB nesta faixa")
        continue
    kw_max = kw_range[abaixo[0]-1]
    f_max = kw_max*c0/(2*np.pi*w)
    print(f"  N={N}: kw_max ≈ {kw_max:.1f}  →  f_max ≈ {f_max:.0f} Hz  "
          f"(N·π = {N*np.pi:.1f}, razão = {kw_max/(N*np.pi):.2f})")