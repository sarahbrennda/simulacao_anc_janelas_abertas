"""
V2 — ANC estacionário em janela aberta com regularização do esforço de controle.

Derivada da V1 (mesma geometria, mesmos pontos de avaliação, mesma formulação
matricial). A única mudança é a solução das fontes secundárias:

    q_s = -(G^H G + beta I)^(-1) G^H d          (Lam et al., 2018, Eq. 4)

com beta normalizado pelo maior autovalor de G^H G:

    beta = alpha * lambda_max(G^H G)

alpha é adimensional e é a variável de análise. alpha = 0 reproduz a V1.

Métricas de esforço:
  E_rel = 10 log10( ||q(alpha)||^2 / ||q(0)||^2 )   [dB re V1]
  E_abs = 10 log10( ||q(alpha)||^2 / |Q_ap|^2 )     [dB re abertura]
  onde Q_ap = soma das fontes de Huygens da abertura (intensidade líquida).
  E_abs = 0 dB  ~ uma única fonte reproduzindo a intensidade da própria abertura.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import hankel1
import os

c0 = 343.0
w  = 2.0
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figuras')
os.makedirs(output_dir, exist_ok=True)

# ─── Núcleo (idêntico à V1) ──────────────────────────────────────────────────

def huygens_sources(freq, theta_deg, N_aperture=300):
    """Fontes de Huygens equivalentes da abertura (ruído primário)."""
    k = 2*np.pi*freq/c0
    th = np.radians(theta_deg)
    y_ap = np.linspace(-w/2, w/2, N_aperture)
    dy = w/N_aperture
    q = -2j*k*np.cos(th)*np.exp(1j*k*np.sin(th)*y_ap)*dy
    return y_ap, q, k

def field_at(k, xs, ys, x_src, y_src, q_src):
    """Campo em (xs,ys) devido a fontes de linha em (x_src,y_src)."""
    R = np.sqrt((xs[..., None]-x_src)**2 + (ys[..., None]-y_src)**2) + 1e-10
    G = (1j/4.0)*hankel1(0, k*R)
    return G @ q_src, G

def ctrl_positions(N):
    """Fontes a meio passo da borda: espaçamento w/N -> critério λ/2 em kw = Nπ."""
    if N == 1:
        return np.array([0.0])
    return np.linspace(-w/2 + w/(2*N), w/2 - w/(2*N), N)

def eval_points(r_eval=10.0, N_eval=181):
    phi = np.linspace(-np.pi/2, np.pi/2, N_eval)
    return r_eval*np.cos(phi), r_eval*np.sin(phi)

# ─── Novidade da V2: solução regularizada ────────────────────────────────────

def anc_regularized(freq, theta_deg, N_ctrl, alphas, r_eval=10.0, N_eval=181):
    """Resolve o problema regularizado para vários alpha de uma vez (via SVD).

    Retorna dicionário com atenuação [dB], E_rel [dB], E_abs [dB] (arrays no
    mesmo formato de `alphas`) e as amplitudes q (N_alpha x N_ctrl).
    """
    y_ap, q_ap, k = huygens_sources(freq, theta_deg)
    xe, ye = eval_points(r_eval, N_eval)
    d, _ = field_at(k, xe, ye, np.zeros_like(y_ap), y_ap, q_ap)

    y_c = ctrl_positions(N_ctrl)
    _, G = field_at(k, xe, ye, np.zeros_like(y_c), y_c, np.ones(N_ctrl))

    # G = U S V^H ;  q(beta) = -V diag(s/(s^2+beta)) U^H d
    U, s, Vh = np.linalg.svd(G, full_matrices=False)
    lam_max = s[0]**2                      # maior autovalor de G^H G
    Uhd = U.conj().T @ d

    alphas = np.atleast_1d(alphas).astype(float)
    betas = alphas*lam_max
    filt = s[None, :]/(s[None, :]**2 + betas[:, None])       # (Na, N)
    q = -(filt*Uhd[None, :]) @ Vh.conj()                      # (Na, N)

    # Resíduo e atenuação
    e = d[None, :] + q @ G.T
    Pd = np.sum(np.abs(d)**2)
    Pe = np.sum(np.abs(e)**2, axis=1)
    atten = 10*np.log10(Pd/(Pe + 1e-30))

    # Esforço
    q0 = -(Vh.conj().T @ (Uhd/s))                 # alpha = 0 (= V1)
    eff = np.sum(np.abs(q)**2, axis=1)
    E_rel = 10*np.log10(eff/np.sum(np.abs(q0)**2))
    Q_ap = np.abs(np.sum(q_ap))**2
    E_abs = 10*np.log10(eff/Q_ap)

    cond = s[0]/s[-1]
    return dict(atten=atten, E_rel=E_rel, E_abs=E_abs, q=q, y_c=y_c,
                k=k, cond=cond, lam_max=lam_max)

def kw_to_f(kw):
    return kw*c0/(2*np.pi*w)

def kw_max_10dB(kw_range, att, limiar=10.0):
    """Maior kw antes da primeira queda abaixo do limiar (critério da V1)."""
    abaixo = np.where(att < limiar)[0]
    if len(abaixo) == 0:
        return np.nan            # nunca cai: estender a faixa
    if abaixo[0] == 0:
        return 0.0               # nunca atinge o limiar
    return kw_range[abaixo[0]-1]

# ─── Parâmetros da varredura ─────────────────────────────────────────────────
theta = 0.0
LIMIAR = 10.0
kw_range = np.linspace(0.5, 60, 600)
N_list = [2, 3, 5, 8]
alphas = np.array([0.0, 1e-6, 1e-4, 1e-2, 1e-1])
alpha_labels = ['α = 0 (V1)', 'α = 1e-6', 'α = 1e-4', 'α = 1e-2', 'α = 1e-1']
cores_a = plt.cm.plasma(np.linspace(0.0, 0.85, len(alphas)))

# Varre tudo uma vez: R[N] -> arrays (N_kw x N_alpha)
R = {}
for N in N_list:
    att = np.zeros((len(kw_range), len(alphas)))
    Er = np.zeros_like(att); Ea = np.zeros_like(att); cn = np.zeros(len(kw_range))
    for i, kw in enumerate(kw_range):
        r = anc_regularized(kw_to_f(kw), theta, N, alphas)
        att[i], Er[i], Ea[i], cn[i] = r['atten'], r['E_rel'], r['E_abs'], r['cond']
    R[N] = dict(att=att, E_rel=Er, E_abs=Ea, cond=cn)
    print(f"N={N}: varredura concluída")

# ─── Fig. 1: atenuação e esforço vs kw (N = 5) ───────────────────────────────
Nd = 5
fig, ax = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
for j, (lab, cor) in enumerate(zip(alpha_labels, cores_a)):
    ax[0].plot(kw_range, R[Nd]['att'][:, j], color=cor, lw=2, label=lab)
    ax[1].plot(kw_range, R[Nd]['E_abs'][:, j], color=cor, lw=2, label=lab)
for a in ax:
    a.axvline(Nd*np.pi, color='gray', ls=':', alpha=0.7)
    a.grid(alpha=0.3)
ax[0].axhline(LIMIAR, color='gray', ls='--', alpha=0.7)
ax[0].set_ylabel('Atenuação global (dB)')
ax[0].set_ylim(-2, 175)
ax[0].legend(fontsize=9)
ax[0].set_title(f'V2 — Efeito da regularização β = α·λ_max(GᴴG) | M = {Nd} fontes, θ = 0°\n'
                f'linha pontilhada: kw = Mπ (espaçamento λ/2)', fontweight='bold')
ax[1].set_ylabel('Esforço E_abs (dB re |Q_abertura|²)')
ax[1].set_xlabel('Frequência normalizada kw')
ax[1].set_xlim(0, 60)
plt.tight_layout()
plt.savefig(f'{output_dir}/v2_atenuacao_esforco_kw.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()

# ─── Fig. 2: curva de compromisso atenuação × esforço (N = 5) ────────────────
alpha_fino = np.logspace(-10, 1, 200)
kw_sel = [2, 8, 14, 20]
cores_k = plt.cm.viridis(np.linspace(0.1, 0.85, len(kw_sel)))
plt.figure(figsize=(9, 6.5))
for kw, cor in zip(kw_sel, cores_k):
    r = anc_regularized(kw_to_f(kw), theta, Nd, alpha_fino)
    plt.plot(r['E_rel'], r['atten'], color=cor, lw=2, label=f'kw = {kw}')
    # marca as décadas de alpha
    for a_mk in [1e-8, 1e-6, 1e-4, 1e-2, 1e0]:
        i = np.argmin(np.abs(np.log10(alpha_fino) - np.log10(a_mk)))
        plt.plot(r['E_rel'][i], r['atten'][i], 'o', color=cor, ms=4)
plt.axhline(LIMIAR, color='gray', ls='--', alpha=0.7)
plt.xlabel('Esforço relativo E_rel (dB re solução da V1)')
plt.ylabel('Atenuação global (dB)')
plt.title(f'V2 — Compromisso atenuação × esforço | M = {Nd}\n'
          f'pontos: α = 1e-8, 1e-6, 1e-4, 1e-2, 1 (da direita para a esquerda)',
          fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/v2_compromisso_atenuacao_esforco.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()

# ─── Fig. 3: kw_max (10 dB) vs N para cada alpha ─────────────────────────────
plt.figure(figsize=(9, 6.5))
for j, (lab, cor) in enumerate(zip(alpha_labels, cores_a)):
    km = [kw_max_10dB(kw_range, R[N]['att'][:, j]) for N in N_list]
    plt.plot(N_list, km, 'o-', color=cor, lw=2, label=lab)
Nl = np.array([1, 9])
plt.plot(Nl, Nl*np.pi, 'k--', lw=1.2, label='kw = Mπ (λ/2)')
plt.xlabel('Número de fontes de controle M')
plt.ylabel('kw_máx (atenuação ≥ 10 dB)')
plt.title('V2 — Frequência máxima controlável vs M para cada α', fontweight='bold')
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/v2_kwmax_vs_N.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()

# ─── Tabela-resumo ───────────────────────────────────────────────────────────
print("\nkw_máx (atenuação ≥ 10 dB) e razão kw_máx/(Nπ):")
print(f"{'N':>3} | " + " | ".join(f"{l:>14}" for l in alpha_labels))
for N in N_list:
    linha = []
    for j in range(len(alphas)):
        km = kw_max_10dB(kw_range, R[N]['att'][:, j])
        linha.append(f"{km:5.1f} ({km/(N*np.pi):.2f})")
    print(f"{N:>3} | " + " | ".join(f"{c:>14}" for c in linha))

print("\nNúmero de condição de G e esforço da V1 (N=5) em kw selecionados:")
for kw in [1, 2, 8, 14, 20, 30]:
    i = np.argmin(np.abs(kw_range - kw))
    print(f"  kw={kw_range[i]:5.1f}: cond(G)={R[5]['cond'][i]:9.2e}  "
          f"E_abs(V1)={R[5]['E_abs'][i, 0]:6.1f} dB  "
          f"Att(V1)={R[5]['att'][i, 0]:6.1f} dB  "
          f"Att(α=1e-4)={R[5]['att'][i, 2]:6.1f} dB")

# ─── Tabela 5 do TCC: atenuação em kw exatos (M = 5) e número de condição ────
print("\n[Tabela 5 do TCC] Atenuação global (dB), M = 5, e cond(G):")
print(f"{'kw':>4} | {'cond(G)':>9} | " + " | ".join(f"{l:>10}" for l in alpha_labels))
for kw in [1, 2, 4, 8, 16, 26]:
    r = anc_regularized(kw_to_f(kw), theta, 5, alphas)
    print(f"{kw:>4} | {r['cond']:9.1e} | " + " | ".join(f"{a:10.1f}" for a in r['atten']))
