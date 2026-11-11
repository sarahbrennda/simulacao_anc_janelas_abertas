"""
Verificacoes numericas de apoio ao Cap. 5 (V1/V2).

1) Convergencia de kw_max e da atenuacao-demo quanto ao numero de fontes
   elementares da abertura (N_ap), de pontos do arco (N) e do raio do arco (r).
2) Distribuicao do esforco entre as M fontes de controle (a norma total
   esconde o comportamento individual?).
3) As atenuacoes muito altas de baixa frequencia vem de amplitudes grandes?

Formulacao identica a de v1_anc_freq.py (Kirchhoff + Rayleigh 2D, minimos
quadrados sobre o arco de avaliacao). Rode com: python3 verificacoes_numericas.py
"""
import numpy as np
from scipy.special import hankel1

c0, w = 343.0, 2.0

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

def anc_optimal(freq, theta_deg, M, r=10.0, N=181, N_ap=300):
    y_ap, q_ap, k = huygens_sources(freq, theta_deg, N_ap)
    phi = np.linspace(-np.pi/2, np.pi/2, N)
    xe, ye = r*np.cos(phi), r*np.sin(phi)
    d = green(k, xe, ye, np.zeros_like(y_ap), y_ap) @ q_ap
    yc = ctrl_positions(M)
    A = green(k, xe, ye, np.zeros_like(yc), yc)
    qc, *_ = np.linalg.lstsq(A, -d, rcond=None)
    e = d + A @ qc
    atten = 10*np.log10(np.sum(np.abs(d)**2)/(np.sum(np.abs(e)**2)+1e-30))
    Eabs = 10*np.log10(np.sum(np.abs(qc)**2)/np.abs(np.sum(q_ap))**2)
    return dict(atten=atten, qc=qc, Eabs=Eabs, cond=np.linalg.cond(A), Qap=np.abs(np.sum(q_ap)))

kw_to_f = lambda kw: kw*c0/(2*np.pi*w)

def kwmax(M, theta=0.0, r=10.0, N=181, N_ap=300, thr=10.0,
          grid=np.linspace(0.5, 60, 600)):
    att = np.array([anc_optimal(kw_to_f(kw), theta, M, r, N, N_ap)["atten"] for kw in grid])
    idx = np.where(att >= thr)[0]
    if len(idx) == 0:
        return np.nan
    brk = np.where(np.diff(idx) > 1)[0]
    last = idx[brk[0]] if len(brk) else idx[-1]
    return grid[last]

if __name__ == "__main__":
    print("="*68)
    print("1) CONVERGENCIA de kw_max (M=5, theta=0, criterio 10 dB)")
    print("="*68)
    print(f"{'referencia (N_ap=300, N=181, r=10)':40s} {kwmax(5):7.2f}")
    print(f"{'N_ap=600':40s} {kwmax(5, N_ap=600):7.2f}")
    print(f"{'N_ap=1200':40s} {kwmax(5, N_ap=1200):7.2f}")
    print(f"{'N=361':40s} {kwmax(5, N=361):7.2f}")
    print(f"{'N=721':40s} {kwmax(5, N=721):7.2f}")
    print(f"{'r=20 m':40s} {kwmax(5, r=20.0):7.2f}")
    print(f"{'r=40 m':40s} {kwmax(5, r=40.0):7.2f}")

    print("\n" + "="*68)
    print("   Atenuacao-demo (kw=8, M=5) sob refinamento")
    print("="*68)
    f8 = kw_to_f(8.0)
    for lbl, kw in [("ref", {}), ("N_ap=600", dict(N_ap=600)), ("N_ap=1200", dict(N_ap=1200)),
                    ("N=361", dict(N=361)), ("N=721", dict(N=721)),
                    ("r=20", dict(r=20.0)), ("r=40", dict(r=40.0))]:
        print(f"  {lbl:10s} atten = {anc_optimal(f8, 0.0, 5, **kw)['atten']:6.2f} dB")

    print("\n" + "="*68)
    print("2) ESFORCO POR FONTE (M=5, theta=0): |q| individual / media")
    print("="*68)
    print(f"{'kw':>4s} {'atten':>7s} {'Eabs':>7s} {'cond':>9s}   |q|/media por fonte      max/med")
    for kw in [1, 2, 4, 8, 14, 20, 26]:
        d = anc_optimal(kw_to_f(kw), 0.0, 5)
        rel = np.abs(d["qc"])/np.abs(d["qc"]).mean()
        print(f"{kw:4d} {d['atten']:7.1f} {d['Eabs']:7.2f} {d['cond']:9.1e}   "
              f"[{' '.join(f'{v:4.2f}' for v in rel)}]   {rel.max():5.2f}")

    print("\n" + "="*68)
    print("3) ATEN. ALTA EM BAIXO kw VEM DE AMPLITUDE GRANDE? (max|q|/|Qap|)")
    print("="*68)
    for kw in [1, 2, 4, 8]:
        d = anc_optimal(kw_to_f(kw), 0.0, 5)
        maxq = 20*np.log10(np.abs(d["qc"]).max()/d["Qap"])
        print(f"  kw={kw:2d}: atten={d['atten']:6.1f} dB | Eabs={d['Eabs']:6.2f} dB | "
              f"max|q|/|Qap|={maxq:6.2f} dB | cond={d['cond']:.1e}")
