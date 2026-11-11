# Simulação de cancelamento ativo de ruído em janelas abertas

Códigos em Python do Trabalho de Conclusão de Curso **"Simulação de cancelamento ativo de ruído em janelas abertas"**, de Sarah Brennda Bispo Almeida Gomes, orientado pelo Prof. Dr. Cláudio Antônio Cardoso (Engenharia Física, Departamento de Física, UFSCar, 2026).

O trabalho desenvolve um modelo numérico bidimensional do controle ativo de ruído (ANC) em uma abertura de 2 m em parede rígida, tomando como referência Lam et al. (2015). O modelo tem três etapas:

- **V0**: campo transmitido pela abertura, pela integral de Rayleigh com a aproximação de Kirchhoff, discretizada em fontes de Huygens;
- **V1**: fontes secundárias na abertura, com amplitudes ótimas por mínimos quadrados (controle estacionário, domínio da frequência);
- **V2**: solução regularizada (penalização do esforço das fontes) e ensaio de sensibilidade a erros de atuação.

## Como executar

Requisitos: Python 3, com NumPy, SciPy e Matplotlib.

```bash
pip install -r requirements.txt
python3 codigos/v1_anc_freq.py
```

Cada script é independente e pode ser executado diretamente. As figuras são salvas na pasta `figuras/` e as tabelas são impressas no terminal. Os scripts também rodam no Google Colab (basta enviar o arquivo e executar `!python3 nome_do_arquivo.py`).

Executar todos os scripts leva cerca de 10 minutos em um computador comum; os mais demorados são `v1_reproducao_lam2015.py` (cerca de 4 min) e `v1_comparacao_lam2015.py` (cerca de 2 min).

## Scripts e resultados correspondentes no TCC

| Script | Etapa | O que calcula | No TCC |
|---|---|---|---|
| `codigos/anc_propagacao.py` | V0 | Mapas de NPS do campo transmitido; transição do regime monopolar ao diretivo (kw = 1, 4, 8, 16) | Figura 4 |
| `codigos/v0_mapas_formato_lam.py` | V0 | Mapas de NPS em 100 Hz e 1000 Hz, no formato das figuras de Lam et al. (2015) | Figura 3(a) |
| `codigos/v1_anc_freq.py` | V1 | Caso demonstrativo (kw = 8, M = 5) e atenuação global em função de kw para M = 2, 3, 5 e 8 | Figuras 5 e 6, Tabela 3 |
| `codigos/v1_comparacao_lam2015.py` | V1 | kw máximo (10 dB) para o arranjo deste trabalho e o de Lam et al. (2015), em quatro ângulos de incidência | Tabela 4 |
| `codigos/v1_reproducao_lam2015.py` | V1 | Configurações de 1, 2, 3 e 5 fontes de Lam et al. (2015); kw máximo em malha fina e nas frequências do artigo | Figuras 7(b) e 8(b) |
| `codigos/v2_anc_beta.py` | V2 | Solução regularizada: atenuação, esforço e kw máximo em função de α | Figuras 9, 10 e 11, Tabelas 5 e 6 |
| `codigos/robustez_atuacao.py` | V2 | Sensibilidade da solução da V1 a erros de amplitude e fase (600 repetições, semente 42) | Figura 12, Tabela 7 |
| `codigos/verificacoes_numericas.py` | V1/V2 | Convergência quanto à discretização e esforço por fonte | Subseções 5.2.3 e 5.3.1 |

As figuras em `figuras/` são as saídas diretas dos scripts. As versões usadas no texto do TCC são as mesmas, em alguns casos sem o título interno do gráfico.

## Parâmetros comuns

- Largura da abertura: w = 2 m; velocidade do som: c0 = 343 m/s
- Fontes elementares na abertura: 300
- Arco de avaliação: 181 pontos, raio de 10 m, no lado protegido
- Faixa de frequência normalizada: 0,5 ≤ kw ≤ 60 (600 valores)
- Critério de desempenho: atenuação global de pelo menos 10 dB

## Testado com

Python 3 com NumPy 2.5.3, SciPy 1.18.1 e Matplotlib 3.11.2. Os valores impressos pelos scripts coincidem com os das tabelas do TCC.

## Referência principal

LAM, B.; ELLIOTT, S. J.; CHEER, J.; GAN, W.-S. The Physical Limits of Active Noise Control of Open Windows. In: *Proceedings of the 12th Western Pacific Acoustics Conference (WESPAC)*. Singapore, 2015. p. 184–189.

## Como citar

GOMES, Sarah Brennda Bispo Almeida. **Simulação de cancelamento ativo de ruído em janelas abertas**. 2026. Trabalho de Conclusão de Curso (Graduação em Engenharia Física) – Universidade Federal de São Carlos, São Carlos, 2026.
