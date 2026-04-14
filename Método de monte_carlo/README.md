# Suíte Avançada de Métodos de Monte Carlo para Econometria

**Autor:** Luiz Tiago Wilcke  
**Data:** Abril 2026

Este repositório contém uma implementação sofisticada e de alta performance dos métodos de Monte Carlo aplicados à teoria e prática econométrica. O projeto foi desenvolvido com o objetivo de fornecer ferramentas robustas para simulação, inferência e validação de modelos complexos, abrangendo tanto a abordagem clássica (frequentista) quanto a moderna (Bayesiana).

## Conteúdo do Projeto

O projeto está dividido em duas frentes tecnológicas principais:

### 1. Motor de Simulação em Python (`monte_carlo_engine.py`)
Implementação modular em Python com foco em escalabilidade e diagnósticos profundos.
- **Integração de Monte Carlo**: Técnicas de redução de variância como Variáveis Antitéticas e Amostragem por Importância.
- **MQO e GMM**: Avaliação de viés, consistência e poder de testes sob violações (Heterocedasticidade, Autocorrelação e Endogeneidade).
- **MCMC (Markov Chain Monte Carlo)**: Algoritmos Metropolis-Hastings e Amostrador de Gibbs para regressão linear e logística.
- **Bayesiana de Fronteira**: Regressão Quantílica Bayesiana e Slice Sampling.
- **Painel Dinâmico**: Estudo do Viés de Nickell em modelos de efeitos fixos.

### 2. Motor de Simulação em R (`monte_carlo_engine.R`)
Foco em inferência estatística avançada e visualização de dados.
- **Bootstrapping**: Residual, Pairs e Wild Bootstrap para inferência robusta.
- **Séries Temporais**: Simulação de processos SARIMA, testes de raiz unitária (ADF) e cointegração (Engle-Granger).
- **Bayesiana Probit**: Implementação de Metropolis-Hastings para modelos de escolha qualitativa.
- **Análise de Estabilidade**: Simulação do poder de testes de quebras estruturais.

## Requisitos e Execução

### Python
Dependências: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `joblib`, `statsmodels`.
```bash
python3 monte_carlo_engine.py
```

### R
Dependências: `foreach`, `doParallel`, `MASS`, `ggplot2`, `sandwich`, `lmtest`, `tseries`.
```bash
Rscript monte_carlo_engine.R
```

---
Este projeto reflete o estado da arte em simulação computacional aplicada às ciências econômicas, garantindo rigor matemático e eficiência algorítmica.
