"""
PROJETO: ECONOMETRIA DE FRONTEIRA - FINANÇAS E DEEP LEARNING
AUTOR: LUIZ TIAGO WILCKE
DESCRIÇÃO: Implementação de modelos avançados de risco (GARCH/VaR) e
           redes neurais híbridas (Autoencoders + LSTM).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from arch import arch_model
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# --- Módulo 1: Motor de Risco Financeiro (GARCH, VaR, ES) ---

class MotorRiscoFinanceiro:
    def __init__(self, retornos):
        self.retornos = retornos
        self.modelo_ajustado = None
        
    def estimar_garch_robusto(self):
        # Estima GARCH(1,1) com distribuição t-Student para capturar caudas pesadas
        modelo = arch_model(self.retornos * 100, vol='Garch', p=1, q=1, dist='t')
        self.modelo_ajustado = modelo.fit(disp='off')
        return self.modelo_ajustado

    def calcular_metricas_risco(self, alfa=0.99):
        volatilidade_condicional = self.modelo_ajustado.conditional_volatility / 100
        graus_liberdade = self.modelo_ajustado.params['nu']
        media_estimada = self.modelo_ajustado.params['mu'] / 100
        
        # VaR Paramétrico t-Student
        percentil_t = stats.t.ppf(alfa, df=graus_liberdade)
        valor_em_risco = -media_estimada - volatilidade_condicional * percentil_t
        
        # Expected Shortfall (ES)
        densidade_t = stats.t.pdf(percentil_t, df=graus_liberdade)
        es_aux = densidade_t / (1 - alfa)
        es_ajustado = es_aux * (graus_liberdade + percentil_t**2) / (graus_liberdade - 1)
        expected_shortfall = -media_estimada + volatilidade_condicional * es_ajustado
        
        return valor_em_risco, expected_shortfall

# --- Módulo 2: Deep Learning Híbrido (Autoencoder + LSTM) ---

class AutoencoderFatores(nn.Module):
    def __init__(self, num_entradas, dimensao_latente=3):
        super().__init__()
        self.codificador = nn.Sequential(
            nn.Linear(num_entradas, 16),
            nn.ReLU(),
            nn.Linear(16, dimensao_latente)
        )
        self.decodificador = nn.Sequential(
            nn.Linear(dimensao_latente, 16),
            nn.ReLU(),
            nn.Linear(16, num_entradas)
        )
        
    def forward(self, x):
        z = self.codificador(x)
        x_reconstruido = self.decodificador(z)
        return x_reconstruido, z

class MotorPrevisaoLSTM(nn.Module):
    def __init__(self, entrada_dim=3, oculta_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(entrada_dim, oculta_dim, batch_first=True)
        self.saida = nn.Linear(oculta_dim, entrada_dim)
        
    def forward(self, x):
        saida_lstm, _ = self.lstm(x)
        return self.saida(saida_lstm[:, -1, :])

def orquestrar_deep_learning_avancado(dados_painel, passos_futuros=5):
    # 1. Pré-processamento
    normalizador = StandardScaler()
    dados_normalizados = torch.tensor(normalizador.fit_transform(dados_painel), dtype=torch.float32)
    
    # 2. Treinar Autoencoder para Extração de Fatores
    dim_entrada = dados_painel.shape[1]
    ae_modelo = AutoencoderFatores(dim_entrada)
    otimizador_ae = torch.optim.Adam(ae_modelo.parameters(), lr=0.01)
    criterio = nn.MSELoss()
    
    for epoca in range(100):
        recons, latente = ae_modelo(dados_normalizados)
        perda = criterio(recons, dados_normalizados)
        otimizador_ae.zero_grad()
        perda.backward()
        otimizador_ae.step()
        
    # 3. Preparar sequências para LSTM sobre os fatores latentes
    _, fatores_latentes = ae_modelo(dados_normalizados)
    fatores_latentes = fatores_latentes.detach()
    
    # Criar janelas de tempo (look_back = 4)
    janela = 4
    X, Y = [], []
    for i in range(len(fatores_latentes) - janela):
        X.append(fatores_latentes[i:i+janela].numpy())
        Y.append(fatores_latentes[i+janela].numpy())
    
    X_tens = torch.tensor(X, dtype=torch.float32)
    Y_tens = torch.tensor(Y, dtype=torch.float32)
    
    # 4. Treinar LSTM
    lstm_modelo = MotorPrevisaoLSTM()
    otimizador_lstm = torch.optim.Adam(lstm_modelo.parameters(), lr=0.01)
    
    for epoca in range(150):
        pred = lstm_modelo(X_tens)
        perda_lstm = criterio(pred, Y_tens)
        otimizador_lstm.zero_grad()
        perda_lstm.backward()
        otimizador_lstm.step()
        
    return FaktorenLatentes = fatores_latentes

def main():
    print("================================================================")
    print("SISTEMA DE FRONTEIRA: FINANÇAS E DEEP LEARNING - LUIZ TIAGO WILCKE")
    print("================================================================\n")
    
    # Simulação de Dados Financeiros
    np.random.seed(42)
    n_obs = 500
    retornos_simulados = np.random.normal(0.001, 0.02, n_obs)
    
    # 1. Análise de Risco
    motor_risco = MotorRiscoFinanceiro(retornos_simulados)
    motor_risco.estimar_garch_robusto()
    var_99, es_99 = motor_risco.calcular_metricas_risco()
    
    print(f"VaR Condicional (99%): {var_99[-1]*100:.4f}%")
    print(f"Expected Shortfall (99%): {es_99[-1]*100:.4f}%")
    
    # 2. Deep Learning em Painel de Ativos
    print("\nIniciando Extração de Fatores via Autoencoder e Previsão LSTM...")
    dados_ativos = np.random.randn(n_obs, 10) # 10 ativos correlacionados
    fatores = orquestrar_deep_learning_avancado(dados_ativos)
    print("Fatores Latentes Extraídos com Sucesso.")
    
    print("\n================================================================")
    print("AUTOR: LUIZ TIAGO WILCKE")
    print("================================================================")

if __name__ == "__main__":
    main()
