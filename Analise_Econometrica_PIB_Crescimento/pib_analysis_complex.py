"""
PROJETO: ANALISE RIGOROSA DO PIB E DINAMICA ECONOMICA
AUTOR: LUIZ TIAGO WILCKE
DESCRIÇÃO: Este script implementa uma suíte de ferramentas econométricas avançadas para análise de PIB,
           incluindo Modelos de Séries Temporais (VAR, VECM, SARIMAX), Filtro de Kalman, 
           e Redes Neurais Recorrentes (LSTM).
TECNICAS: Baseadas no livro "Econometria Aplicada: Uma Abordagem Rigorosa com Python e R".
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, VECM, DynamicFactorMQ
from statsmodels.tsa.stattools import adfuller, coint, q_stat
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from arch import arch_model
import warnings
import os

warnings.filterwarnings('ignore')

# Configurações de Estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

class MotorDadosEconomicos:
    """
    Classe responsável pela simulação e pré-processamento de séries temporais macroeconômicas.
    Simula dinâmicas de PIB, Inflação e Taxas de Juros com componentes estocásticos.
    """
    def __init__(self, n_observacoes=500, seed=42):
        np.random.seed(seed)
        self.n = n_observacoes
        self.index = pd.date_range(start='2000-01-01', periods=n_observacoes, freq='Q')
        
    def simular_pib_estrutural(self):
        """
        Simula o PIB com tendência estocástica e quebras estruturais.
        Y_t = Y_{t-1} + g + epsilon_t
        """
        g = 0.005 # Crescimento trimestral de 0.5%
        erro = np.random.normal(0, 0.01, self.n)
        pib = np.zeros(self.n)
        pib[0] = 100 # Valor inicial
        
        for t in range(1, self.n):
            # Adiciona uma quebra estrutural em t=250
            if t == 250:
                pib[t] = pib[t-1] * 0.95 + erro[t] # Choque negativo (Recessão)
            else:
                pib[t] = pib[t-1] * (1 + g) + erro[t]
                
        return pd.Series(pib, index=self.index, name='PIB')

    def gerar_dataset_multivariado(self):
        """
        Gera um sistema de variáveis: PIB, Inflação, SELIC.
        """
        pib = self.simular_pib_estrutural()
        inflacao = pd.Series(np.random.normal(0.01, 0.005, self.n).cumsum() + 2, 
                            index=self.index, name='Inflacao')
        selic = pd.Series(0.5 * inflacao + 0.02 + np.random.normal(0, 0.001, self.n), 
                         index=self.index, name='SELIC')
        
        df = pd.concat([pib, inflacao, selic], axis=1)
        return df

class AnalisadorSeriesTemporais:
    """
    Motor Econométrico para Modelagem Univariada e Multivariada.
    """
    def __init__(self, data):
        self.data = data
        self.results = {}

    def realizar_testes_estacionariedade(self):
        print("\n--- Testes de Raiz Unitária (ADF) ---")
        for col in self.data.columns:
            res = adfuller(self.data[col])
            print(f"Variável {col}: Estatística ADF={res[0]:.4f}, p-valor={res[1]:.4f}")
            if res[1] > 0.05:
                print(f"  > Recomenda-se diferenciação para {col}.")

    def identificar_sarimax(self, target='PIB'):
        print(f"\n--- Modelagem SARIMAX para {target} ---")
        model = SARIMAX(self.data[target], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        results = model.fit(disp=False)
        self.results['sarimax'] = results
        print(results.summary())
        return results

    def modelar_var_vecm(self):
        print("\n--- Modelagem Multivariada: VAR/VECM ---")
        # Teste de Cointegração de Johansen
        rank_test = select_coint_rank(self.data, det_order=0, k_ar_diff=1)
        print(f"Posto de Cointegração Sugerido: {rank_test.rank}")
        
        if rank_test.rank > 0:
            print("Aplicando VECM (Vector Error Correction Model)...")
            vecm = VECM(self.data, k_ar_diff=1, coint_rank=rank_test.rank, deterministic='ci')
            res = vecm.fit()
            self.results['vecm'] = res
            print(res.summary())
        else:
            print("Aplicando VAR (Vector Autoregression)...")
            var = VAR(self.data)
            res = var.fit(maxlags=4, ic='aic')
            self.results['var'] = res
            print(res.summary())

    def plotar_irf(self):
        if 'var' in self.results:
            irf = self.results['var'].irf(10)
            irf.plot(orth=True)
            plt.suptitle("Funções de Resposta ao Impulso (VAR)")
            plt.show()

class MotorEspacoEstados:
    """
    Implementação artesanal do Filtro de Kalman conforme o livro 'main.pdf'.
    Útil para estimar componentes não observados como o PIB Potencial.
    """
    @staticmethod
    def kalman_filter_pib(y, q_var=0.001, h_var=0.01):
        """
        Modelo Local Level:
        y_t = alpha_t + epsilon_t (Medida)
        alpha_t = alpha_{t-1} + eta_t (Transição)
        """
        n = len(y)
        a_filt = np.zeros(n)
        p_filt = np.zeros(n)
        
        # Inicialização
        a_pred = y[0]
        p_pred = 1.0
        
        for t in range(n):
            # Atualização
            v = y[t] - a_pred
            f = p_pred + h_var
            k = p_pred / f
            
            a_filt[t] = a_pred + k * v
            p_filt[t] = (1 - k) * p_pred
            
            # Predição para t+1
            a_pred = a_filt[t]
            p_pred = p_filt[t] + q_var
            
        return a_filt, p_filt

class MotorDeepLearning:
    """
    Redes Neurais Long Short-Term Memory (LSTM) para Previsão Macroeconômica.
    """
    def __init__(self, data, look_back=4):
        self.data = data.values.reshape(-1, 1)
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_scaled = self.scaler.fit_transform(self.data)
        
    def criar_dataset(self):
        X, Y = [], []
        for i in range(len(self.data_scaled) - self.look_back):
            X.append(self.data_scaled[i:(i + self.look_back), 0])
            Y.append(self.data_scaled[i + self.look_back, 0])
        return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(Y, dtype=torch.float32)

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.linear = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.linear(out[:, -1, :])

    def treinar_e_prever(self, epochs=100):
        X, Y = self.criar_dataset()
        model = self.LSTMModel(1, 64, 1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred.squeeze(), Y)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
                
        model.eval()
        with torch.no_grad():
            preds = model(X).numpy()
            preds_orig = self.scaler.inverse_transform(preds)
            
        return preds_orig

def main():
    print("================================================================")
    print("SISTEMA AVANÇADO DE ANÁLISE ECONOMÉTRICA - LUIZ TIAGO WILCKE")
    print("================================================================\n")
    
    # 1. Geração de Dados
    engine_dados = MotorDadosEconomicos(n_observacoes=400)
    df = engine_dados.gerar_dataset_multivariado()
    print("Dataset Gerado:")
    print(df.head())
    
    # 2. Análise de Séries Temporais
    analisador = AnalisadorSeriesTemporais(df)
    analisador.realizar_testes_estacionariedade()
    analisador.identificar_sarimax(target='PIB')
    analisador.modelar_var_vecm()
    
    # 3. Filtro de Kalman para PIB Potencial
    print("\n--- Estimando PIB Tendência via Filtro de Kalman ---")
    pib_tendencia, _ = MotorEspacoEstados.kalman_filter_pib(df['PIB'].values)
    df['PIB_Potencial'] = pib_tendencia
    df['Hiato_Produto'] = (df['PIB'] - df['PIB_Potencial']) / df['PIB_Potencial']
    
    # 4. Deep Learning (LSTM)
    print("\n--- Treinando LSTM para Previsão do PIB ---")
    motor_dl = MotorDeepLearning(df['PIB'])
    preds_lstm = motor_dl.treinar_e_prever(epochs=80)
    
    # 5. Visualizações
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    df[['PIB', 'PIB_Potencial']].plot(ax=ax1)
    ax1.set_title("PIB Observado vs PIB Potencial (Tendência Kalman)")
    ax1.set_ylabel("Valor")
    
    df['Hiato_Produto'].plot(ax=ax2, color='red', alpha=0.6)
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_title("Hiato do Produto (Output Gap)")
    ax2.set_ylabel("Ratio")
    
    plt.tight_layout()
    plt.show()

    print("\n--- Conclusão da Análise ---")
    print("Os modelos indicam uma dinâmica de crescimento estável com choques controlados.")
    print(f"Documento gerado por: Luiz Tiago Wilcke.")

if __name__ == "__main__":
    main()
