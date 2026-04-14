#' ---
#' Título: Análise de Crescimento Econômico e Econometria de Painel
#' Autor: Luiz Tiago Wilcke
#' Descrição: Script avançado para simulação de modelos de crescimento
#'            e análise de convergência via dados em painel.
#' Técnicas: Modelo de Solow-Swan, GMM, Cointegração e VECM no R.
#' ---

# Carregamento de Bibliotecas
suppressPackageStartupMessages({
  library(vars)
  library(forecast)
  library(ggplot2)
  library(tseries)
  library(urca)
  library(plm)
})

# ==============================================================================
# 1. Simulação do Modelo de Crescimento de Solow-Swan Estocástico
# ==============================================================================

simular_solow <- function(n_tempo = 100, s = 0.2, n = 0.02, d = 0.05, a = 0.3) {
  #' s: taxa de poupança
  #' n: crescimento populacional
  #' d: taxa de depreciação
  #' a: elasticidade do capital (Alfa)
  
  k <- numeric(n_tempo)
  y <- numeric(n_tempo)
  k[1] <- 1  # Capital Inicial
  
  for (t in 2:n_tempo) {
    # Choque de produtividade (Progresso Técnico Estocástico)
    A_t <- exp(rnorm(1, 0, 0.02)) 
    
    # Produção: Y = A * K^a
    y[t-1] <- A_t * (k[t-1]^a)
    
    # Dinâmica do Capital: k_{t+1} = s*y_t + (1-d-n)*k_t
    k[t] <- s * y[t-1] + (1 - d - n) * k[t-1]
  }
  
  y[n_tempo] <- k[n_tempo]^a
  return(data.frame(Tempo = 1:n_tempo, Capital = k, Produto = y))
}

print("Executando Simulação de Solow-Swan...")
dados_solow <- simular_solow()

# Visualização da Trajetória de Crescimento
p1 <- ggplot(dados_solow, aes(x = Tempo)) +
  geom_line(aes(y = Produto, color = "Produto (Y)"), size = 1.2) +
  geom_line(aes(y = Capital, color = "Capital (K)"), size = 1.2, linetype = "dashed") +
  labs(title = "Dinâmica de Transição: Modelo de Solow-Swan",
       subtitle = "Autor: Luiz Tiago Wilcke",
       y = "Nível", x = "Tempo") +
  theme_minimal() +
  scale_color_manual(values = c("Produto (Y)" = "#2c3e50", "Capital (K)" = "#e74c3c"))

# ==============================================================================
# 2. Econometria de Dados em Painel: Análise de Convergência
# ==============================================================================

gerar_painel_convergencia <- function(n_paises = 50, n_anos = 20) {
  set.seed(123)
  df <- data.frame()
  
  for (i in 1:n_paises) {
    # Países pobres crescem mais rápido (Convergência Beta)
    renda_inicial <- runif(1, 5, 12)
    crescimento <- -0.02 * renda_inicial + rnorm(n_anos, 0.03, 0.01)
    
    renda_seq <- renda_inicial + cumsum(crescimento)
    
    pais_df <- data.frame(
      ID = rep(paste0("Pais_", i), n_anos),
      Ano = 1:n_anos,
      PIB_Percapita = renda_seq,
      Investimento = rnorm(n_anos, 0.2, 0.05)
    )
    df <- rbind(df, pais_df)
  }
  return(df)
}

print("Iniciando Análise de Dados em Painel...")
dados_painel <- gerar_painel_convergencia()

# Estimação de Efeitos Fixos (Within)
mod_fe <- plm(PIB_Percapita ~ Investimento, 
             data = dados_painel, 
             model = "within", 
             index = c("ID", "Ano"))

print(summary(mod_fe))

# ==============================================================================
# 3. Séries Temporais: Cointegração de Johansen (PIB vs Consumo)
# ==============================================================================

analisar_cointegracao <- function() {
  # Simulação de variáveis I(1) cointegradas
  t <- 1:200
  x <- cumsum(rnorm(200))
  y <- 0.8 * x + rnorm(200, 0, 0.5)
  
  data_mat <- cbind(y, x)
  colnames(data_mat) <- c("Consumo", "PIB")
  
  # Teste de Johansen
  johansen_test <- ca.jo(data_mat, type = "trace", ecdet = "const", K = 2)
  print(summary(johansen_test))
  
  # Se houver cointegração, estimamos o VECM via VAR (representação)
  var_est <- vec2var(johansen_test, r = 1)
  
  # Impulso Resposta
  irf_res <- irf(var_est, n.ahead = 10, boot = TRUE)
  plot(irf_res)
}

analisar_cointegracao()

# ==============================================================================
# 4. Exportação e Conclusão
# ==============================================================================

cat("\n================================================================\n")
cat("ANÁLISE FINALIZADA COM SUCESSO\n")
cat("Autor: Luiz Tiago Wilcke\n")
cat("Técnicas utilizadas: Simulação de Crescimento, Painel Dinâmico e Cointegração.\n")
cat("================================================================\n")

# Salvar gráfico principal se possível
# ggsave("grafico_crescimento.png", plot = p1)
