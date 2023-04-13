library(energy)
library(fastDummies)

df_original <- read.csv("data/rossman_store_sales/source/store.csv")
df_mostlyai <- read.csv("data/rossman_store_sales/synthetic_mostlyai/store_mostly_ai.csv")
df_gretelai <- read.csv("data/rossman_store_sales/synthetic_gretelai/store_gretel_tabular_lstm.csv")
df_realtabf <- read.csv("data/rossman_store_sales/synthetic_realtabformer/realtabformer_store.csv")[, -1]
df_sdv      <- read.csv("data/rossman_store_sales/synthetic_sdv/store.csv")
#df_ydata    <- read.csv("data/rossman_store_sales/synthetic_ydata/store_ydata.csv")

names(df_sdv) = names(df_original)
sample_df <- function(df, N) {
  return (df[sample(nrow(df), N), ])
}

calculate_energy_stat <- function(df1, df2, N=min(nrow(df1), nrow(df2)), R = 10, seed = NULL, dummy=TRUE, cols=2:length(df1), na='drop') {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  if (na == 'drop') {
    df1 <- na.omit(df1)
    df2 <- na.omit(df2)
  } else if (na == 'impute') {
    df1[is.na(df1)] <- -1
    df2[is.na(df2)] <- -1
  }
  
  df_pooled   <- rbind(sample_df(df1, N), sample_df(df2, N))
  # select columns
  x   <- df_pooled[, cols]
  if (dummy) {
    x <- dummy_cols(x, remove_selected_columns = TRUE)
  }
  
  eqdist.etest(x, sizes = c(N, N), distance=FALSE, R=R)
}
N <- 200
seed = 0
# with dummy encoding
calculate_energy_stat(df_original, df_original, seed=seed)
calculate_energy_stat(df_original, df_mostlyai, seed=seed)
calculate_energy_stat(df_original, df_gretelai, seed=seed)
calculate_energy_stat(df_original, df_realtabf, seed=seed)
calculate_energy_stat(df_original, df_sdv, seed=seed)

# without categorical
cols = 4:9
calculate_energy_stat(df_original, df_original, seed=seed, dummy=FALSE, cols=cols)
calculate_energy_stat(df_original, df_mostlyai, seed=seed, dummy=FALSE, cols=cols)
calculate_energy_stat(df_original, df_gretelai, seed=seed, dummy=FALSE, cols=cols)
calculate_energy_stat(df_original, df_realtabf, seed=seed, dummy=FALSE, cols=cols)
calculate_energy_stat(df_original, df_sdv,      seed=seed, dummy=FALSE, cols=cols)

# with imputed nans
N <- 500
calculate_energy_stat(df_original, df_original, seed=seed, na = 'impute', N=N)
calculate_energy_stat(df_original, df_mostlyai, seed=seed, na = 'impute', N=N)
calculate_energy_stat(df_original, df_gretelai, seed=seed, na = 'impute', N=N)
calculate_energy_stat(df_original, df_realtabf, seed=seed, na = 'impute', N=N)
calculate_energy_stat(df_original, df_sdv,      seed=seed, na = 'impute', N=N)
