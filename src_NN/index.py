import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Configurazione del framework Keras Legacy per compatibilità con CausalForge
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import keras.backend as K
from causalforge.model import Model, PROBLEM_TYPE
from causalforge.models.dragonnet import DragonNet


def get_data_pipeline(file_name, features):
    """
    Carica il dataset, esegue il cleaning e la standardizzazione.
    """
    path = os.path.join(os.path.dirname(__file__), "data", file_name)
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    # Rimozione valori mancanti nelle feature chiave e nell'outcome
    df = df.dropna(subset=features + ['Condition', 'AttentionScore'])

    X = df[features].values
    T = df['Condition'].map({'RET': 1, 'SHT': 0}).values
    y = df['AttentionScore'].values

    # Standardizzazione delle covariate (Mean=0, Std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, T, y


def run_experiments(X, T, y, n_runs=10):
    """
    Motore di esecuzione multi-simulazione per la stima della varianza dell'ATE.
    """
    results = {"Model": [], "ATE": [], "Run": []}
    input_dim = X.shape[1]

    for i in range(n_runs):
        K.clear_session()  # Reset dello stato dei pesi neurali

        # --- Configurazione DragonNet ---
        try:
            dn_model = DragonNet()
            dn_params = {
                'input_dim': input_dim,
                'epochs': 100,
                'use_targ_term': True,  # Attivazione Targeted Regularization
                'verbose': False
            }
            dn_model.build(dn_params)
            dn_model.fit(X, T, y)
            results["Model"].append("DragonNet")
            results["ATE"].append(dn_model.predict_ate(X, T, y))
            results["Run"].append(i + 1)
        except Exception as e:
            print(f"Errore DragonNet (Run {i + 1}): {e}")

        # --- Configurazione BCAUSS ---
        try:
            bc_params = {
                'input_dim': input_dim,
                'epochs': 100,
                'use_targ_term': True,
                'ratio': 1.0,
                'scale_preds': False,
                'verbose': False
            }
            bcauss_model = Model.create_model(
                "bcauss", bc_params,
                problem_type=PROBLEM_TYPE.CAUSAL_TREATMENT_EFFECT_ESTIMATION
            )
            bcauss_model.fit(X, T, y)
            results["Model"].append("BCAUSS")
            results["ATE"].append(bcauss_model.predict_ate(X, T, y))
            results["Run"].append(i + 1)
        except Exception as e:
            print(f"Errore BCAUSS (Run {i + 1}): {e}")

    return pd.DataFrame(results)


def plot_results(df_results):
    """
    Genera il boxplot comparativo per la stabilità dell'ATE.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Model', y='ATE', data=df_results, palette='Set2')
    sns.stripplot(x='Model', y='ATE', data=df_results, color=".25", alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', label='Baseline (No Effect)')
    plt.title('Distribuzione ATE: DragonNet vs BCAUSS')
    plt.ylabel('Average Treatment Effect (ATE)')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Definizione delle covariate di interesse
    FEATS = ['Age_in_month', 'ADOS_Total', 'is_male']

    # Esecuzione pipeline
    X_scaled, T, y = get_data_pipeline("df_final.xlsx", FEATS)
    df_results = run_experiments(X_scaled, T, y, n_runs=10)

    # Output statistico descrittivo
    summary = df_results.groupby('Model')['ATE'].agg(['mean', 'std']).reset_index()
    print(summary)

    # Confronto stabilità ATE: DragonNet vs BCAUSS
    plot_results(df_results)