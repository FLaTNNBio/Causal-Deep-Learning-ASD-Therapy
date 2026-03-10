# Causal Deep Learning for ASD Therapy Evaluation



[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![Deep Learning](https://img.shields.io/badge/Framework-TensorFlow/Keras-orange.svg)](https://www.tensorflow.org/)

[![Causal Inference](https://img.shields.io/badge/Method-Causal%20Inference-green.svg)](https://en.wikipedia.org/wiki/Causal\_inference)



## 🎯 Obiettivo del Progetto

L'obiettivo principale è stimare l'**effetto causale medio (ATE)** dell'intervento assistito da robot (RET) rispetto alla terapia classica (SHT) sugli outcome terapeutici di bambini con ASD. 



La sfida centrale risiede nel rispondere alla domanda controffattuale:  **"Cosa sarebbe successo agli stessi bambini se non avessero usato i robot?"**. Poiché questa quantità non è osservabile direttamente, il progetto utilizza l'inferenza causale per simulare scenari non avvenuti e misurare il reale impatto del robot sull'attenzione del paziente (*AttentionScore*).







## 📊 Dataset: DREAM

Il dataset DREAM (Development of Robot-Enhanced therapy for children with Autism spectrum disorder) contiene dati provenienti da sensori ad alta frequenza che monitorano il comportamento dei bambini durante le sessioni terapeutiche.

- **Campione:** 61 soggetti.

- **Trattamento:** Robot-Enhanced Therapy (RET) vs Standard Human Therapy (SHT).

- **Covariate considerate:** Età, genere, punteggio clinico ADOS.



## 🧠 Metodologia e Modelli

Per isolare l'effetto causale e neutralizzare i bias di selezione, sono state implementate due architetture neurali allo stato dell'arte:



1. **DragonNet:** Una rete neurale che sfrutta la *sufficienza del propensity score* per stimare contemporaneamente l'assegnazione del trattamento e l'outcome.

2. **BCAUSS:** Un modello basato sul bilanciamento tramite compiti di ricostruzione auto-supervisionata, utile per gestire campioni di dimensioni ridotte.



La pipeline include la standardizzazione delle covariate e l'attivazione della **Targeted Regularization** per raffinare la stima dell'ATE.







## 📈 Risultati Principali

Le simulazioni hanno evidenziato un ATE prossimo alla neutralità ($-0.016$ per BCAUSS e $-0.030$ per DragonNet), indicando che:

- Il robot è **efficace quanto un terapista umano** nel mantenere l'attenzione dei bambini.

- Esiste una forte **eterogeneità individuale**, suggerendo che specifici sottogruppi di pazienti beneficiano maggiormente dell'uso della tecnologia.



## 🚀 Sviluppi Futuri

- **Stima del CATE:** Identificazione dei profili clinici ottimali per la terapia robotica (Medicina di Precisione).



---

**Autore:** Malafronte Sabato  
Università degli Studi di Salerno




