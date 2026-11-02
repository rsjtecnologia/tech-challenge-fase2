"""Exporta contrato web do PCOS. Usado pelo notebook 05 e para validação."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def encontrar_raiz() -> Path:
    atual = Path.cwd().resolve()
    for candidato in [atual, *atual.parents]:
        if (candidato / "src" / "db").is_dir():
            return candidato
    raise FileNotFoundError("Raiz do repositório não encontrada.")


def exportar(raiz: Path | None = None) -> Path:
    raiz = raiz or encontrar_raiz()
    output_base = raiz / "src/db/outputs/polycystic-ovary-syndrome-pcos"
    pasta_entrada = output_base / "04-avaliacao-metricas"
    pasta_preprocessamento = output_base / "02-preprocessamento"
    pasta_base = output_base / "05-export-web"
    pasta_artefatos = pasta_base / "artifacts"
    pasta_metricas = pasta_base / "metrics"

    caminho_modelos = pasta_entrada / "artifacts" / "modelos_treinados.joblib"
    caminho_metricas = pasta_entrada / "metrics" / "metricas_teste.json"
    caminho_preprocessamento = pasta_preprocessamento / "artifacts" / "dados_preprocessados.joblib"
    caminho_csv = raiz / "src/db/models/polycystic-ovary-syndrome-pcos/PCOS_infertility.csv"

    for pasta in (pasta_artefatos, pasta_metricas):
        pasta.mkdir(parents=True, exist_ok=True)

    for caminho, msg in [
        (caminho_modelos, "Execute o notebook 04 primeiro."),
        (caminho_metricas, "Execute o notebook 04 primeiro."),
        (caminho_preprocessamento, "Execute o notebook 02 primeiro."),
        (caminho_csv, f"Dataset não encontrado: {caminho_csv}"),
    ]:
        if not caminho.exists():
            raise FileNotFoundError(msg)

    pacote = joblib.load(caminho_modelos)
    pacote_preprocessamento = joblib.load(caminho_preprocessamento)
    dados_brutos = pd.read_csv(caminho_csv)
    dados_brutos.columns = dados_brutos.columns.str.strip()

    modelos_treinados = pacote["modelos_treinados"]
    atributos_treino = pacote_preprocessamento["atributos_treino"]
    rotulo_treino = pacote_preprocessamento["rotulo_treino"]
    atributos_teste = pacote["atributos_teste"]
    rotulo_teste = pacote["rotulo_teste"]
    colunas_atributos = pacote["colunas_atributos"]
    semente = int(pacote_preprocessamento["semente"])

    with caminho_metricas.open(encoding="utf-8") as arquivo:
        metricas_origem = json.load(arquivo)

    nome_melhor = metricas_origem["_melhor_modelo"]
    metricas_modelos = {
        nome: valores
        for nome, valores in metricas_origem.items()
        if not nome.startswith("_")
    }

    rotulos = np.asarray(rotulo_teste, dtype=int)
    dados_grid = atributos_teste.reset_index(drop=True).copy()
    dados_grid.insert(0, "id", np.arange(1, len(dados_grid) + 1, dtype=int))
    dados_grid["rotulo_real"] = rotulos
    dados_grid["diagnostico_real"] = np.where(rotulos == 1, "Com PCOS", "Sem PCOS")

    linhas_longas = []
    matrizes: dict = {}
    nomes_modelos = list(modelos_treinados)

    for nome, pipeline in modelos_treinados.items():
        previstos = np.asarray(pipeline.predict(atributos_teste), dtype=int)
        probabilidades = np.asarray(pipeline.predict_proba(atributos_teste)[:, 1], dtype=float)
        acertos = previstos == rotulos

        dados_grid[f"pred_{nome}"] = previstos
        dados_grid[f"prob_{nome}"] = np.round(probabilidades, 6)
        dados_grid[f"acerto_{nome}"] = acertos

        matriz = confusion_matrix(rotulos, previstos, labels=[0, 1])
        tn, fp, fn, tp = (int(valor) for valor in matriz.ravel())
        matrizes[nome] = {"matriz": matriz.astype(int).tolist(), "tn": tn, "fp": fp, "fn": fn, "tp": tp}

        for indice, (previsto, probabilidade, acerto) in enumerate(
            zip(previstos, probabilidades, acertos)
        ):
            linhas_longas.append(
                {
                    "id": int(dados_grid.at[indice, "id"]),
                    "modelo": nome,
                    "rotulo_real": int(rotulos[indice]),
                    "diagnostico_real": "Com PCOS" if rotulos[indice] == 1 else "Sem PCOS",
                    "rotulo_previsto": int(previsto),
                    "diagnostico_previsto": "Com PCOS" if previsto == 1 else "Sem PCOS",
                    "prob_pcos": round(float(probabilidade), 6),
                    "acerto": bool(acerto),
                }
            )

    colunas_acerto = [f"acerto_{nome}" for nome in nomes_modelos]
    colunas_probabilidade = [f"prob_{nome}" for nome in nomes_modelos]
    dados_grid["n_modelos_acertaram"] = dados_grid[colunas_acerto].sum(axis=1).astype(int)
    dados_grid["consenso"] = dados_grid["n_modelos_acertaram"].isin([0, len(nomes_modelos)])
    dados_grid["amplitude_probabilidade"] = (
        dados_grid[colunas_probabilidade].max(axis=1)
        - dados_grid[colunas_probabilidade].min(axis=1)
    ).round(6)

    dados_longos = pd.DataFrame(linhas_longas)

    atributos_completos = pd.concat([atributos_treino, atributos_teste], ignore_index=True)
    rotulos_completos = pd.concat(
        [rotulo_treino.reset_index(drop=True), rotulo_teste.reset_index(drop=True)],
        ignore_index=True,
    ).astype(int)
    conjuntos = {
        "completo": (atributos_completos, rotulos_completos),
        "treino": (atributos_treino.reset_index(drop=True), rotulo_treino.reset_index(drop=True).astype(int)),
        "teste": (atributos_teste.reset_index(drop=True), rotulo_teste.reset_index(drop=True).astype(int)),
    }

    def resumo_classes(rotulos_conjunto: pd.Series) -> dict:
        total = int(len(rotulos_conjunto))
        contagens = rotulos_conjunto.value_counts().to_dict()
        return {
            "total": total,
            "sem_pcos": {"n": int(contagens.get(0, 0)), "pct": float(contagens.get(0, 0) / total)},
            "com_pcos": {"n": int(contagens.get(1, 0)), "pct": float(contagens.get(1, 0) / total)},
        }

    splits = {nome: resumo_classes(rotulos_split) for nome, (_, rotulos_split) in conjuntos.items()}

    histogramas: dict = {}
    for nome_split, (atributos_split, rotulos_split) in conjuntos.items():
        histogramas[nome_split] = {}
        for atributo in colunas_atributos:
            valores = atributos_split[atributo].astype(float).to_numpy()
            limites = np.histogram_bin_edges(valores[~np.isnan(valores)], bins=12)
            centros = ((limites[:-1] + limites[1:]) / 2).round(6)
            histogramas[nome_split][atributo] = {
                "centros": centros.tolist(),
                "sem_pcos": np.histogram(
                    valores[rotulos_split.to_numpy() == 0], bins=limites
                )[0].astype(int).tolist(),
                "com_pcos": np.histogram(
                    valores[rotulos_split.to_numpy() == 1], bins=limites
                )[0].astype(int).tolist(),
            }

    base_correlacao = atributos_completos.copy()
    base_correlacao["pcos"] = rotulos_completos.to_numpy()
    matriz_correlacao = base_correlacao.corr().round(4)

    pipeline_arvore = modelos_treinados["Árvore de Decisão"]
    classificador_arvore = pipeline_arvore.named_steps["classificador"]
    importancias = sorted(
        [
            {"atributo": atributo, "valor": round(float(valor), 8)}
            for atributo, valor in zip(colunas_atributos, classificador_arvore.feature_importances_)
        ],
        key=lambda item: item["valor"],
        reverse=True,
    )

    metricas_chave = ["acuracia", "recall_pcos", "f1_pcos", "precisao_pcos"]
    ranking = {}
    for metrica in metricas_chave:
        ordenados = sorted(
            metricas_modelos.items(),
            key=lambda item: float(item[1][metrica]),
            reverse=True,
        )
        ranking[metrica] = [
            {"modelo": nome, "valor": float(valores[metrica]), "posicao": posicao}
            for posicao, (nome, valores) in enumerate(ordenados, start=1)
        ]

    classificadores = {
        nome: pipeline.named_steps["classificador"]
        for nome, pipeline in modelos_treinados.items()
    }
    parametros = {nome: classificador.get_params() for nome, classificador in classificadores.items()}

    modelos_configuracao = {
        "Regressão Logística": {
            "algoritmo": type(classificadores["Regressão Logística"]).__name__,
            "justificativa": "Baseline interpretável com probabilidades e balanceamento de classes.",
            "hiperparametros": {
                "max_iter": int(parametros["Regressão Logística"]["max_iter"]),
                "class_weight": parametros["Regressão Logística"]["class_weight"],
                "random_state": int(parametros["Regressão Logística"]["random_state"]),
            },
        },
        "Árvore de Decisão": {
            "algoritmo": type(classificadores["Árvore de Decisão"]).__name__,
            "justificativa": "Regras não lineares e importância global dos biomarcadores.",
            "hiperparametros": {
                "class_weight": parametros["Árvore de Decisão"]["class_weight"],
                "random_state": int(parametros["Árvore de Decisão"]["random_state"]),
            },
        },
        "SVM": {
            "algoritmo": "SVC calibrado",
            "justificativa": "Fronteira não linear com probabilidades calibradas.",
            "hiperparametros": {
                "kernel": parametros["SVM"]["estimator__kernel"],
                "class_weight": parametros["SVM"]["estimator__class_weight"],
                "calibracao": parametros["SVM"]["method"],
                "cv": int(parametros["SVM"]["cv"]),
            },
        },
        "Gradient Boosting": {
            "algoritmo": type(classificadores["Gradient Boosting"]).__name__,
            "justificativa": "Ensemble sequencial para relações entre biomarcadores.",
            "hiperparametros": {
                "random_state": int(parametros["Gradient Boosting"]["random_state"]),
            },
        },
        "Random Forest": {
            "algoritmo": type(classificadores["Random Forest"]).__name__,
            "justificativa": "Múltiplas árvores com balanceamento de classes.",
            "hiperparametros": {
                "n_estimators": int(parametros["Random Forest"]["n_estimators"]),
                "class_weight": parametros["Random Forest"]["class_weight"],
                "random_state": int(parametros["Random Forest"]["random_state"]),
            },
        },
    }

    preprocessamento = {
        "imputacao": "mediana",
        "padronizacao": "StandardScaler",
        "censura_beta_hcg": "valores 1.99 tratados como ausentes",
        "ajuste": "somente no conjunto de treino",
        "atributos": len(colunas_atributos),
    }

    por_modelo = {}
    for nome, valores in metricas_modelos.items():
        probs = dados_grid[f"prob_{nome}"].astype(float).to_numpy()
        preds = dados_grid[f"pred_{nome}"].astype(int).to_numpy()
        confiancas = np.where(preds == 1, probs, 1 - probs)
        acertos = int(dados_grid[f"acerto_{nome}"].sum())
        total_avaliacao = len(dados_grid)
        melhores = {
            metrica
            for metrica in metricas_chave
            if np.isclose(
                float(valores[metrica]),
                max(float(item[metrica]) for item in metricas_modelos.values()),
            )
        }
        piores = {
            metrica
            for metrica in metricas_chave
            if np.isclose(
                float(valores[metrica]),
                min(float(item[metrica]) for item in metricas_modelos.values()),
            )
        }
        por_modelo[nome] = {
            **{metrica: float(valores[metrica]) for metrica in metricas_chave},
            **matrizes[nome],
            "media_prob_pcos": round(float(probs.mean()), 6),
            "media_prob_sem_pcos": round(float((1 - probs).mean()), 6),
            "media_confianca": round(float(confiancas.mean()), 6),
            "acertos": acertos,
            "total_avaliacao": total_avaliacao,
            "melhor_em": sorted(melhores),
            "pior_em": sorted(piores),
            "configuracao": {
                **modelos_configuracao[nome],
                "amostras_treino": len(atributos_treino),
                "amostras_avaliacao": len(atributos_teste),
            },
        }

    payload_metricas = {
        "versao_contrato": 1,
        "dataset_id": "pcos",
        "dataset_label": "SOP (PCOS)",
        "criterio_selecao": "recall_pcos",
        "classe_positiva": "Com PCOS",
        "classe_negativa": "Sem PCOS",
        "melhor_modelo": nome_melhor,
        "quantidade_modelos": len(nomes_modelos),
        "quantidade_amostras": len(dados_grid),
        "quantidade_atributos": len(colunas_atributos),
        "atributos": list(colunas_atributos),
        "preprocessamento": preprocessamento,
        "ranking": ranking,
        "por_modelo": por_modelo,
    }

    payload_graficos = {
        "versao_contrato": 1,
        "dataset_id": "pcos",
        "inventario": {
            "registros_brutos": len(dados_brutos),
            "registros_usados": len(atributos_completos),
            "registros_descartados": 0,
            "colunas_removidas_modelagem": ["Sl. No", "Patient File No."],
            "quantidade_atributos": len(colunas_atributos),
            "semente": semente,
            "test_size": 0.2,
            "estratificado": True,
        },
        "splits": splits,
        "histogramas": histogramas,
        "correlacao": {
            "atributos": matriz_correlacao.columns.tolist(),
            "matriz": matriz_correlacao.to_numpy().tolist(),
            "conjunto": "completo",
        },
        "importancia_atributos": {
            "modelo": "Árvore de Decisão",
            "metodo": "feature_importances_",
            "itens": importancias,
        },
        "destaques_eda": {
            "amh_media_sem_pcos": round(
                float(atributos_completos.loc[rotulos_completos == 0, "amh"].mean()), 3
            ),
            "amh_media_com_pcos": round(
                float(atributos_completos.loc[rotulos_completos == 1, "amh"].mean()), 3
            ),
            "censura_beta_i": int((dados_brutos.iloc[:, 3] == 1.99).sum()),
            "censura_beta_ii": int((pd.to_numeric(dados_brutos.iloc[:, 4], errors="coerce") == 1.99).sum()),
        },
    }

    caminho_grid = pasta_artefatos / "amostras_comparativo.parquet"
    caminho_longo = pasta_artefatos / "predicoes_longas.parquet"
    caminho_metricas_web = pasta_metricas / "metricas_web.json"
    caminho_graficos_web = pasta_metricas / "graficos_web.json"

    dados_grid.to_parquet(caminho_grid, index=False, engine="pyarrow", compression="snappy")
    dados_longos.to_parquet(caminho_longo, index=False, engine="pyarrow", compression="snappy")

    with caminho_metricas_web.open("w", encoding="utf-8") as arquivo:
        json.dump(payload_metricas, arquivo, indent=2, ensure_ascii=False)
    with caminho_graficos_web.open("w", encoding="utf-8") as arquivo:
        json.dump(payload_graficos, arquivo, indent=2, ensure_ascii=False)

    print(f"Export concluído: {pasta_base}")
    return pasta_base


if __name__ == "__main__":
    exportar()
