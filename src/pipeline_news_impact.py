"""
src/pipeline_news_impact.py
Pipeline independiente para el módulo de Impacto Histórico de Noticias.

Corre de forma SEPARADA al pipeline principal (pipeline.py).
No modifica ningún artefacto del modelo existente.

Uso:
    python src/pipeline_news_impact.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.model.train_news_impact import train_news_impact_model


def main():
    features_path = os.path.join(_ROOT, "data", "features.csv")

    if not os.path.exists(features_path):
        print("❌ data/features.csv no encontrado.")
        print("   Ejecutá el pipeline principal primero: python src/pipeline.py")
        sys.exit(1)

    print("=" * 55)
    print("  AgroCast — Módulo: Impacto Histórico de Noticias")
    print("=" * 55)

    result = train_news_impact_model(features_path)

    print("\n📊 RESUMEN FINAL:")
    print(f"  MAE modelo news-impact : {result['mae_news_model']}")
    print(f"  MAE modelo base        : {result['mae_base_model']}")
    print(f"  Mejora por noticias    : {result['improvement_pct']}%")
    print(f"  Cobertura GDELT        : {result['gdelt_coverage_pct']}%")
    print(f"  Señal actual           : {result['current_signal']}")
    print(f"  Retorno predicho       : {result['current_pred_return']}%")
    print(f"  Entrenado              : {result['trained_at']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
