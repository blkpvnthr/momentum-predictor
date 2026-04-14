# Momentum Predictor

Intraday Momentum / Breakout Predictor.

rm -f outputs/signals/predictions.csv
rm -f outputs/training/labeled_predictions.csv
python scripts/train.py
python scripts/build_labeled_predictions.py
python scripts/train_v5_meta_model.py

rm live/vec_normalize.pkl
rm live/trading_model.zip
rm live/regime_engine.pkl
python live/train.py# momentum-predictor
