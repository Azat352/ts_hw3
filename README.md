# ts_hw3
Репо с дз по временным рядам

## Структура проекта

```
ts_hw3/
├── data/               # Исходные данные (CSV и др.)
├── notebooks/          # Jupyter-ноутбуки для EDA
│   └── exploration.ipynb
├── src/                # Исходный код
│   ├── data.py         # Загрузка и предобработка данных
│   ├── model.py        # Определение модели
│   └── train.py        # Скрипт обучения
├── requirements.txt
└── README.md
```

## Быстрый старт

```bash
pip install -r requirements.txt
python -m src.train data/dataset.csv
```
