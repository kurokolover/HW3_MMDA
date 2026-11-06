import json
from io import StringIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title='Вино в подборку', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model(path, version):
    return joblib.load(path)


@st.cache_data
def load_test(path, version):
    return pd.read_csv(path)


@st.cache_data
def test_proba(_model, X, model_version):
    # Порог не входит в ключ кэша: на тесте вероятности считаются один раз.
    return _model.predict_proba(X)[:, 1]


def predict_batch(model, frame, features, threshold):
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('В файле нет строк с винами.')
    clean = frame.loc[:, features].apply(pd.to_numeric, errors='coerce')
    invalid = clean.columns[~np.isfinite(clean.to_numpy()).all(axis=0)].tolist()
    if invalid:
        raise ValueError('Проверьте числа и пропуски в столбцах: ' + ', '.join(invalid))
    if not clean['is_red'].isin([0, 1]).all():
        raise ValueError('В is_red допустимы только 0 для белого и 1 для красного вина.')
    if (clean < 0).any().any():
        raise ValueError('Характеристики вина не должны быть отрицательными.')
    result = clean.copy()
    result['probability_good'] = model.predict_proba(clean)[:, 1]
    result['prediction'] = (result['probability_good'] >= threshold).astype(int)
    result['decision'] = np.where(result['prediction'] == 1, 'В подборку', 'Пропустить')
    return result


try:
    config = json.loads((BASE_DIR / 'config.json').read_text(encoding='utf-8'))
    model_path, test_path = BASE_DIR / 'model.joblib', BASE_DIR / 'test.csv'
    model_version = model_path.stat().st_mtime_ns
    model = load_model(str(model_path), model_version)
    data = load_test(str(test_path), test_path.stat().st_mtime_ns)
except (OSError, ValueError) as exc:
    st.error(f'Не удалось прочитать файлы приложения: {exc}')
    st.stop()

features = list(model.feature_names_in_)
X_test = data.loc[:, features]
y_test = data['target'].to_numpy()
scores = test_proba(model, X_test, model_version)
labels = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность', 'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар', 'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы', 'density': 'Плотность',
    'pH': 'pH', 'sulphates': 'Сульфаты', 'alcohol': 'Алкоголь', 'is_red': 'Красное вино',
}

st.title('🍷 Вино в подборку')
st.write('Оценка качества вина по лабораторным характеристикам.')
st.caption('Хорошим считается вино с экспертной оценкой от 7 баллов.')
with st.sidebar:
    st.header('Правило отбора')
    threshold = st.slider('Порог хорошего вина', 0.0, 1.0,
                          value=float(config['threshold']), step=0.001,
                          format='%.3f', key='threshold')
    st.caption(f'Порог из валидации: {config["threshold"]:.4f}. Он применяется во всех вкладках.')
    st.write('При повышении порога в подборку попадёт меньше вин. При снижении '
             'подборка расширится, и вместе с хорошими винами могут попасть обычные.')

one, report, batch, explain = st.tabs([
    'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with one:
    st.subheader('Характеристики вина')
    values = {}
    columns = st.columns(3)
    for i, name in enumerate(f for f in features if f != 'is_red'):
        low, high = float(X_test[name].min()), float(X_test[name].max())
        with columns[i % 3]:
            values[name] = st.number_input(labels[name], min_value=low, max_value=high,
                                           value=float(X_test[name].median()), format='%.4f',
                                           key=f'feature_{name}', help=f'{name}: от {low:g} до {high:g}')
    values['is_red'] = int(st.toggle('Красное вино', value=bool(X_test['is_red'].median())))
    st.caption('Выключенный переключатель означает белое вино. Диапазоны и начальные значения взяты из test.csv.')
    sample = pd.DataFrame([values], columns=features)
    probability = float(model.predict_proba(sample)[0, 1])
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    if probability >= threshold:
        st.success('В подборку: вероятность не ниже порога.')
    else:
        st.info('Пропустить: вероятность ниже порога.')

with report:
    st.subheader('Модельные метрики')
    st.caption('Оценка на тестовой выборке. Рабочий порог был выбран отдельно на валидации.')
    pred = (scores >= threshold).astype(int)
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)
    columns = st.columns(4)
    for col, name, value in zip(columns[:3], ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        col.metric(name, f'{value:.3f}')
    columns[3].metric('Вин в подборке', f'{pred.sum()} из {len(pred)}')
    st.write('Precision показывает долю хороших вин в подборке, recall показывает, '
             'какую долю всех хороших вин удалось найти, а F1 объединяет эти оценки.')
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['Остальные', 'Хорошие']).plot(
        ax=ax, cmap='Purples', colorbar=False)
    ax.set_xlabel('Решение модели'); ax.set_ylabel('Настоящий класс')
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption(f'Обычных вин в подборке: {cm[0, 1]}. Пропущено хороших вин: {cm[1, 0]}.')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metrics.RocCurveDisplay.from_predictions(y_test, scores, ax=axes[0], name='Модель')
    metrics.PrecisionRecallDisplay.from_predictions(y_test, scores, ax=axes[1], name='Модель')
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].axhline(y_test.mean(), ls='--', color='gray', label=f'Случайная модель: {y_test.mean():.3f}')
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    axes[0].scatter(fpr, recall, s=90, color='darkorange', zorder=5, label=f'Порог {threshold:.3f}')
    pr_point = precision if pred.sum() else 1.0
    axes[1].scatter(recall, pr_point, s=90, color='darkorange', zorder=5, label=f'Порог {threshold:.3f}')
    if pred.sum() == 0:
        st.info('Подборка пустая. В метриках precision принят равным 0, а на PR-кривой '
                'показана условная конечная точка (recall=0, precision=1).')
    axes[0].legend(); axes[1].legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption(f'ROC-AUC: {metrics.roc_auc_score(y_test, scores):.4f}; '
               f'PR-AUC (average precision): {metrics.average_precision_score(y_test, scores):.4f}. '
               'Эти оценки не зависят от порога.')

with batch:
    st.subheader('Проверить несколько вин')
    st.write('Загрузите CSV с 12 признаками без столбца target. Пример можно скачать ниже.')
    st.download_button('Скачать пример CSV', X_test.head(5).to_csv(index=False).encode('utf-8'),
                       file_name='example_wines.csv', mime='text/csv')
    uploaded = st.file_uploader('CSV с характеристиками вин', type=['csv'])
    if uploaded is not None:
        try:
            frame = pd.read_csv(StringIO(uploaded.getvalue().decode('utf-8-sig')), sep=None, engine='python')
            result = predict_batch(model, frame, features, threshold)
            extra = [name for name in frame.columns if name not in features]
            if extra:
                st.caption('Лишние столбцы не используются: ' + ', '.join(extra))
            st.dataframe(result, use_container_width=True)
            st.download_button('Скачать предсказания', result.to_csv(index=False).encode('utf-8-sig'),
                               file_name='wine_predictions.csv', mime='text/csv')
        except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
            st.error(f'Не удалось обработать CSV. {exc}')

with explain:
    st.subheader('Какие признаки влияют на оценку')
    coefficients = pd.Series(model.steps[-1][1].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([labels[name] for name in coefficients.index], coefficients.to_numpy(),
            color=np.where(coefficients.to_numpy() >= 0, '#2e7d68', '#a34d68'))
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Коэффициент после StandardScaler')
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    positive = coefficients[coefficients > 0].nlargest(3)
    negative = coefficients[coefficients < 0].nsmallest(3)
    st.write('Сильнее всего за хорошее вино: ' + ', '.join(labels[n].lower() for n in positive.index) + '.')
    st.write('Сильнее всего против: ' + ', '.join(labels[n].lower() for n in negative.index) + '.')
    st.caption('Коэффициент показывает изменение логарифма шансов при увеличении признака на одно '
               'стандартное отклонение и неизменных остальных признаках. Это связь внутри модели, '
               'она сама по себе не доказывает причинного влияния.')
