import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import requests
import plotly.express as px

# API URL
API_URL = "http://127.0.0.1:8000"

# Цветовая схема для метрик
metric_colors = {
    "context_recall": "#0039A6",
    "context_precision": "#FF6B35",
    "answer_correctness_literal": "#16c47c",
    "answer_correctness_neural": "#9C27B0",
}

# Создание Dash-приложения
app = dash.Dash(__name__)

# Лейаут дашборда
app.layout = html.Div([
    html.H1("Дашборд для оценки модели RAG", style={"textAlign": "center", "color": "#0039A6"}),

    # Фильтры
    html.Div([
        html.Label("Выберите кампус:"),
        dcc.Dropdown(id="campus-dropdown", options=[], value=None),
        html.Label("Выберите уровень образования:"),
        dcc.Dropdown(id="education-level-dropdown", options=[], value=None),
        html.Label("Выберите категорию вопроса:"),
        dcc.Dropdown(id="question-category-dropdown", options=[], value=None),
    ], style={"padding": "10px"}),

    dcc.Graph(id="bar-chart"),
    dcc.Graph(id="pie-chart-context-recall"),
    dcc.Graph(id="pie-chart-context-precision"),
    dcc.Graph(id="pie-chart-answer-literal"),
    dcc.Graph(id="pie-chart-answer-neural"),
])

# Колбэк для загрузки данных фильтров
@app.callback(
    [Output("campus-dropdown", "options"),
     Output("education-level-dropdown", "options"),
     Output("question-category-dropdown", "options")],
    [Input("campus-dropdown", "id")]
)
def load_filters(_):
    response = requests.get(f"{API_URL}/get_filters")
    if response.status_code == 200:
        filters = response.json()
        return [
            [{"label": c, "value": c} for c in filters["campuses"]],
            [{"label": e, "value": e} for e in filters["education_levels"]],
            [{"label": q, "value": q} for q in filters["question_categories"]]
        ]
    return [], [], []

# Колбэк для обновления графиков
@app.callback(
    [Output("bar-chart", "figure"),
     Output("pie-chart-context-recall", "figure"),
     Output("pie-chart-context-precision", "figure"),
     Output("pie-chart-answer-literal", "figure"),
     Output("pie-chart-answer-neural", "figure")],
    [Input("campus-dropdown", "value"),
     Input("education-level-dropdown", "value"),
     Input("question-category-dropdown", "value")]
)
def update_graphs(campus, education_level, category):
    response = requests.get(f"{API_URL}/get_metrics", params={
        "campus": campus, "education": education_level, "question": category
    })
    if response.status_code == 200:
        metrics = response.json()

        # Графики
        bar_fig = px.bar(x=list(metrics.keys()), y=list(metrics.values()), color=list(metrics.keys()),
                         color_discrete_map=metric_colors, title="Метрики качества")

        def create_pie_chart(metric_name, metric_value, color):
            return px.pie(names=[metric_name], values=[metric_value], title=f"{metric_name}: {metric_value:.2f}",
                          color_discrete_sequence=[color])

        return (
            bar_fig,
            create_pie_chart("Context Recall", metrics["context_recall"], metric_colors["context_recall"]),
            create_pie_chart("Context Precision", metrics["context_precision"], metric_colors["context_precision"]),
            create_pie_chart("Answer Correctness Literal", metrics["answer_correctness_literal"], metric_colors["answer_correctness_literal"]),
            create_pie_chart("Answer Correctness Neural", metrics["answer_correctness_neural"], metric_colors["answer_correctness_neural"])
        )
    return {}, {}, {}, {}, {}

if __name__ == "__main__":
    app.run_server(debug=True)
