import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from markdown_bio import *
from .Dashboard import Dashboard
from .BioquivalenceMathsModel import BioquivalenceMathsModel


class BioequivalenceDashboard(Dashboard):

    def __init__(self, settings):
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        external_scripts = ['https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML']

        # create Dash(Flask) server
        self.app = dash.Dash(
            server=True, external_stylesheets=external_stylesheets, external_scripts=external_scripts)

        # increase port
        Dashboard.port += 1
        mathdata = BioquivalenceMathsModel(settings)
        mathdata.run_bio_model()
        self.mathdata = mathdata

    def _generate_layout(self):
        # metrics inludings is checked inside method
        graph_list = [self._generate_concentration_time(True),
        self._generate_concentration_time(False),
        self._generate_concentration_time_log(True),
        self._generate_concentration_time_log(False),
        self._generate_concentration_time_mean(True),
        self._generate_concentration_time_mean(False),
        self._generate_concentration_time_log_mean(True),
        self._generate_concentration_time_log_mean(False),
        self._generate_criteria(),
        self._generate_param(),
        self._generate_anova(),
        self._generate_interval()]
        return html.Div(graph_list)

    def _generate_criteria(self):
        data = {'Критерий':['Колмогорова-Смирнова', 'Колмогорова-Смирнова', 'Шапиро-Уилка',
        'Шапиро-Уилка', 'F-критерий', 'Левена'],
            'Группа':['R', 'T', 'R', 'T', 'R, T', 'R, T'],
            'Значение критерия':[self.mathdata.kstest_r[0], self.mathdata.kstest_t[0], self.mathdata.shapiro_r[0],
            self.mathdata.shapiro_t[0], self.mathdata.f[0], self.mathdata.levene[0]],
            'p-уровень':[self.mathdata.kstest_r[1], self.mathdata.kstest_t[1], self.mathdata.shapiro_r[1],
            self.mathdata.shapiro_t[1], self.mathdata.f[1], self.mathdata.levene[1]]}
        df = pd.DataFrame(data)
        df = round_df(df)
        return html.Div([html.Div(html.H1(children='Выполнение критериев'), style={'text-align':'center'}),
            html.Div([
            html.Div([
                html.Div([dash_table.DataTable(
                    id='criteria',
                    columns=[{"name": i, "id": i, "deletable":True} for i in df.columns],
                    data=df.to_dict('records'),
                    style_cell_conditional=[
                    {'if': {'column_id': 'Критерий'},
                     'width': '25%'},
                    {'if': {'column_id': 'Группа'},
                     'width': '25%'},
                    {'if': {'column_id': 'Значение критерия'},
                     'width': '25%'},
                    {'if': {'column_id': 'p-уровень'},
                     'width': '25%'},
                ]
                )],style={'border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px','margin':'5px'})],
                style={'width': '78%', 'display': 'inline-block'}),
                html.Div(dcc.Markdown(children=markdown_text_criteria), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})
                ])
            ], style={'margin':'50px'}
        )

    def _generate_param(self):
        data = {'Группа':['R', 'T'],
        'AUC':[float(np.mean(self.mathdata.auc_r_notlog)), float(np.mean(self.mathdata.auc_t_notlog))],
            'AUC_inf':[float(np.mean(self.mathdata.auc_r_infty)), float(np.mean(self.mathdata.auc_t_infty))],
            'ln AUC':[float(np.mean(self.mathdata.auc_r)), float(np.mean(self.mathdata.auc_t))],
            'ln AUC_inf':[float(np.mean(self.mathdata.auc_r_infty_log)), float(np.mean(self.mathdata.auc_t_infty_log))],
            'ln Tmax':[float(np.log(self.mathdata.concentration_r.columns.max())), float(np.log(self.mathdata.concentration_t.columns.max()))],
            'ln Cmax':[float(np.log(self.mathdata.concentration_r.max().max())), float(np.log(self.mathdata.concentration_t.max().max()))]}
        df = pd.DataFrame(data)
        df = round_df(df)
        return html.Div([html.Div(html.H1(children='Распределение ключевых параметров по группам'), style={'text-align':'center'}),
            html.Div([
            html.Div([
                html.Div([dash_table.DataTable(
                    id='param',
                    columns=[{"name": i, "id": i, "deletable":True} for i in df.columns],
                    data=df.to_dict('records')
                )],style={'border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px','margin':'5px'})],
                style={'width': '78%', 'display': 'inline-block'}),
                html.Div(dcc.Markdown(children=markdown_text_param), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})
                ])
            ], style={'margin':'50px'}
        )

    def _generate_anova(self):
        df = self.mathdata.anova[0]
        df = round_df(df)
        return html.Div([html.Div(html.H1(children='ANOVA'), style={'text-align':'center'}),
            html.Div([
            html.Div([
                html.Div([dash_table.DataTable(
                    id='anova',
                    columns=[{"name": i, "id": i, "deletable":True} for i in df.columns],
                    data=df.to_dict('records'),
                    style_cell_conditional=[
                    {'if': {'column_id': 'SS'},
                     'width': '20%'},
                    {'if': {'column_id': 'df'},
                     'width': '20%'},
                    {'if': {'column_id': 'MS'},
                     'width': '20%'},
                    {'if': {'column_id': 'F'},
                     'width': '20%'},
                    {'if': {'column_id': 'F крит.'},
                     'width': '20%'}
                ]
                )],style={'border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px','margin':'5px'})],
                style={'width': '78%', 'display': 'inline-block'}),
                html.Div(dcc.Markdown(children=markdown_text_anova), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})
                ])
            ], style={'margin':'50px'}
        )

    def _generate_interval(self):
        data = {'Нижняя граница':[100*(e**self.mathdata.oneside[0])],
            'Верхняя граница':[100*(e**self.mathdata.oneside[1])],
            'Доверительный интервал критерия':['80.00-125.00%'],
            'Критерий биоэквивалентности':['Выполнен' if (self.mathdata.oneside[0]>-0.223 and
            self.mathdata.oneside[1]<0.223) else  'Не выполнен'],
            'Критерий бионеэквивалентности':['Выполнен' if (self.mathdata.oneside[0]>0.223 or
            self.mathdata.oneside[1]<-0.223) else  'Не выполнен']}
        df = pd.DataFrame(data)
        df = round_df(df)
        return html.Div([html.Div(html.H1(children='Результаты оценки биоэквивалентности'), style={'text-align':'center'}),
            html.Div([
            html.Div([
                html.Div([dash_table.DataTable(
                    id='interval',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict('records')
                )],style={'border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px','margin':'5px'})],
                style={'width': '78%', 'display': 'inline-block'}),
                html.Div(dcc.Markdown(children=markdown_text_interval), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})
                ])
            ], style={'margin':'50px'}
        )

    def _generate_concentration_time(self, ref=True):
        if ref:
            df = self.mathdata.concentration_r
            time = df.columns

            def update_graph(yaxis_column_name_conc_r):
                fig = px.scatter(x=time, y=df.loc[yaxis_column_name_conc_r])
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_r,
                                 type='linear')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_r', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_r', 'value')])(update_graph)

            available_indicators = df.index

            return html.Div([html.Div(html.H1(children='Концентрация от времени референтный препарат'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([
                        dcc.Markdown(children="Выберите показатель для оси ОY:"),
                        dcc.Dropdown(
                            id='yaxis_column_name_conc_r',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value=available_indicators[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={'padding': '5px'}),
                dcc.Graph(id='concentration_time_r')], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_r), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )
        else:
            df = self.mathdata.concentration_t
            time = df.columns

            def update_graph(yaxis_column_name_conc_t):
                fig = px.scatter(x=time, y=df.loc[yaxis_column_name_conc_t])
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_t,
                                 type='linear')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_t', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_t', 'value')])(update_graph)


            available_indicators = df.index

            return html.Div([html.Div(html.H1(children='Концентрация от времени тестовый препарат'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([
                        dcc.Markdown(children="Выберите показатель для оси ОY:"),
                        dcc.Dropdown(
                            id='yaxis_column_name_conc_t',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value=available_indicators[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={'padding': '5px'}),
                dcc.Graph(id='concentration_time_t')], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_t), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )

    def _generate_concentration_time_log(self, ref=True):
        if ref:
            df = self.mathdata.concentration_r
            time = df.columns

            def update_graph(yaxis_column_name_conc_r_log):
                fig = px.scatter(x=time, y=df.loc[yaxis_column_name_conc_r_log])
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_r_log,
                                 type='log')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_r_log', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_r_log', 'value')])(update_graph)


            available_indicators = df.index

            return html.Div([html.Div(html.H1(children='Прологарифмированная концентрация от времени референтный препарат'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([
                        dcc.Markdown(children="Выберите показатель для оси ОY:"),
                        dcc.Dropdown(
                            id='yaxis_column_name_conc_r_log',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value=available_indicators[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={'padding': '5px'}),
                dcc.Graph(id='concentration_time_r_log')], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_r_log), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )
        else:
            df = self.mathdata.concentration_t
            time = df.columns

            def update_graph(yaxis_column_name_conc_t_log):
                fig = px.scatter(x=time, y=df.loc[yaxis_column_name_conc_t_log])
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_t_log,
                                 type='log')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_t_log', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_t_log', 'value')])(update_graph)


            available_indicators = df.index

            return html.Div([html.Div(html.H1(children='Прологарифмированная концентрация от времени тестовый препарат'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([
                        dcc.Markdown(children="Выберите показатель для оси ОY:"),
                        dcc.Dropdown(
                            id='yaxis_column_name_conc_t_log',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value=available_indicators[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={'padding': '5px'}),
                dcc.Graph(id='concentration_time_t_log')], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_t_log), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )

    def _generate_concentration_time_mean(self, ref=True):
        if ref:
            df = self.mathdata.concentration_r.mean()
            time = df.index

            fig = px.scatter(x = time, y = df)
            fig.update_xaxes(title='Время')
            fig.update_yaxes(type='linear')


            return html.Div([html.Div(html.H1(children='Обобщенная концентрация от времени референтный препарат'), style={'text-align':'center'}),
                html.Div([
                dcc.Graph(id='concentration_time_r_mean', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_r_mean), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )
        else:
            df = self.mathdata.concentration_t.mean()
            time = df.index

            fig = px.scatter(x = time, y = df)
            fig.update_xaxes(title='Время')
            fig.update_yaxes(type='linear')


            return html.Div([html.Div(html.H1(children='Обобщенная концентрация от времени тестовый препарат'), style={'text-align':'center'}),
                html.Div([
                dcc.Graph(id='concentration_time_t_mean', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_t_mean), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )

    def _generate_concentration_time_log_mean(self, ref=True):
        if ref:
            df = self.mathdata.concentration_r.mean()
            time = df.index

            fig = px.scatter(x = time, y = df)
            fig.update_xaxes(title='Время')
            fig.update_yaxes(type='log')


            return html.Div([html.Div(html.H1(children='Прологарифмированная обобщенная концентрация от времени референтный препарат'), style={'text-align':'center'}),
                html.Div([
                dcc.Graph(id='concentration_time_r_log_mean', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_r_log_mean), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )
        else:
            df = self.mathdata.concentration_t.mean()
            time = df.index

            fig = px.scatter(x = time, y = df)
            fig.update_xaxes(title='Время')
            fig.update_yaxes(type='log')


            return html.Div([html.Div(html.H1(children='Прологарифмированная обобщенная концентрация от времени тестовый препарат'), style={'text-align':'center'}),
                html.Div([
                dcc.Graph(id='concentration_time_t_log_mean', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_t_log_mean), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )