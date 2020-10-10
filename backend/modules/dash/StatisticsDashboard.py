import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
import pandas as pd

from .Dashboard import Dashboard


class StatisticsDashboard(Dashboard):
	def _generate_layout(self):
		return html.Div([self._generate_table(),
						 self._generate_linear(),
						 self._generate_scatter(),
						 self._generate_heatmap(),
						 self._generate_corr(),
						 self._generate_box(),
						 self._generate_hist(),
						 self._generate_log(),
						 self._generate_linlog()])

	def _generate_table(self, max_rows=10):
		df = self.settings['data'].describe().reset_index()
		df = df[df['index'].isin(self.settings['metrics'])]
		cols = df.columns
		markdown_text='''
		Table illustrates main statistical characteristics of the sample.  
		Count: number of elements in a column.  
		Mean: $$\frac{\sum\limits_{i=1}^n x_i}{n}$$'''
		for j in range(1,len(cols)):
			for i in range(len(df)):
				df.iloc[i, j] = float('{:.3f}'.format(float(df.iloc[i, j])))
		if len(df.columns) <= 5:
			return html.Div([html.Div(html.H1(children='Describe table'), style={'text-align':'center'}),
				html.Div([
					html.Div([html.Table([
					html.Thead(
						html.Tr([html.Th(col) for col in df.columns])
					),
					html.Tbody([
						html.Tr([
							html.Td(df.iloc[i][col]) for col in df.columns
						]) for i in range(min(len(df), max_rows))
					])
				])],style={'width': '48%', 'display': 'inline-block'}),
					html.Div(dcc.Markdown(children=markdown_text), style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
					])
				], style={'margin':'50px'}
			)
		else:
			return html.Div([html.Div(html.H1(children='Desribe table'), style={'text-align':'center'}),
				dcc.Markdown(children=markdown_text),
					html.Div([html.Table([
					html.Thead(
						html.Tr([html.Th(col) for col in df.columns])
					),
					html.Tbody([
						html.Tr([
							html.Td(df.iloc[i][col]) for col in df.columns
						]) for i in range(min(len(df), max_rows))
					])
				])])	
				], style={'margin':'50px'}
			)

	def _generate_linear(self):

		def update_graph(xaxis_column_name, yaxis_column_name,):
			fig = px.scatter(
				self.settings['data'], x=xaxis_column_name, y=yaxis_column_name)
			fig.update_xaxes(title=xaxis_column_name,
							 type='linear')
			fig.update_yaxes(title=yaxis_column_name,
							 type='linear')

			return fig

		self.app.callback(dash.dependencies.Output('linear_graph', 'figure'),
						  [dash.dependencies.Input('xaxis_column_name', 'value'),
						   dash.dependencies.Input('yaxis_column_name', 'value')])(update_graph)

		markdown_text='''
		Graph shows relation between two columns of the data (x-axis coordinates are values of the data column chosen in the left dropdown
		and y-axis coordinate are values of the data column chosen in the right dropdown).
		'''

		available_indicators = self.settings['data'].columns.unique()
		return html.Div([html.Div(html.H1(children='Linear graph'), style={'text-align':'center'}),
			html.Div([
			html.Div([
				html.Div([
					dcc.Dropdown(
						id='xaxis_column_name',
						options=[{'label': i, 'value': i}
								 for i in available_indicators],
						value=available_indicators[0]
					)
				], style={'width': '48%', 'display': 'inline-block'}),
				html.Div([
					dcc.Dropdown(
						id='yaxis_column_name',
						options=[{'label': i, 'value': i}
								 for i in available_indicators],
						 value=available_indicators[1]
					)
				], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
			]),
			dcc.Graph(id='linear_graph')], style={'width': '78%', 'display': 'inline-block'}),
			html.Div(dcc.Markdown(children=markdown_text), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
		)

	def _generate_scatter(self):
		df = self.settings['data'].copy()
		df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.scatter_matrix(df, height=700)
		markdown_text='''
		Scatter matrix illustrates correlation between all columns of the data.
		'''
		return html.Div([html.Div(html.H1(children='Scatter matrix'), style={'text-align':'center'}),
			html.Div([
				html.Div(dcc.Graph(
        			id='scatter_matrix',
        			figure=fig
    			),style={'width': '78%', 'display': 'inline-block'}),
    			html.Div(dcc.Markdown(children=markdown_text), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})])
			], style={'margin':'100px'})

	def _generate_heatmap(self):
		df = self.settings['data'].copy()
		df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.imshow(df)
		markdown_text='''
		Heat map depicts the magnitude for each column using different colors.
		'''
		return html.Div([html.Div(html.H1(children='Heat map'), style={'text-align':'center'}),
			html.Div([
				html.Div(dcc.Graph(
        			id='heatmap',
        			figure=fig
    			),style={'width': '78%', 'display': 'inline-block'}),
    			html.Div(dcc.Markdown(children=markdown_text), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})])
			], style={'margin':'100px'})

	def _generate_corr(self, max_rows=10):
		df = self.settings['data'].copy()
		df.rename(columns=lambda x: x[:11], inplace=True)
		df = df.corr()
		cols = df.columns
		for j in range(len(cols)):
			for i in range(len(df)):
				df.iloc[i, j] = float('{:.3f}'.format(float(df.iloc[i, j])))
		markdown_text='''
		Table shows pairwise pearson correlation of columns.
		'''
		if len(df.columns) <= 5:
			return html.Div([html.Div(html.H1(children='Correlation'), style={'text-align':'center'}),
				html.Div([
					html.Div([html.Table([
					html.Thead(
						html.Tr([html.Th(col) for col in df.columns])
					),
					html.Tbody([
						html.Tr([
							html.Td(df.iloc[i][col]) for col in df.columns
						]) for i in range(min(len(df), max_rows))
					])
				])],style={'width': '48%', 'display': 'inline-block'}),
					html.Div(dcc.Markdown(children=markdown_text), style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
					])
				], style={'margin':'50px'}
			)
		else:
			return html.Div([html.Div(html.H1(children='Correlation'), style={'text-align':'center'}),
				dcc.Markdown(children=markdown_text),
					html.Div([html.Table([
					html.Thead(
						html.Tr([html.Th(col) for col in df.columns])
					),
					html.Tbody([
						html.Tr([
							html.Td(df.iloc[i][col]) for col in df.columns
						]) for i in range(min(len(df), max_rows))
					])
				])])	
				], style={'margin':'50px'}
			)

	def _generate_box(self):
		df = self.settings['data'].copy()
		df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.box(df)
		markdown_text='''
		A box plot is a statistical representation of numerical data through their quartiles. 
		'''
		return html.Div([html.Div(html.H1(children='Box plot'), style={'text-align':'center'}),
			html.Div([
				html.Div(dcc.Graph(
        			id='box',
        			figure=fig
    			),style={'width': '78%', 'display': 'inline-block'}),
    			html.Div(dcc.Markdown(children=markdown_text), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})])
			], style={'margin':'100px'})

	def _generate_hist(self):
		df = self.settings['data'].copy()
		df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.histogram(df)
		markdown_text='''
		A histogram is representation of the distribution of numerical data, where the data are binned and the count for each bin is represented. 
		'''
		
		def update_hist(xaxis_column_name_hist):
			fig = px.histogram(
				self.settings['data'], x=xaxis_column_name_hist)
			fig.update_xaxes(title=xaxis_column_name_hist)

			return fig

		self.app.callback(dash.dependencies.Output('Histogram', 'figure'),
						  dash.dependencies.Input('xaxis_column_name_hist', 'value'))(update_hist)


		available_indicators = self.settings['data'].columns.unique()
		return html.Div([html.Div(html.H1(children='Histogram'), style={'text-align':'center'}),
					html.Div([
					html.Div([
					dcc.Dropdown(
						id='xaxis_column_name_hist',
						options=[{'label': i, 'value': i}
								 for i in available_indicators],
						value=available_indicators[0]
					),
				dcc.Graph(id='Histogram')],style={'width': '78%', 'display': 'inline-block'}),
					html.Div(dcc.Markdown(children=markdown_text), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})])],
					 style={'margin':'100px'}
		)

	def _generate_log(self):

		def update_graph(xaxis_column_name_log, yaxis_column_name_log,):
			fig = px.scatter(
				self.settings['data'], x=xaxis_column_name_log, y=yaxis_column_name_log)
			fig.update_xaxes(title=xaxis_column_name_log,
							 type='log')
			fig.update_yaxes(title=yaxis_column_name_log,
							 type='log')

			return fig

		self.app.callback(dash.dependencies.Output('log_graph', 'figure'),
						   [dash.dependencies.Input('xaxis_column_name_log', 'value'),
						   dash.dependencies.Input('yaxis_column_name_log', 'value')])(update_graph)

		markdown_text='''
		Graph shows relation between two columns of the data (x-axis coordinates are values of the data column chosen in the left dropdown
		and y-axis coordinate are values of the data column chosen in the right dropdown).
		'''

		available_indicators = self.settings['data'].columns.unique()
		return html.Div([html.Div(html.H1(children='Logarithmic graph'), style={'text-align':'center'}),
			html.Div([
			html.Div([
				html.Div([
					dcc.Dropdown(
						id='xaxis_column_name_log',
						options=[{'label': i, 'value': i}
								 for i in available_indicators],
						value=available_indicators[0]
					)
				], style={'width': '48%', 'display': 'inline-block'}),
				html.Div([
					dcc.Dropdown(
						id='yaxis_column_name_log',
						options=[{'label': i, 'value': i}
								 for i in available_indicators],
						 value=available_indicators[1]
					)
				], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
			]),
			dcc.Graph(id='log_graph')], style={'width': '78%', 'display': 'inline-block'}),
			html.Div(dcc.Markdown(children=markdown_text), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
		)

	def _generate_linlog(self):

		def update_graph(xaxis_column_name_linlog, yaxis_column_name_linlog,
						 xaxis_type_linlog, yaxis_type_linlog):
			fig = px.scatter(
				self.settings['data'], x=xaxis_column_name_linlog, y=yaxis_column_name_linlog)
			fig.update_xaxes(title=xaxis_column_name_linlog,
							 type='linear' if xaxis_type_linlog == 'Linear' else 'log')
			fig.update_yaxes(title=yaxis_column_name_linlog,
							 type='linear' if yaxis_type_linlog == 'Linear' else 'log')

			return fig

		self.app.callback(dash.dependencies.Output('linlog_graph', 'figure'),
						   [dash.dependencies.Input('xaxis_column_name_linlog', 'value'),
						   dash.dependencies.Input('yaxis_column_name_linlog', 'value')],
						  dash.dependencies.Input('xaxis_type_linlog', 'value'),
						  dash.dependencies.Input('yaxis_type_linlog', 'value'))(update_graph)

		available_indicators = self.settings['data'].columns.unique()
		return html.Div([
			html.Div([
				html.Div([
					dcc.Dropdown(
						id='xaxis_column_name_linlog',
						options=[{'label': i, 'value': i}
								 for i in available_indicators],
						value=available_indicators[0]
					),
					dcc.RadioItems(
						id='xaxis_type_linlog',
						options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
						value='Linear'
					)
				], style={'width': '48%', 'display': 'inline-block'}),
				html.Div([
					dcc.Dropdown(
						id='yaxis_column_name_linlog',
						options=[{'label': i, 'value': i} for i in available_indicators],
						value=available_indicators[0]
					),
					dcc.RadioItems(
						id='yaxis_type_linlog',
						options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
						value='Linear'
					)
				], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
			]),
			dcc.Graph(id='linlog_graph')]
		)