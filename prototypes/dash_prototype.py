import time
import dash
from dash import html, Output, Input

start_time = time.perf_counter()
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1('Battery Test Data Analysis'),
    html.Button('Click', id='btn'),
    html.Div(id='output')
])
load_time = time.perf_counter() - start_time

@app.callback(Output('output', 'children'), Input('btn', 'n_clicks'))
def update(n_clicks):
    return f'Clicked {n_clicks} times' if n_clicks else 'Not clicked'

if __name__ == '__main__':
    print(f'Dash load time: {load_time:.4f} s')
    start_interact = time.perf_counter()
    update(1)
    interaction_latency = time.perf_counter() - start_interact
    print(f'Dash interaction latency: {interaction_latency:.4f} s')
