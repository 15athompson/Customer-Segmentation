import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Assume we have a function to load and update data
def load_data():
    # This function should load the latest customer segmentation data
    # For now, we'll use a dummy dataframe
    return pd.DataFrame({
        'Customer ID': range(1, 101),
        'Segment': ['A', 'B', 'C', 'D'] * 25,
        'Spending': np.random.randint(100, 1000, 100),
        'Frequency': np.random.randint(1, 10, 100)
    })

# Layout of the dashboard
app.layout = html.Div([
    html.H1('Customer Segmentation Dashboard'),
    
    dcc.Graph(id='segment-distribution'),
    
    dcc.Graph(id='spending-by-segment'),
    
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # in milliseconds, update every 1 minute
        n_intervals=0
    )
])

@app.callback(
    [Output('segment-distribution', 'figure'),
     Output('spending-by-segment', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n):
    df = load_data()
    
    segment_dist = px.pie(df, names='Segment', title='Customer Segment Distribution')
    
    spending_by_segment = px.box(df, x='Segment', y='Spending', title='Spending Distribution by Segment')
    
    return segment_dist, spending_by_segment

if __name__ == '__main__':
    app.run_server(debug=True)
