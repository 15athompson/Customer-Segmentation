import plotly.express as px
import plotly.graph_objects as go

def create_3d_scatter(data, x_column, y_column, z_column, color_column):
    """
    Create an interactive 3D scatter plot.
    """
    fig = px.scatter_3d(data, x=x_column, y=y_column, z=z_column, color=color_column)
    return fig

def create_parallel_coordinates(data, columns, color_column):
    """
    Create a parallel coordinates plot for multi-dimensional data visualization.
    """
    fig = px.parallel_coordinates(data, dimensions=columns, color=color_column)
    return fig

def create_customizable_dashboard(data):
    """
    Create a customizable dashboard with multiple plots.
    """
    # Create a layout with multiple subplots
    fig = go.Figure()
    
    # Add plots to the dashboard (example plots, customize as needed)
    fig.add_trace(go.Scatter(x=data['feature1'], y=data['feature2'], mode='markers', name='Scatter Plot'))
    fig.add_trace(go.Histogram(x=data['feature3'], name='Histogram'))
    
    # Add dropdown for selecting features
    fig.update_layout(
        updatemenus=[
            {
                'buttons': [
                    {'label': 'Feature 1', 'method': 'update', 'args': [{'visible': [True, False]}]},
                    {'label': 'Feature 2', 'method': 'update', 'args': [{'visible': [False, True]}]}
                ],
                'direction': 'down',
                'showactive': True,
            }
        ]
    )
    
    return fig
