# Packages
# --------

import plotly
import plotly.graph_objects as go
import numpy as np

from typing import List

# Mean-Variance Plot
# ------------------


def mv_plot(mv_pairs: List[List[float]], rf: float = 0.05) -> None:
    mv_pairs = np.array(mv_pairs)

    risk_free_rate = rf

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mv_pairs[:, 1]**0.5, y=mv_pairs[:, 0],
                             marker=dict(color=(mv_pairs[:, 0]-risk_free_rate)/(mv_pairs[:, 1]**0.5),
                                         showscale=True,
                                         size=7,
                                         line=dict(width=1),
                                         colorscale="RdBu",
                                         colorbar=dict(title="Sharpe<br>Ratio")
                                         ),
                             mode='markers'))
    fig.update_layout(template='plotly_white',
                      xaxis=dict(title='Annualised Risk (Volatility)'),
                      yaxis=dict(title='Annualised Return'),
                      title='Sample of Random Portfolios',
                      width=850,
                      height=500)
    fig.update_xaxes(range=[0.18, 0.32])
    fig.update_yaxes(range=[0.02, 0.27])
    fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))
    fig.show()
