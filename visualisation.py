import polars as pl
import numpy as np
import os
from pathlib import Path


import Utils.constants
from Utils.constants import FOLDER, DATA_FOLDER
from Utils.constants import N_LANES, N_SPEEDS, N_TIME_LIMIT, N_DIRECTIONS, N_SPEED_DEVIATION, N_LANE_DEVIATION, N_TIME_LIMIT
from Utils.constants import MAX_COST_VALUE

import plotly.graph_objects as go
# def make_track_plot_no_car(track_df):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#             x = track_df["x_m"].to_numpy(),
#             y = track_df["y_m"].to_numpy(),
#             mode = "markers",
#             marker  = dict(size=1),
#             name = "track"
#         ))
#     fig.update_layout(
#         width=800,
#         height=800,
#         title="Track Lanes",
#         xaxis=dict(scaleanchor="y", scaleratio=1),
#     )
#     fig.show()
#     return fig

def make_track_plot(ot_df, save_path: str | None = None):
    fig = go.Figure()
    def add_line(i):
        fig.add_trace(go.Scatter(
            x = ot_df["x_lane" + str(i)].to_numpy(),
            y = ot_df["y_lane" + str(i)].to_numpy(),
            mode = "lines",
            marker  = dict(size=1, color="lightgray"),
            line = dict(color="lightgray", width=1),
            #name = "lane " + str(i)
            )
        )
    add_line(0)
    add_line(N_LANES-1)   
    # for i in range(N_LANES):
    #     fig.add_trace(go.Scatter(
    #         x = ot_df["x_lane" + str(i)].to_numpy(),
    #         y = ot_df["y_lane" + str(i)].to_numpy(),
    #         mode = "markers",
    #         marker  = dict(size=3, color="lightgray"),
    #         #name = "lane " + str(i)
    #     ))

    x_coord_car = ot_df["tr_x"]
    y_coord_car = ot_df["tr_y"]
    speed = ot_df["s0"]*N_SPEED_DEVIATION
    custom_speed_scale = [
        [0.00, "rgb(  0,  0,  40)"],  # very low speed
        [0.25, "rgb(  0,  80, 200)"],
        [0.50, "rgb(  0, 200, 200)"],
        [0.75, "rgb(240, 220,  70)"],
        [1.00, "rgb(220,  30,  30)"],  # very high speed
    ]
    vmin, vmax = 0, N_SPEED_DEVIATION * N_SPEEDS  # or speed.min(), speed.max()

    fig.add_trace(go.Scatter(
        x=x_coord_car,
        y=y_coord_car,
        mode="lines+markers",
        #line=dict(color="red", width=1),
        line=dict(color="lightgray", width=2),
        marker=dict(
            size=2,
            color=speed,                 
            colorscale=custom_speed_scale,
            cmin=vmin,                      
            cmax=vmax,                   
            #showscale=False,
            colorbar = dict(
                title="Speed",
                thickness=12,   # narrower bar (default ~20)
                len=0.5)
        ),
        name=f"car"
    ))

    fig.update_layout(
        width=800,
        height=1000,
        title="Track Lanes",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        plot_bgcolor="white",   # inside axes
        paper_bgcolor="white",  # outside axes
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()
    if save_path is not None:
        fig.write_html(save_path)
    return fig



def make_track_plot_2(ot_df, show_plot, save_path):
    fig = go.Figure()
    def add_line(i):
        fig.add_trace(go.Scatter(
            x = ot_df["x_lane" + str(i)].to_numpy(),
            y = ot_df["y_lane" + str(i)].to_numpy(),
            mode = "lines",
            marker  = dict(size=1, color="lightgray"),
            line = dict(color="white", width=3),
            showlegend=False
            )
        )   
    for i in range(N_LANES):
        add_line(i)

    x_coord_car = ot_df["tr_x"]
    y_coord_car = ot_df["tr_y"]
    speed = ot_df["s0"]*N_SPEED_DEVIATION

    custom_speed_scale = [
        [0.00, "rgb(  0,  0,  40)"],  # very low speed
        [0.2, "rgb(  0,  80, 200)"],
        [0.40, "rgb(  0, 200, 200)"],
        [0.60, "rgb(240, 220,  70)"],
        [1.00, "rgb(220,  30,  30)"],  # very high speed
    ]
    vmin, vmax = 0, N_SPEED_DEVIATION * N_SPEEDS  # or speed.min(), speed.max()

    fig.add_trace(go.Scatter(
        x=x_coord_car,
        y=y_coord_car,
        mode="lines+markers",
        #line=dict(color="red", width=1),
        line=dict(color="lightgray", width=2),
        marker=dict(
            size=4,
            color=speed,                 
            colorscale=custom_speed_scale,
            cmin=vmin,                      
            cmax=vmax,                   
            #showscale=False,
            colorbar = dict(
                title="Speed",
                thickness=12,   # narrower bar (default ~20)
                len=0.5)
        ),
        showlegend=False,
        #name=f"car"
    ))

    fig.update_layout(
        width=800,
        height=1000,
        title="Optimal Path",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        #paper_bgcolor="#303030",  # outside plot
        #plot_bgcolor="#303030",
        plot_bgcolor="lightgray"
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    if show_plot:
        fig.show()
    if save_path is not None:
        fig.write_html(save_path)
    return fig

#def make_just_track_plot(ot_df, save_path: str | None = None):
def make_just_track_plot(ot_df, show_plot, save_path):
    fig = go.Figure()
    def add_line(i, c):
        fig.add_trace(go.Scatter(
            x = ot_df["x_lane" + str(i)].to_numpy(),
            y = ot_df["y_lane" + str(i)].to_numpy(),
            mode = "lines",
            marker  = dict(size=1, color=c),
            line = dict(color=c, width=1),
            showlegend=False
            )
        ) 
    for i in range(N_LANES):
        add_line(i,"white")  
    add_line(0,"black")
    add_line(N_LANES-1,"black")

    fig.update_layout(
        width=800,
        height=1000,
        title="Silverstone Track",
        xaxis=dict(scaleanchor="y", scaleratio=1)
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    if show_plot:
        fig.show()
    if save_path is not None:
        #fig.write_html(save_path)
        fig.write_image(save_path)
    return fig

def make_just_track_plot(ot_df, show_plot, save_path: str | None = None):
    fig = go.Figure()
    def add_line(i, c):
        fig.add_trace(go.Scatter(
            x = ot_df["x_lane" + str(i)].to_numpy(),
            y = ot_df["y_lane" + str(i)].to_numpy(),
            mode = "markers",
            line = dict(color="white", width=1),
            marker  = dict(size=1, color=c),
            showlegend=False
            )
        ) 
    for i in range(N_LANES):
        add_line(i,"lightgray")

    fig.update_layout(
        width=800,
        height=1000,
        title="Silverstone Track",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        plot_bgcolor="white"
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    if show_plot:
        fig.show()
    if save_path is not None:
        #fig.write_html(save_path)
        fig.write_html(save_path)
    return fig


def pl_row_plots(
    x_data,
    y_data_dic,
    plot_title: str,
    show_plot,
    save_path):
    fig = go.Figure()
    for key in y_data_dic.keys():
        y_data = y_data_dic[key]
        fig.add_trace(go.Scatter(
            x = x_data ,#ot_df["x_lane" + str(i)].to_numpy(),
            y = y_data,
            mode = "markers+lines",
            marker  = dict(size=2),
            name = key
        ))
    fig.update_layout(
        width=1200,
        height=400,
        title=plot_title,
    )
    if show_plot:
        fig.show()
    if save_path is not None:
        fig.write_html(save_path)

    return fig


# combined = make_subplots(rows=1, cols=3, subplot_titles=["Fig 1", "Fig 2", "Fig 3"])

# for trace in fig1.data:
#     combined.add_trace(trace, row=1, col=1)

# for trace in fig2.data:
#     combined.add_trace(trace, row=1, col=2)

# for trace in fig3.data:
#     combined.add_trace(trace, row=1, col=3)

# combined.update_layout(height=400, width=1200, title_text="All together")
# combined.show()


