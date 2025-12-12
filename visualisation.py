import polars as pl
import numpy as np
import os
from pathlib import Path

import plotly.graph_objects as go
import Utils.config as configs


def make_track_plot(config:configs.config,
                    ot_df, save_path: str | None = None):
    #lanes
    gc = config.gc
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
    add_line(gc.N_LANES-1) 
    #car
    x_coord_car = ot_df["tr_x"]
    y_coord_car = ot_df["tr_y"]
    speed = ot_df["s0"]*gc.INDEX_SPEED_MULTIPLIER
    custom_speed_scale = [
        [0.00, "rgb(  0,  0,  40)"],  # very low speed
        [0.25, "rgb(  0,  80, 200)"],
        [0.50, "rgb(  0, 200, 200)"],
        [0.75, "rgb(240, 220,  70)"],
        [1.00, "rgb(220,  30,  30)"],  # very high speed
    ]
    vmin, vmax = 0, gc.INDEX_SPEED_MULTIPLIER * gc.N_SPEEDS  # or speed.min(), speed.max()

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



def make_track_plot_2(  config:configs.config,
                        ot_df, 
                        show_plot, 
                        save_path):
    gc = config.gc
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
    for i in range(gc.N_LANES):
        add_line(i)

    x_coord_car = ot_df["tr_x"]
    y_coord_car = ot_df["tr_y"]
    speed = ot_df["s0"]*gc.INDEX_SPEED_MULTIPLIER

    custom_speed_scale = [
        [0.00, "rgb(  0,  0,  40)"],  # very low speed
        [0.2, "rgb(  0,  80, 200)"],
        [0.40, "rgb(  0, 200, 200)"],
        [0.60, "rgb(240, 220,  70)"],
        [1.00, "rgb(220,  30,  30)"],  # very high speed
    ]
    vmin, vmax = 0, gc.INDEX_SPEED_MULTIPLIER * gc.N_SPEEDS  # or speed.min(), speed.max()

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

def make_just_track_plot(config:configs.config,
                        ot_df, 
                        show_plot, 
                        save_path: str | None = None):
    gc = config.gc
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
    for i in range(gc.N_LANES):
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
    save_path=None):
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


#Plots a dictionary of lines
def make_just_track_plot(   pair_list,
                            show_plot, 
                            save_path: str | None = None):
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
    for i in range(gc.N_LANES):
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



from dataclasses import dataclass
import plotly.graph_objects as go

@dataclass
class Trajectory:
    x_data: np
    y_data: np
    color : str = "black"
    size : int = 1

    
#Plots a dictionary of lines
def make_just_track_plot(   lines:list,
                            show_plot, 
                            save_path: str | None = None):
    fig = go.Figure()
    def add_line(x_data, y_data, color, size):
        fig.add_trace(go.Scatter(
            x = x_data,
            y = y_data,
            mode = "markers",
            line = dict(color="white", width=size),
            marker  = dict(size=size, color=color),
            showlegend=True
            )
        ) 
    for line in lines:
        add_line(line.x_data, line.y_data, line.color, line.size)

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
