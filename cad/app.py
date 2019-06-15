# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq             as daq
from dash.dependencies import Output, Input, State
import dash_table
import pandas as pd
import sys
import os
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from scipy.cluster.vq import vq, kmeans, whiten
from collections import OrderedDict
from collections import namedtuple

from bsplines_utilities import point_on_bspline_curve
from bsplines_utilities import curve_insert_knot
from bsplines_utilities import curve_elevate_degree
from bsplines_utilities import point_on_bspline_surface
from bsplines_utilities import surface_insert_knot
#from bsplines_utilities import surface_elevate_degree

SplineCurve   = namedtuple('SplineCurve',   'knots, degree, points')
SplineSurface = namedtuple('SplineSurface', 'knots, degree, points')
SplineVolume  = namedtuple('SplineVolume',  'knots, degree, points')

namespace = OrderedDict()
model_id = 0

# ... TODO to be moved to gallery
def make_line(origin=(0.,0.), end=(1.,0.)):
    knots  = [0., 0., 1., 1.]
    degree = 1
    n      = len(knots) - degree - 1

    P = np.zeros((n, 2))
    P[:, 0] = [origin[0], end[0]]
    P[:, 1] = [origin[1], end[1]]

    return SplineCurve(knots=knots, degree=degree, points=P)

def make_square(origin=(0,0), length=1.):
    Tu  = [0., 0., 1., 1.]
    Tv  = [0., 0., 1., 1.]
    pu = 1
    pv = 1
    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    gridu = np.unique(Tu)
    gridv = np.unique(Tv)

    origin = np.asarray(origin)

    P = np.asarray([[[0.,0.],[0.,1.]],[[1.,0.],[1.,1.]]])
    for i in range(0, 2):
        for j in range(0, 2):
            P[i,j,:] = origin + P[i,j,:]*length

    return SplineSurface(knots=(Tu, Tv), degree=(pu, pv), points=P)

def make_circle(center=(0.,0.), radius=1.):
    Tu  = [0., 0., 0., 1, 1., 1.]
    Tv  = [0., 0., 0., 1, 1., 1.]
    pu = 2
    pv = 2
    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    gridu = np.unique(Tu)
    gridv = np.unique(Tv)

    s = 1./np.sqrt(2)
    P          = np.zeros((nu,nv,2))
    P[0,0,:]   = np.asarray([-s   , -s   ])
    P[1,0,:]   = np.asarray([-2*s , 0.   ])
    P[2,0,:]   = np.asarray([-s   , s    ])
    P[0,1,:]   = np.asarray([0.   , -2*s ])
    P[1,1,:]   = np.asarray([0.   , 0.0  ])
    P[2,1,:]   = np.asarray([0.   , 2*s  ])
    P[0,2,:]   = np.asarray([s    , -s   ])
    P[1,2,:]   = np.asarray([2*s  , 0.   ])
    P[2,2,:]   = np.asarray([s    , s    ])

    P *= radius
    P[:,:,0] += center[0]
    P[:,:,1] += center[1]

    return SplineSurface(knots=(Tu, Tv), degree=(pu, pv), points=P)
# ...

# ...
def plot_curve(crv, nx=101):
    knots  = crv.knots
    degree = crv.degree
    P      = crv.points

    n  = len(knots) - degree - 1

    # ... curve
    xs = np.linspace(0., 1., nx)

    Q = np.zeros((nx, 2))
    for i,x in enumerate(xs):
        Q[i,:] = point_on_bspline_curve(knots, P, x)

    x = Q[:,0] ; y = Q[:,1]

    trace_crv = go.Scatter(
        x=x,
        y=y,
        mode = 'lines',
        name='Curve'
    )
    # ...

    # ... control polygon
    x = P[:,0] ; y = P[:,1]

    trace_ctrl = go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name='Control polygon'
    )
    # ...

    return [trace_crv, trace_ctrl]
# ...

# ...
def plot_surface(srf, Nu=101, Nv=101):
    Tu, Tv = srf.knots
    pu, pv = srf.degree
    P      = srf.points

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    gridu = np.unique(Tu)
    gridv = np.unique(Tv)

    us = np.linspace(0., 1., Nu)
    vs = np.linspace(0., 1., Nv)

    lines = []
    line_marker = dict(color='#0066FF', width=2)

    # ...
    Q = np.zeros((len(gridu), Nv, 2))
    for i,u in enumerate(gridu):
        for j,v in enumerate(vs):
            Q[i,j,:] = point_on_bspline_surface(Tu, Tv, P, u, v)

    for i in range(len(gridu)):
        lines += [go.Scatter(mode = 'lines', line=line_marker,
                             x=Q[i,:,0],
                             y=Q[i,:,1])
                 ]
    # ...

    # ...
    Q = np.zeros((Nu, len(gridv), 2))
    for i,u in enumerate(us):
        for j,v in enumerate(gridv):
            Q[i,j,:] = point_on_bspline_surface(Tu, Tv, P, u, v)

    for j in range(len(gridv)):
        lines += [go.Scatter(mode = 'lines', line=line_marker,
                             x=Q[:,j,0],
                             y=Q[:,j,1])
                 ]
    # ...

    # ... TODO control polygon
    # ...

    return lines
# ...

# =================================================================
tab_line = dcc.Tab(label='line', children=[
                              html.Label('origin'),
                              dcc.Input(id='line_origin',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('end'),
                              dcc.Input(id='line_end',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
])

# =================================================================
tab_arc = dcc.Tab(label='arc', children=[
                              html.Label('center'),
                              dcc.Input(id='arc_center',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('origin'),
                              dcc.Input(id='arc_origin',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('angle'),
                              dcc.Input(id='arc_angle',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
])

# =================================================================
tab_square = dcc.Tab(label='square', children=[
                              html.Label('origin'),
                              dcc.Input(id='square_origin',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('length'),
                              dcc.Input(id='square_length',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
])

# =================================================================
tab_circle = dcc.Tab(label='circle', children=[
                              html.Label('center'),
                              dcc.Input(id='circle_center',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('radius'),
                              dcc.Input(id='circle_radius',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
])

# =================================================================
tab_cube = dcc.Tab(label='cube', children=[
                              html.Label('origin'),
                              dcc.Input(id='cube_origin',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
])

# =================================================================
tab_cylinder = dcc.Tab(label='cylinder', children=[
                              html.Label('origin'),
                              dcc.Input(id='cylinder_origin',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
])

# =================================================================
tab_geometry_1d = dcc.Tab(label='1D', children=[
                          dcc.Tabs(children=[
                                   tab_line,
                                   tab_arc,
                          ]),
])

# =================================================================
tab_geometry_2d = dcc.Tab(label='2D', children=[
                          dcc.Tabs(children=[
                                   tab_square,
                                   tab_circle,
                          ]),
])

# =================================================================
tab_geometry_3d = dcc.Tab(label='3D', children=[
                          dcc.Tabs(children=[
                                   tab_cube,
                                   tab_cylinder,
                          ]),
])


# =================================================================
tab_loader = dcc.Tab(label='Load', children=[
                     html.Button('load', id='button_load'),
                     dcc.Store(id='loaded_model'),
                     dcc.Tabs(children=[
                              tab_geometry_1d,
                              tab_geometry_2d,
                              tab_geometry_3d
                     ]),
])


# =================================================================
tab_insert_knot = dcc.Tab(label='Insert knot', children=[
                          html.Div([
                              html.Label('Knot'),
                              dcc.Input(id='insert_knot_value',
                                        placeholder='Enter a value ...',
                                        value='',
                                        # we use text rather than number to avoid
                                        # having the incrementation/decrementation
                                        type='text'
                              ),
                              html.Label('times'),
                              daq.NumericInput(id='insert_knot_times',
                                               min=1,
                                               value=1
                              ),
                          ]),
])

# =================================================================
tab_elevate_degree = dcc.Tab(label='Elevate degree', children=[
                             html.Div([
                                 html.Label('times'),
                                 daq.NumericInput(id='elevate_degree_times',
                                                  min=0,
                                                  value=0
                                 ),
                             ]),
])

# =================================================================
tab_subdivision = dcc.Tab(label='Subdivision', children=[
                             html.Div([
                                 html.Label('times'),
                                 daq.NumericInput(id='subdivision_times',
                                                  min=0,
                                                  value=0
                                 ),
                             ]),
])

# =================================================================
tab_refinement = dcc.Tab(label='Refinement', children=[
                         dcc.Store(id='refined_model'),
                         html.Div([
                             # ...
                             html.Label('Axis'),
                             dcc.Dropdown(id="axis",
                                          options=[{'label': 'u', 'value': '0'},
                                                   {'label': 'v', 'value': '1'},
                                                   {'label': 'w', 'value': '2'}],
                                          value=[],
                                          multi=True),
                             html.Button('Apply', id='button_refine'),
                             html.Hr(),
                             # ...

                             # ...
                             dcc.Tabs(children=[
                                      tab_insert_knot,
                                      tab_elevate_degree,
                                      tab_subdivision
                             ]),
                             # ...
                         ])
])

# =================================================================
tab_translate = dcc.Tab(label='Translate', children=[
                             html.Div([
                                 html.H3(children='Translate'),
                                 html.Label('displacement'),
                             ]),
])

# =================================================================
tab_rotate = dcc.Tab(label='Rotate', children=[
                             html.Div([
                                 html.H3(children='Rotate'),
                                 html.Label('angle'),
                             ]),
])

# =================================================================
tab_homothetie = dcc.Tab(label='Homothetie', children=[
                             html.Div([
                                 html.H3(children='Homothetie'),
                                 html.Label('angle'),
                             ]),
])


# =================================================================
tab_transformation = dcc.Tab(label='Transformation', children=[
                             html.Div([
                                 # ...
                                 dcc.Tabs(children=[
                                          tab_translate,
                                          tab_rotate,
                                          tab_homothetie,
                                 ]),
                                 # ...
                             ])
])

# =================================================================
names = ['i', 'j', 'x', 'y']
tab_editor = dcc.Tab(label='Editor', children=[
                     html.Button('Edit', id='button_editor'),
                     dcc.Store(id='edit_model'),
                     html.Div([
                         dash_table.DataTable(id='editor',
                                              columns=[],
                                              editable=True),
                     ])
])

# =================================================================
tab_viewer = dcc.Tab(label='Viewer', children=[

                    html.Label('Geometry'),
                    dcc.Dropdown(id="model",
                                 options=[{'label':name, 'value':name}
                                          for name in namespace.keys()],
                                 value=[],
                                 multi=True),

                     html.Div([
                         # ...
                         html.Div([
                             dcc.Graph(id="graph")]),
                         # ...
                     ])
])

# =================================================================
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    # ...
    html.H1(children='CAID'),
    # ...

    # ...
    dcc.Tabs(id="tabs", children=[
        tab_viewer,
        tab_loader,
        tab_editor,
        tab_refinement,
        tab_transformation,
    ]),
    html.Div(id='tabs-content-example')
    # ...
])

# =================================================================
@app.callback(
    Output("loaded_model", "data"),
    [Input('button_load', 'n_clicks'),
     Input('line_origin', 'value'),
     Input('line_end', 'value'),
     Input('square_origin', 'value'),
     Input('square_length', 'value'),
     Input('circle_center', 'value'),
     Input('circle_radius', 'value')]
)
def load_model(n_clicks,
               line_origin, line_end,
               square_origin, square_length,
               circle_center, circle_radius):

    if n_clicks is None:
        return None

    if not( line_origin is '' ) and not( line_end is '' ):
        # ...
        try:
            line_origin = [float(i) for i in line_origin.split(',')]

        except:
            raise ValueError('Cannot convert line_origin')
        # ...

        # ...
        try:
            line_end = [float(i) for i in line_end.split(',')]

        except:
            raise ValueError('Cannot convert line_end')
        # ...

        spl = make_line(origin=line_origin,
                        end=line_end)

    elif not( square_origin is '' ) and not( square_length is '' ):
        # ...
        try:
            square_origin = [float(i) for i in square_origin.split(',')]

        except:
            raise ValueError('Cannot convert square_origin')
        # ...

        # ...
        try:
            square_length = float(square_length)

        except:
            raise ValueError('Cannot convert square_length')
        # ...

        spl = make_square(origin=square_origin,
                          length=square_length)

    elif not( circle_center is '' ) and  not( circle_radius is '' ):
        # ...
        try:
            circle_center = [float(i) for i in circle_center.split(',')]

        except:
            raise ValueError('Cannot convert circle_center')
        # ...

        # ...
        try:
            circle_radius = float(circle_radius)

        except:
            raise ValueError('Cannot convert circle_radius')
        # ...

        spl = make_circle(center=circle_center,
                          radius=circle_radius)

    else:
        raise NotImplementedError('TODO')

    print('load done')
    return spl

# =================================================================
@app.callback(
    Output("refined_model", "data"),
    [Input("model", "value"),
     Input('button_refine', 'n_clicks'),
     Input('insert_knot_value', 'value'),
     Input('insert_knot_times', 'value'),
     Input('elevate_degree_times', 'value'),
     Input('subdivision_times', 'value')]
)
def apply_refine(models, n_clicks, t, t_times, m, levels):

    if n_clicks is None:
        return None

    if len(models) == 0:
        return None

    if len(models) > 1:
        return None

    name  = models[0]
    model = namespace[name]

    # ... insert knot
    if not( t is '' ):
        times = int(t_times)
        t = float(t)

        if isinstance(model, SplineCurve):
            t_min = model.knots[ model.degree]
            t_max = model.knots[-model.degree]
            if t > t_min and t < t_max:
                knots, degree, P = curve_insert_knot( model.knots,
                                                      model.degree,
                                                      model.points,
                                                      t, times=times )

                model = SplineCurve(knots=knots, degree=degree, points=P)

                if not( n_clicks is None ):
                    namespace[name] = model

        elif isinstance(model, SplineSurface):
            u_min = model.knots[0][ model.degree[0]]
            u_max = model.knots[0][-model.degree[0]]
            v_min = model.knots[1][ model.degree[1]]
            v_max = model.knots[1][-model.degree[1]]
            condition = False
            # TODO
            if t > u_min and t < u_max:
                Tu, Tv, pu, pv, P = surface_insert_knot( *model.knots,
                                                         *model.degree,
                                                          model.points,
                                                         t, times=times,
                                                         axis=None)

                model = SplineSurface(knots=(Tu, Tv), degree=(pu, pv), points=P)
    # ...

    # ... degree elevation
    if m > 0:
        m = int(m)

        if isinstance(model, SplineCurve):
            knots, degree, P = curve_elevate_degree( model.knots,
                                                     model.degree,
                                                     model.points,
                                                     m=m)

            model = SplineCurve(knots=knots, degree=degree, points=P)
    # ...

    # ...subdivision
    if levels > 0:
        levels = int(levels)

        for level in range(levels):
            grid = np.unique(model.knots)
            for a,b in zip(grid[:-1], grid[1:]):
                t = (a+b)/2.

                knots, degree, P = curve_insert_knot( model.knots,
                                                      model.degree,
                                                      model.points,
                                                      t, times=1 )

                model = SplineCurve(knots=knots, degree=degree, points=P)
    # ...

    print('refinement done')
    return model


# =================================================================
@app.callback(
    [Output("model", "options"),
     Output("loaded_model", "clear_data"),
     Output("refined_model", "clear_data")],
    [Input('loaded_model', 'data'),
     Input('refined_model', 'data')]
)
def update_namespace(loaded_model, refined_model):
    data = None
    clear_load   = False
    clear_refine = False
    if not( loaded_model is None ):
        data = loaded_model
        clear_load = True

    elif not( refined_model is None ):
        data = refined_model
        clear_refine = True

    if data is None:
        print('PAR ICI', clear_load)
        options = [{'label':name, 'value':name} for name in namespace.keys()]
        return options, clear_load, clear_refine

    knots, degree, points = data
    if isinstance(knots, (tuple, list)):
        knots = [np.asarray(T) for T in knots]

    points = np.asarray(points)

    if isinstance(degree, int):
        current_model = SplineCurve(knots=knots,
                                    degree=degree,
                                    points=points)

    elif len(degree) == 2:
        current_model = SplineSurface(knots=knots,
                                      degree=degree,
                                      points=points)

    global model_id
    namespace['model_{}'.format(model_id)] = current_model
    model_id += 1

    print('PAR LA', clear_load)
    print(list(namespace.keys()))
    options = [{'label':name, 'value':name} for name in namespace.keys()]

    return options, clear_load, clear_refine


# =================================================================
@app.callback(
    Output("graph", "figure"),
    [Input("model", "value")]
)
def update_graph(models):

    if len(models) == 0:
        return {'data': []}

    # ...
    _models = []
    for model in models:
        if isinstance(model, str):
            _models += [namespace[model]]

        else:
            _models += [model]

    models = _models
    # ...

    # ...
    traces = []
    for model in models:
        if isinstance(model, SplineCurve):
            traces += plot_curve(model, nx=101)

        elif isinstance(model, SplineSurface):
            traces += plot_surface(model, Nu=101, Nv=101)

        else:

            raise TypeError('Only SplineCurve is available, given {}'.format(type(model)))

    # showlegend is True only for curves
    showlegend = len([i for i in models if not isinstance(i, SplineCurve)]) == 0

    layout = go.Layout( yaxis=dict(scaleanchor="x", scaleratio=1),
                        showlegend=showlegend )
    # ...

    return {'data': traces, 'layout': layout}


###########################################################
if __name__ == '__main__':

    app.run_server(debug=True)
