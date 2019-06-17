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
from bsplines_utilities import point_on_bspline_surface
from bsplines_utilities import insert_knot_bspline_curve
from bsplines_utilities import elevate_degree_bspline_curve
from bsplines_utilities import insert_knot_bspline_surface
#from bsplines_utilities import elevate_degree_bspline_surface

from bsplines_utilities import point_on_nurbs_curve
from bsplines_utilities import point_on_nurbs_surface
from bsplines_utilities import insert_knot_nurbs_curve
from bsplines_utilities import insert_knot_nurbs_surface
from bsplines_utilities import elevate_degree_nurbs_curve


SplineCurve   = namedtuple('SplineCurve',   'knots, degree, points')
SplineSurface = namedtuple('SplineSurface', 'knots, degree, points')
SplineVolume  = namedtuple('SplineVolume',  'knots, degree, points')
NurbsCurve    = namedtuple('NurbsCurve',    'knots, degree, points, weights')
NurbsSurface  = namedtuple('NurbsSurface',  'knots, degree, points, weights')
NurbsVolume   = namedtuple('NurbsVolume',   'knots, degree, points, weights')

# ... global variables
namespace = OrderedDict()
model_id = 0
# ...

# ... global dict for time stamps
d_timestamp = OrderedDict()
d_timestamp['load'] = -10000
d_timestamp['refine'] = -10000

d_timestamp['line']     = -10000
d_timestamp['arc']      = -10000
d_timestamp['square']   = -10000
d_timestamp['circle']   = -10000
d_timestamp['cube']     = -10000
d_timestamp['cylinder'] = -10000
# ...

# ... TODO to be moved to gallery
def make_line(origin=(0.,0.), end=(1.,0.)):
    knots  = [0., 0., 1., 1.]
    degree = 1
    n      = len(knots) - degree - 1

    P = np.zeros((n, 2))
    P[:, 0] = [origin[0], end[0]]
    P[:, 1] = [origin[1], end[1]]

    return SplineCurve(knots=knots, degree=degree, points=P)

def make_arc(center=(0.,0.), radius=1., angle=90.):
    if angle == 90.:
        knots  = [0., 0., 0., 1., 1., 1.]
        degree = 2
        n      = len(knots) - degree - 1

        P = np.zeros((n, 2))
        P[:, 0] = [1., 1., 0.]
        P[:, 1] = [0., 1., 1.]

        # weights
        s2 = 1./np.sqrt(2)
        W = np.zeros(n)
        W[:] = [1., s2, 1.]

    elif angle == 120.:
        knots  = [0., 0., 0., 1., 1., 1.]
        degree = 2
        n      = len(knots) - degree - 1

        P = np.zeros((n, 2))
        a = np.cos(np.pi/6.)
        P[:, 0] = [ a, 0., -a]
        P[:, 1] = [.5, 2., .5]

        # weights
        W = np.zeros(n)
        W[:] = [1., 1./2., 1.]

    elif angle == 180.:
        knots  = [0., 0., 0., 0., 1., 1., 1., 1.]
        degree = 3
        n      = len(knots) - degree - 1

        P = np.zeros((n, 2))
        P[:, 0] = [1., 1., -1., -1.]
        P[:, 1] = [0., 2.,  2.,  0.]

        # weights
        W = np.zeros(n)
        W[:] = [1., 1./3., 1./3., 1.]

    else:
        raise NotImplementedError('TODO, given {}'.format(angle))

    P *= radius
    P[:,0] += center[0]
    P[:,1] += center[1]

    return NurbsCurve(knots=knots, degree=degree, points=P, weights=W)

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

    W       = np.zeros((3,3))
    W[0,0]  = 1.
    W[1,0]  = s
    W[2,0]  = 1.
    W[0,1]  = s
    W[1,1]  = 1.
    W[2,1]  = s
    W[0,2]  = 1.
    W[1,2]  = s
    W[2,2]  = 1.

    return NurbsSurface(knots=(Tu, Tv), degree=(pu, pv), points=P, weights=W)
# ...

# ...
def plot_curve(crv, nx=101, control_polygon=False):
    knots  = crv.knots
    degree = crv.degree
    P      = crv.points

    n  = len(knots) - degree - 1

    # ... curve
    xs = np.linspace(0., 1., nx)

    Q = np.zeros((nx, 2))

    if isinstance(crv, SplineCurve):
        for i,x in enumerate(xs):
            Q[i,:] = point_on_bspline_curve(knots, P, x)

    elif isinstance(crv, NurbsCurve):
        W = crv.weights
        for i,x in enumerate(xs):
            Q[i,:] = point_on_nurbs_curve(knots, P, W, x)

    line_marker = dict(color='#0066FF', width=2)
    x = Q[:,0] ; y = Q[:,1]

    trace_crv = go.Scatter(
        x=x,
        y=y,
        mode = 'lines',
        name='Curve',
        line=line_marker,
    )
    # ...

    if not control_polygon:
        return [trace_crv]

    # ... control polygon
    line_marker = dict(color='#ff7f0e', width=2)

    x = P[:,0] ; y = P[:,1]

    trace_ctrl = go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name='Control polygon',
        line=line_marker,
    )
    # ...

    return [trace_crv, trace_ctrl]
# ...

# ...
def plot_surface(srf, Nu=101, Nv=101, control_polygon=False):
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
    if isinstance(srf, SplineSurface):
        for i,u in enumerate(gridu):
            for j,v in enumerate(vs):
                Q[i,j,:] = point_on_bspline_surface(Tu, Tv, P, u, v)

    elif isinstance(srf, NurbsSurface):
        W = srf.weights
        for i,u in enumerate(gridu):
            for j,v in enumerate(vs):
                Q[i,j,:] = point_on_nurbs_surface(Tu, Tv, P, W, u, v)

    for i in range(len(gridu)):
        lines += [go.Scatter(mode = 'lines', line=line_marker,
                             x=Q[i,:,0],
                             y=Q[i,:,1])
                 ]
    # ...

    # ...
    Q = np.zeros((Nu, len(gridv), 2))
    if isinstance(srf, SplineSurface):
        for i,u in enumerate(us):
            for j,v in enumerate(gridv):
                Q[i,j,:] = point_on_bspline_surface(Tu, Tv, P, u, v)

    elif isinstance(srf, NurbsSurface):
        W = srf.weights
        for i,u in enumerate(us):
            for j,v in enumerate(gridv):
                Q[i,j,:] = point_on_nurbs_surface(Tu, Tv, P, W, u, v)

    for j in range(len(gridv)):
        lines += [go.Scatter(mode = 'lines', line=line_marker,
                             x=Q[:,j,0],
                             y=Q[:,j,1])
                 ]
    # ...

    if not control_polygon:
        return lines

    # ... control polygon
    line_marker = dict(color='#ff7f0e', width=2)

    for i in range(nu):
        lines += [go.Scatter(mode = 'lines+markers',
                             line=line_marker,
                             x=P[i,:,0],
                             y=P[i,:,1])
                 ]

    for j in range(nv):
        lines += [go.Scatter(mode = 'lines+markers',
                             line=line_marker,
                             x=P[:,j,0],
                             y=P[:,j,1])
                 ]
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
                              html.Button('Submit', id='line_submit',
                                          n_clicks_timestamp=0),

])

# =================================================================
tab_arc = dcc.Tab(label='arc', children=[
                              html.Label('center'),
                              dcc.Input(id='arc_center',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('radius'),
                              dcc.Input(id='arc_radius',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Label('angle'),
                              dcc.Dropdown(id="arc_angle",
                                           options=[{'label': '90', 'value': '90'},
                                                    {'label': '120', 'value': '120'},
                                                    {'label': '180', 'value': '180'}],
                                           value=[],
                                           multi=False),
                              html.Button('Submit', id='arc_submit',
                                          n_clicks_timestamp=0),
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
                              html.Button('Submit', id='square_submit',
                                          n_clicks_timestamp=0),
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
                              html.Button('Submit', id='circle_submit',
                                          n_clicks_timestamp=0),
])

# =================================================================
tab_cube = dcc.Tab(label='cube', children=[
                              html.Label('origin'),
                              dcc.Input(id='cube_origin',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Button('Submit', id='cube_submit',
                                          n_clicks_timestamp=0),
])

# =================================================================
tab_cylinder = dcc.Tab(label='cylinder', children=[
                              html.Label('origin'),
                              dcc.Input(id='cylinder_origin',
                                        placeholder='Enter a value ...',
                                        value='',
                                        type='text'
                              ),
                              html.Button('Submit', id='cylinder_submit',
                                          n_clicks_timestamp=0),
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
                     html.Button('load', id='button_load',
                                 n_clicks_timestamp=0),
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
                             html.Button('Apply', id='button_refine',
                                         n_clicks_timestamp=0),
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
                     html.Button('Edit', id='button_editor',
                                 n_clicks_timestamp=0),
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
                         daq.BooleanSwitch(label='Control polygon',
                           id='control_polygon',
                           on=False
                         ),
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
    [Input('button_load', 'n_clicks_timestamp'),
     Input('line_origin', 'value'),
     Input('line_end',    'value'),
     Input('line_submit', 'n_clicks_timestamp'),
     Input('arc_center', 'value'),
     Input('arc_radius', 'value'),
     Input('arc_angle',  'value'),
     Input('arc_submit', 'n_clicks_timestamp'),
     Input('square_origin', 'value'),
     Input('square_length', 'value'),
     Input('square_submit', 'n_clicks_timestamp'),
     Input('circle_center', 'value'),
     Input('circle_radius', 'value'),
     Input('circle_submit', 'n_clicks_timestamp')]
)
def load_model(time_clicks,
               line_origin, line_end,
               line_submit_time,
               arc_center, arc_radius, arc_angle,
               arc_submit_time,
               square_origin, square_length,
               square_submit_time,
               circle_center, circle_radius,
               circle_submit_time):

    global d_timestamp

    if time_clicks <= d_timestamp['load']:
        return None

    d_timestamp['load'] = time_clicks

    if ( not( line_origin is '' ) and
         not( line_end is '' ) and
         not( line_submit_time <= d_timestamp['line'] )
       ):
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

        d_timestamp['line'] = line_submit_time

        return make_line(origin=line_origin,
                         end=line_end)

    elif ( not( arc_center is '' ) and
           not( arc_radius is '' ) and
           arc_angle and
           not( arc_submit_time <= d_timestamp['arc'] )
         ):
        # ...
        try:
            arc_center = [float(i) for i in arc_center.split(',')]

        except:
            raise ValueError('Cannot convert arc_center')
        # ...

        # ...
        try:
            arc_radius = float(arc_radius)

        except:
            raise ValueError('Cannot convert arc_radius')
        # ...

        # ...
        try:
            arc_angle = float(arc_angle)

        except:
            raise ValueError('Cannot convert arc_angle')
        # ...

        d_timestamp['arc'] = arc_submit_time

        return make_arc(center=arc_center,
                        radius=arc_radius,
                        angle=arc_angle)


    elif ( not( square_origin is '' ) and
           not( square_length is '' ) and
           not( square_submit_time <= d_timestamp['square'] )
        ):
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

        d_timestamp['square'] = square_submit_time

        return make_square(origin=square_origin,
                           length=square_length)

    elif ( not( circle_center is '' ) and
           not( circle_radius is '' ) and
           not( circle_submit_time <= d_timestamp['circle'] )
         ):
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

        d_timestamp['circle'] = circle_submit_time

        return make_circle(center=circle_center,
                           radius=circle_radius)

    else:
        return None

# =================================================================
@app.callback(
    Output("refined_model", "data"),
    [Input("model", "value"),
     Input('button_refine', 'n_clicks_timestamp'),
     Input('insert_knot_value', 'value'),
     Input('insert_knot_times', 'value'),
     Input('elevate_degree_times', 'value'),
     Input('subdivision_times', 'value')]
)
def apply_refine(models, time_clicks, t, t_times, m, levels):

    global d_timestamp

    if time_clicks <= d_timestamp['refine']:
        return None

    d_timestamp['refine'] = time_clicks

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

        if isinstance(model, (SplineCurve, NurbsCurve)):
            t_min = model.knots[ model.degree]
            t_max = model.knots[-model.degree]
            if t > t_min and t < t_max:
                if isinstance(model, SplineCurve):
                    knots, degree, P = insert_knot_bspline_curve( model.knots,
                                                          model.degree,
                                                          model.points,
                                                          t, times=times )

                    model = SplineCurve(knots=knots,
                                        degree=degree,
                                        points=P)

                elif isinstance(model, NurbsCurve):
                    knots, degree, P, W = insert_knot_nurbs_curve( model.knots,
                                                                model.degree,
                                                                model.points,
                                                                model.weights,
                                                                t, times=times )

                    model = NurbsCurve(knots=knots,
                                       degree=degree,
                                       points=P,
                                       weights=W)


                if not( n_clicks is None ):
                    namespace[name] = model

        elif isinstance(model, (SplineSurface, NurbsSurface)):
            u_min = model.knots[0][ model.degree[0]]
            u_max = model.knots[0][-model.degree[0]]
            v_min = model.knots[1][ model.degree[1]]
            v_max = model.knots[1][-model.degree[1]]
            condition = False
            # TODO
            if t > u_min and t < u_max:
                if isinstance(model, SplineSurface):
                    Tu, Tv, pu, pv, P = insert_knot_bspline_surface( *model.knots,
                                                                     *model.degree,
                                                                      model.points,
                                                                      t,
                                                                      times=times,
                                                                      axis=None)

                    model = SplineSurface(knots=(Tu, Tv), degree=(pu, pv), points=P)

                elif isinstance(model, NurbsSurface):
                    Tu, Tv, pu, pv, P, W = insert_knot_nurbs_surface( *model.knots,
                                                                      *model.degree,
                                                                       model.points,
                                                                       model.weights,
                                                                       t,
                                                                       times=times,
                                                                       axis=None)

                    model = NurbsSurface(knots=(Tu, Tv),
                                         degree=(pu, pv),
                                         points=P,
                                         weights=W)
    # ...

    # ... degree elevation
    if m > 0:
        m = int(m)

        if isinstance(model, SplineCurve):
            knots, degree, P = elevate_degree_bspline_curve( model.knots,
                                                             model.degree,
                                                             model.points,
                                                             m=m)

            model = SplineCurve(knots=knots,
                                degree=degree,
                                points=P)

        elif isinstance(model, NurbsCurve):
            knots, degree, P, W = elevate_degree_nurbs_curve( model.knots,
                                                              model.degree,
                                                              model.points,
                                                              model.weights,
                                                              m=m)

            model = NurbsCurve(knots=knots,
                               degree=degree,
                               points=P,
                               weights=W)
    # ...

    # ...subdivision
    if levels > 0:
        levels = int(levels)

        for level in range(levels):
            grid = np.unique(model.knots)
            for a,b in zip(grid[:-1], grid[1:]):
                t = (a+b)/2.

                knots, degree, P = insert_knot_bspline_curve( model.knots,
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
        options = [{'label':name, 'value':name} for name in namespace.keys()]
        return options, clear_load, clear_refine

    # ...
    weights = None
    try:
        knots, degree, points = data

        points = np.asarray(points)

    except:
        try:
            knots, degree, points, weights = data

            points = np.asarray(points)
            weights = np.asarray(weights)

        except:
            raise ValueError('Could not retrieve data')
    # ...

    if isinstance(knots, (tuple, list)):
        knots = [np.asarray(T) for T in knots]

    if isinstance(degree, int):
        if weights is None:
            current_model = SplineCurve(knots=knots,
                                        degree=degree,
                                        points=points)

        else:
            current_model = NurbsCurve(knots=knots,
                                       degree=degree,
                                       points=points,
                                       weights=weights)

    elif len(degree) == 2:
        if weights is None:
            current_model = SplineSurface(knots=knots,
                                          degree=degree,
                                          points=points)

        else:
            current_model = NurbsSurface(knots=knots,
                                         degree=degree,
                                         points=points,
                                         weights=weights)


    # ...
    global model_id
    namespace['model_{}'.format(model_id)] = current_model
    model_id += 1
    # ...

    options = [{'label':name, 'value':name} for name in namespace.keys()]

    return options, clear_load, clear_refine


# =================================================================
@app.callback(
    Output("graph", "figure"),
    [Input("model", "value"),
     Input('control_polygon', 'on')]
)
def update_graph(models, control_polygon):

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
        if isinstance(model, (SplineCurve, NurbsCurve)):
            traces += plot_curve(model,
                                 nx=101,
                                 control_polygon=control_polygon)

        elif isinstance(model, (SplineSurface, NurbsSurface)):
            traces += plot_surface(model,
                                   Nu=101,
                                   Nv=101,
                                   control_polygon=control_polygon)

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
