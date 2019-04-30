# -*- coding: utf-8 -*-
# @Time    : 2019/4/30 12:12
# @Author  : uhauha2929
import visdom
import requests
import os
import numpy as np

viz = visdom.Visdom(server='http://127.0.0.1', port=8097)
assert viz.check_connection()

# 视频下载可能比较慢，耐心等几分钟
cur_dir = os.path.dirname(__file__)
video_file = os.path.join(cur_dir, 'demo.ogv')
if not os.path.exists(video_file):
    video_url = 'http://media.w3.org/2010/05/sintel/trailer.ogv'
    res = requests.get(video_url)
    with open(video_file, "wb") as f:
        f.write(res.content)

viz.video(videofile=video_file)

# 图片
# 单张图片
viz.image(
    np.random.rand(3, 512, 256),
    opts={
        'title': 'Random',
        'showlegend': True
    }
)
# 多张图片
viz.images(
    np.random.rand(20, 3, 64, 64),
    opts={
        'title': 'multi-images',
    }
)

# 散点图
Y = np.random.rand(100)
Y = (Y[Y > 0] + 1.5).astype(int),  # 100个标签1和2

old_scatter = viz.scatter(
    X=np.random.rand(100, 2) * 100,
    Y=Y,
    opts={
        'title': 'Scatter',
        'legend': ['A', 'B'],
        'xtickmin': 0,
        'xtickmax': 100,
        'xtickstep': 10,
        'ytickmin': 0,
        'ytickmax': 100,
        'ytickstep': 10,
        'markersymbol': 'cross-thin-open',
        'width': 800,
        'height': 600
    },
)
# 更新样式
viz.update_window_opts(
    win=old_scatter,
    opts={
        'title': 'New Scatter',
        'legend': ['Apple', 'Banana'],
        'markersymbol': 'dot'
    }
)
# 3D散点图
viz.scatter(
    X=np.random.rand(100, 3),
    Y=Y,
    opts={
        'title': '3D Scatter',
        'legend': ['Men', 'Women'],
        'markersize': 5
    }
)

# 柱状图
viz.bar(X=np.random.rand(20))
viz.bar(
    X=np.abs(np.random.rand(5, 3)),  # 5个列，每列有3部分组成
    opts={
        'stacked': True,
        'legend': ['A', 'B', 'C'],
        'rownames': ['2012', '2013', '2014', '2015', '2016']
    }
)

viz.bar(
    X=np.random.rand(20, 3),
    opts={
        'stacked': False,
        'legend': ['America', 'Britsh', 'China']
    }
)

# 热力图，地理图，表面图
viz.heatmap(
    X=np.outer(np.arange(1, 6), np.arange(1, 11)),
    opts={
        'columnnames': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        'rownames': ['y1', 'y2', 'y3', 'y4', 'y5'],
        'colormap': 'Electric'
    }
)

# 地表图
x = np.tile(np.arange(1, 101), (100, 1))
y = x.transpose()
X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
viz.contour(X=X, opts=dict(colormap='Viridis'))

# 表面图
viz.surf(X=X, opts={'colormap': 'Hot'})


# line plots
Y = np.linspace(-5, 5, 100)
viz.line(
    Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),
    X=np.column_stack((Y, Y)),
    opts=dict(markers=False),
)

# line using WebGL
webgl_num_points = 200000
webgl_x = np.linspace(-1, 0, webgl_num_points)
webgl_y = webgl_x ** 3
viz.line(X=webgl_x, Y=webgl_y,
         opts=dict(title='{} points using WebGL'.format(webgl_num_points), webgl=True),
         win="WebGL demo")

# line updates
win = viz.line(
    X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
    Y=np.column_stack((np.linspace(5, 10, 10),
                       np.linspace(5, 10, 10) + 5)),
)
viz.line(
    X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
    Y=np.column_stack((np.linspace(5, 10, 10),
                       np.linspace(5, 10, 10) + 5)),
    win=win,
    update='append'
)
viz.line(
    X=np.arange(21, 30),
    Y=np.arange(1, 10),
    win=win,
    name='2',
    update='append'
)
viz.line(
    X=np.arange(1, 10),
    Y=np.arange(11, 20),
    win=win,
    name='delete this',
    update='append'
)
viz.line(
    X=np.arange(1, 10),
    Y=np.arange(11, 20),
    win=win,
    name='4',
    update='insert'
)
viz.line(X=None, Y=None, win=win, name='delete this', update='remove')

viz.line(
    X=webgl_x + 1.,
    Y=(webgl_x + 1.) ** 3,
    win="WebGL demo",
    update='append',
    opts=dict(title='{} points using WebGL'.format(webgl_num_points * 2), webgl=True)
)

win = viz.line(
    X=np.column_stack((
        np.arange(0, 10),
        np.arange(0, 10),
        np.arange(0, 10),
    )),
    Y=np.column_stack((
        np.linspace(5, 10, 10),
        np.linspace(5, 10, 10) + 5,
        np.linspace(5, 10, 10) + 10,
    )),
    opts={
        'dash': np.array(['solid', 'dash', 'dashdot']),
        'linecolor': np.array([
            [0, 191, 255],
            [0, 191, 255],
            [255, 0, 0],
        ]),
        'title': 'Different line dash types'
    }
)

viz.line(
    X=np.arange(0, 10),
    Y=np.linspace(5, 10, 10) + 15,
    win=win,
    name='4',
    update='insert',
    opts={
        'linecolor': np.array([
            [255, 0, 0],
        ]),
        'dash': np.array(['dot']),
    }
)

Y = np.linspace(0, 4, 200)
win = viz.line(
    Y=np.column_stack((np.sqrt(Y), np.sqrt(Y) + 2)),
    X=np.column_stack((Y, Y)),
    opts=dict(
        fillarea=True,
        showlegend=False,
        width=800,
        height=800,
        xlabel='Time',
        ylabel='Volume',
        ytype='log',
        title='Stacked area plot',
        marginleft=30,
        marginright=30,
        marginbottom=80,
        margintop=30,
    ),
)

# Assure that the stacked area plot isn't giant
viz.update_window_opts(
    win=win,
    opts=dict(
        width=300,
        height=300,
    ),
)
