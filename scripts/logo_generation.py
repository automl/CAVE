from bokeh.io import output_file, show
from bokeh.layouts import row
from bokeh.models import Range1d
from bokeh.plotting import figure
import numpy as np
from bokeh.transform import jitter
from bokeh.models import Whisker
from bokeh.models.glyphs import HBar
from bokeh.models import ColumnDataSource, Plot, LinearAxis, Grid

# create a new plot
num = 20
y = np.linspace(0.4, np.pi - 0.4, num)
x = np.sin(y) * 0.8
x = (1 - x)
y = y / np.pi
print(x, y)
size = np.random.normal(15, 5, num)

def rand_jitter(arr):
    stdev = .025*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

x = rand_jitter(x)
y = rand_jitter(y)

s1 = figure(plot_width=250, plot_height=250, title=None)
s1.circle(x, y, size=10 + size, color="navy", alpha=0.5)

# step-function for the "A"
s2 = figure(plot_width=250, plot_height=250, title=None)
x = np.linspace(0.1, 0.9, num)
y = np.linspace(0.1, 0.9, num/2)
y = np.array(list(y) + list(y)[::-1])
for _ in range(int(num * (2/3))):
    x_ = rand_jitter(x)
    y_ = rand_jitter(y)
    s2.step(x_, y_, color="firebrick", alpha=0.5)

y = np.clip(y, a_min=0, a_max=0.5)
for _ in range(int(num * (1/3))):
    x_ = rand_jitter(x)
    y_ = rand_jitter(y)
    s2.step(x_, y_, color="firebrick", alpha=0.5)

# "heatmap" for V
s3 = figure(plot_width=250, plot_height=250, title=None)
x = list(np.linspace(0, 1, 10)) * 10
y = [[i] * 10 for i in np.linspace(0, 1, 10)]
y = [a for b in y for a in b]

s3.add_layout(Whisker(base=0.5, upper=0.9, lower=0.1))

# create and another
s4 = figure(plot_width=250, plot_height=250, title=None)
y = np.linspace(0.2, 0.8, 5)
right = np.array([0.8, 0.1, 0.8, 0.1, 0.8])
src = ColumnDataSource(dict(y=y, right=right))
s4.add_glyph(src, HBar(y="y", right="right", left=0, height=0.1, fill_color="#b3de69"))

for s in [s1, s2, s3, s4]:
    s.axis.visible = False
    s.toolbar.logo = None
    s.toolbar_location = None
    s.x_range = Range1d(0, 1)
    s.y_range = Range1d(0, 1)

s4.x_range = Range1d(-0.1, 0.9)

# put the results in a row
show(row(s1, s2, s3, s4))