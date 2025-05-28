# reference system

Geographic data can be expressed in different coordinate systems:

- 3D cartesian
- Spherical
- Ellipsoidal
- Projected

## ellipsoidal coordinates

```{jupyter-execute}
import matplotlib.pyplot as plt
import numpy as np

a = 1
f_ = 3
f = 1 / f_

b = a * (1 - f)

angle = np.deg2rad(np.arange(0, 360, 0.01))

def angle_to_xy(angle, a, b):
    x = a * np.cos(angle)
    y = b * np.sin(angle)

    return x, y

def normal_at(angle_p, a, b):
    x_p, y_p = angle_to_xy(angle_p, a, b)

    m_x = 2 * x_p / a**2
    m_y = 2 * y_p / b**2

    return m_x, m_y


fig, ax = plt.subplots(figsize=(12, 12))
ax.plot(*angle_to_xy(angle, a, b), color="black", linewidth=1)
ax.hlines(y=0, xmin=-a, xmax=a, color="black", linewidth=1)
ax.vlines(x=0, ymin=-b, ymax=b, color="black", linewidth=1)

angle_p = np.deg2rad(50)
m_x, m_y = normal_at(angle_p, a, b)

h_p = 0.05
x_p0, y_p0 = angle_to_xy(angle_p, a, b)
x_p = x_p0 + h_p * m_x
y_p = y_p0 + h_p * m_y

t = np.linspace(-0.05, 0.05, 100)
x_t = x_p0 - m_y * t
y_t = y_p0 + m_x * t
ax.plot(x_t, y_t, linewidth=1, color="black")

n = np.linspace(-0.5, 0.05, 200)
x_n = x_p0 + m_x * n
y_n = y_p0 + m_y * n
ax.plot(x_n, y_n, linewidth=1, color="black")

v = np.linspace(0, 1, 200)
x_geocentric = x_p0 * v
y_geocentric = y_p0 * v
ax.plot(x_geocentric, y_geocentric, linewidth=1, color="black")

ax.scatter(x_p0, y_p0, zorder=4)
ax.scatter(x_p, y_p, zorder=4)

ax.axis("equal")
ax.axis("off");
```
