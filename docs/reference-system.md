# reference system

Geographic data can be expressed in different coordinate systems:

- 3D cartesian
- Spherical
- Ellipsoidal
- Projected

3D cartesian coordinates are the simplest coordinates, but rather tricky to use to describe coordinates on earth's curved surface.

Instead, the earth's surface is commonly approximated in first order as a sphere, and in second order as an ellipsoid.

## Ellipsoidal coordinates

```{jupyter-execute}
---
hide-code: true
hide-output: true
---
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse
import numpy as np

a = 1
f_ = 3
f = 1 / f_
e = np.sqrt(2 * f - f**2)

b = a * (1 - f)

def angle_to_xy(angle, a, b):
    x = a * np.cos(angle)
    y = b * np.sin(angle)

    return x, y

def normal_at(angle_p, a, b):
    x_p, y_p = angle_to_xy(angle_p, a, b)

    m_x = 2 * x_p / a**2
    m_y = 2 * y_p / b**2

    return m_x, m_y

def q(ϕ, e):
    # from USGS Professional Paper 1395, "Map projections – A working manual" by John P. Snyder
    return (1 - e**2) * (
        np.sin(ϕ) / (1 - e**2 * np.sin(ϕ) ** 2)
        - (1 / (2 * e)) * np.log((1 - e * np.sin(ϕ)) / (1 + e * np.sin(ϕ)))
    )

def authalic_radius(a, e):
    q_p = q(np.pi / 2, e)
    return a * np.sqrt(q_p / 2)

def convert(geographic, e):
    q_ = q(geographic, e)
    q_p = q(np.pi / 2, e)
    xi = np.arcsin(q_ / q_p)

    return xi


fig, ax = plt.subplots(figsize=(12, 12))
# ellipse
ellipse = Ellipse(
    (0, 0),
    width=2 * a,
    height=2 * b,
    edgecolor="black",
    linewidth=1,
    facecolor="none"
)
ax.add_patch(ellipse)

ax.hlines(y=0, xmin=-a, xmax=a, color="black", linewidth=1)
ax.vlines(x=0, ymin=-b, ymax=b, color="black", linewidth=1)
ax.annotate("a", xy=(0.5 * a, 0), xytext=(0.5 * a, -0.05 * b))
ax.annotate("b", xy=(0, 0.5 * b), xytext=(-0.04 * a, 0.5 * b))

parametric_latitude = np.deg2rad(50)
m_x, m_y = normal_at(parametric_latitude, a, b)

# points
h_p = 0.05 * a
x_p0, y_p0 = angle_to_xy(parametric_latitude, a, b)
x_p = x_p0 + h_p * m_x
y_p = y_p0 + h_p * m_y

ax.scatter(x_p0, y_p0, zorder=4)
ax.scatter(x_p, y_p, zorder=4)

ax.annotate("$P$", xy=(x_p, y_p), xytext=(x_p + 0.015 * a, y_p - 0.05 * b))
ax.annotate("$P_0$", xy=(x_p0, y_p0), xytext=(x_p0 + 0.015 * a, y_p0))
ax.annotate("$h$", xy=(x_p0, y_p0), xytext=(x_p0 + 0.05 * a, y_p0 + 0.05 * a))

# tangent at P_0
t = np.linspace(-0.05, 0.05, 100)
x_t = x_p0 - m_y * t
y_t = y_p0 + m_x * t
ax.plot(x_t, y_t, linewidth=1, color="black")

# geographic latitude
t_y = -x_p / m_x
n = np.linspace(t_y, 0, 200)
x_geographic = x_p + m_x * n
y_geographic = y_p + m_y * n
ax.plot(x_geographic, y_geographic, linewidth=1, color="black")

t_x = -y_p / m_y
x_g0 = x_p + m_x * t_x
y_g0 = y_p + m_y * t_x
geographic_latitude = np.arctan(a / b * np.tan(parametric_latitude))
geographic_arc = Arc(
    (x_g0, y_g0),
    width=0.25 * a,
    height=0.25 * a,
    theta1=0,
    theta2=np.rad2deg(geographic_latitude),
)
ax.add_patch(geographic_arc)
ax.annotate(r"$\phi$", xy=(0, 0), xytext=(0 + 0.4 * a, 0 + 0.02 * a))

rectangular_on_tangent1 = Arc(
    (x_p0, y_p0), width=0.08 * a, height=0.08 * a, theta1=np.rad2deg(geographic_latitude), theta2=np.rad2deg(geographic_latitude) + 90,
    linewidth=0.5
)
rectangular_on_tangent2 = Arc(
    (x_p0, y_p0), width=0.095 * a, height=0.095 * a, theta1=np.rad2deg(geographic_latitude), theta2=np.rad2deg(geographic_latitude) + 90,
    linewidth=0.5
)
ax.add_patch(rectangular_on_tangent1)
ax.add_patch(rectangular_on_tangent2)

# geocentric latitude
v = np.linspace(0, 1, 200)
x_geocentric = 0 + x_p0 * v
y_geocentric = 0 + y_p0 * v
ax.plot(x_geocentric, y_geocentric, linewidth=1, color="black")

geocentric_latitude = np.arctan(b**2 / a**2 * np.tan(geographic_latitude))
geocentric_arc = Arc(
    (0.0, 0.0),
    width=0.5 * a,
    height=0.5 * a,
    theta1=0,
    theta2=np.rad2deg(geocentric_latitude),
)
ax.add_patch(geocentric_arc)
ax.annotate(r"$\theta$", xy=(0, 0), xytext=(0 + 0.19 * a, 0 + 0.02 * a))

# reduced latitude
semi_major_circle = Arc(
    (0, 0),
    width=2 * a,
    height=2 * a,
    angle=0,
    theta1=0,
    theta2=90,
    linestyle="--",
)
ax.add_patch(semi_major_circle)
ax.vlines(x=0, ymin=b, ymax=a, color="black", linestyle="--", linewidth=1)
x_r0 = x_p0
y_r0 = np.sqrt(a**2 - x_p0**2)
ax.scatter(x_r0, y_r0, zorder=4)
ax.annotate(r"$P_\beta$", xy=(x_r0, y_r0), xytext=(x_r0 + 0.015 * a, y_r0))

ax.vlines(x=x_r0, ymin=0, ymax=y_r0, color="black", linestyle="--", linewidth=1)
ax.plot([0, x_r0], [0, y_r0], color="black", linewidth=1)

reduced_arc = Arc(
    (0, 0),
    width=0.3 * a,
    height=0.3 * a,
    theta1=0,
    theta2=np.rad2deg(parametric_latitude),
)
ax.add_patch(reduced_arc)
ax.annotate(r"$\beta$", xy=(0, 0), xytext=(0 + 0.07 * a, 0 + 0.02 * a))

# authalic latitude
r_a = authalic_radius(a, e)
authalic_latitude = convert(geographic_latitude, e)
x_a = 0 + r_a * np.cos(authalic_latitude)
y_a = 0 + r_a * np.sin(authalic_latitude)

authalic_arc = Arc(
    (0, 0),
    width=0.7 * a,
    height=0.7 * a,
    theta1=0,
    theta2=np.rad2deg(authalic_latitude),
)
ax.add_patch(authalic_arc)

ax.scatter(x_a, y_a, zorder=4)
ax.annotate(r"$P_\xi$", xy=(x_a, y_a), xytext=(x_a - 0.015 * a, y_a - 0.045 * a))

ax.plot([0, x_a], [0, y_a], color="black", linewidth=1)
ax.annotate(
    r"$r_\xi$",
    xy=(
        0.5 * r_a * np.cos(authalic_latitude),
        0.5 * r_a * np.sin(authalic_latitude),
    ),
    xytext=(
        0.7 * r_a * np.cos(authalic_latitude) + 0.015 * a,
        0.7 * r_a * np.sin(authalic_latitude) - 0.015 * a,
    ),
)

authalic_circle = Arc(
    (0, 0),
    width=2 * r_a,
    height=2 * r_a,
    angle=0,
    theta1=0,
    theta2=90,
    linestyle="--",
)
ax.add_patch(authalic_circle)
ax.annotate(r"$\xi$", xy=(0, 0), xytext=(0 + 0.29 * a, 0 + 0.02 * a))

ax.axis("equal")
ax.axis("off");
fig.savefig("ellipsoidal_latitudes.png", dpi=600, bbox_inches="tight")
```

In coordinate systems based on a ellipsoid of revolution, geographic coordinates of points are expressed as tuples of longitude, latitude, and height above ellipsoid: {math}`(\lambda, \phi, h)`. This is much like the spherical coordinate system, except that the local normal on the ellipsoid in general does not pass through the intersection of the rotational axis and the equatorial plane (the ellipsoid's center).

```{figure} ellipsoidal_latitudes.png
The relation between the geographic latitude and the various auxiliary latitudes.

The point $P$ has the coordinates $(\lambda, \phi, h)$ (only the latitude is shown). Setting the height to $0$ results in $P_0$, which is used to derive all auxiliary latitudes.
```

In addition to the geographic latitude (also called the geodetic, astronomic, or common latitude), there are a range of auxiliary latitudes[^1], including:

- the parametric (or reduced) latitude $\beta$, which is the result of stretching the semi-minor axis $b$ to the length of the semi-major axis $a$, resulting in a sphere. This stretching displaces $P_0$ to $P_\beta$. The conversion is
  ```{math}
  \tan{\beta} = \frac{b}{a}\tan{\phi} = (1 - f)\tan{\phi} = \sqrt{1 - e^2}\tan{\phi}
  ```
- the geocentric latitude $\theta$, which is the spherical latitude for the local radius at $P_0$ (the distance between the center and $P_0$). The conversion is
  ```{math}
  \tan{\theta} = \frac{b^2}{a^2} \tan{\phi} = (1 - e^2)\tan{\phi}
  ```
- the authalic latitude $\xi$, which is the result of stretching $b$ and compressing $a$ such that the resulting sphere (the "authalic sphere") has the same surface area as the ellipsoid. In the process, $P_0$ is displaced to $P_\xi$ to keep the surface area of faces on the authalic sphere the same as on the ellipsoid.

  The conversion is

  ```{math}
  q(\phi) &= \frac{(1-e^2)\sin{\phi}}{1 - e^2\sin^2{\phi}} - \frac{1-e^2}{2e} \ln{\left(\frac{1-e\sin{\phi}}{1 + e\sin{\phi}}\right)} \\
  \sin{\xi} &= \frac{q(\phi)}{q(\frac{\pi}{2})}
  ```

  And the inverse has to be computed iteratively or through a series expansion.

Note that $\beta$ and $\xi$ are latitudes for auxiliary spheres.

```{jupyter-execute}
---
hide-code: true
---
from rich.table import Table
import rich.jupyter
from rich.console import Console

a = 6378137
f_ = 298.257223563
f = 1 / f_
e = np.sqrt(2 * f - f**2)

b = a * (1 - f)

geographic_latitude = 45.0
parametric_latitude = float(np.rad2deg(np.arctan(b / a * np.tan(np.deg2rad(geographic_latitude)))))
geocentric_latitude = float(np.rad2deg(np.arctan(b**2 / a**2 * np.tan(np.deg2rad(geographic_latitude)))))
authalic_latitude = float(np.rad2deg(convert(np.deg2rad(geographic_latitude), e)))

table = Table(caption=f"ellipsoidal latitudes for WGS84\n(a = {a:01f}, f_inv = {f:03f})")
table.add_column("Latitude")
table.add_column("Value", justify="right")
table.add_column("x - ϕ", justify="right")
table.add_row(r"Geographic (ϕ)", f"{geographic_latitude:.8f}", f"{geographic_latitude - geographic_latitude:.8f}")
table.add_row(r"Parametric (β)", f"{parametric_latitude:.8f}", f"{parametric_latitude - geographic_latitude:.8f}")
table.add_row(r"Geocentric (θ)", f"{geocentric_latitude:.8f}", f"{geocentric_latitude - geographic_latitude:.8f}")
table.add_row(r"Authalic (ξ)", f"{authalic_latitude:.8f}", f"{authalic_latitude - geographic_latitude:.8f}")

# rich.jupyter.print(table)
console = Console()
segments = console.render(table)
html = rich.jupyter._render_segments(segments)
text = console._render_buffer(segments)
rich.jupyter.JupyterRenderable(html, text)
```

[^1]: https://doi.org/10.3133/pp1395
