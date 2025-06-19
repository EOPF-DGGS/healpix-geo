import pathlib
import re
import sys

import numpy as np
import pandas as pd
import tabulate

a = 6378137
f_ = 298.257223563
f = 1 / f_
e = np.sqrt(2 * f - f**2)

b = a * (1 - f)


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


def generate_table():
    geographic_latitude = 45.0
    parametric_latitude = float(
        np.rad2deg(np.arctan(b / a * np.tan(np.deg2rad(geographic_latitude))))
    )
    geocentric_latitude = float(
        np.rad2deg(np.arctan(b**2 / a**2 * np.tan(np.deg2rad(geographic_latitude))))
    )
    authalic_latitude = float(np.rad2deg(convert(np.deg2rad(geographic_latitude), e)))

    df = pd.DataFrame(
        [
            (
                r"Geographic ($\phi$)",
                geographic_latitude,
                geographic_latitude - geographic_latitude,
            ),
            (
                r"Parametric ($\beta$)",
                parametric_latitude,
                parametric_latitude - geographic_latitude,
            ),
            (
                r"Geocentric ($\theta$)",
                geocentric_latitude,
                geocentric_latitude - geographic_latitude,
            ),
            (
                r"Authalic ($\xi$)",
                authalic_latitude,
                authalic_latitude - geographic_latitude,
            ),
        ],
        columns=["Latitude", "Value", "Diff"],
    )

    return df


def main():
    latitudes = generate_table()

    formatted_table = tabulate.tabulate(
        latitudes.to_dict("records"),
        headers={
            "Latitude": "Latitude",
            "Value": "Value [°]",
            "Diff": r"$x - \phi$ [°]",
        },
        tablefmt="github",
        floatfmt=".8f",
    )
    col_re = re.compile(r"-(?=\|)")
    table = col_re.sub(":", formatted_table)

    if len(sys.argv) != 2:
        raise ValueError("invalid number of arguments, expected exactly 1")
    path = pathlib.Path(sys.argv[1])
    if not path.exists() or path.read_text() != table:
        path.write_text(table)


if __name__ == "__main__":
    main()
