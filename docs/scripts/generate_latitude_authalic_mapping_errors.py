import matplotlib.pyplot as plt
import numpy as np

from healpix_geo.coordinates import (
    latitude_authalic_to_geographic,
    latitude_geographic_to_authalic,
)


def plot_latitude_authalic_mapping_errors(nb_points=1000000):

    authalic_lat = np.linspace(
        -np.pi / 2, np.pi / 2, nb_points
    )  # range of authalic latitudes in radians used for distortion evaluation

    geographic_lat = latitude_authalic_to_geographic(authalic_lat, "WGS84")
    authalic_lat_roundtrip = latitude_geographic_to_authalic(geographic_lat, "WGS84")

    print(np.abs(authalic_lat_roundtrip - authalic_lat).max())

    plt.title("latitude authalic to geographic to authalic error")
    plt.plot(authalic_lat, authalic_lat_roundtrip - authalic_lat, lw=0.2, c="gray")
    plt.axhline(y=1.5e-16, ls="--", c="red", label=r"$\simeq 1$ nm on Earth")
    plt.axhline(y=-1.5e-16, ls="--", c="red")
    plt.xlabel(r"$\xi$ (rad)")
    plt.ylabel(r"$\xi'-\xi$ (rad)")
    plt.legend()
    plt.show()


def main():
    plot_latitude_authalic_mapping_errors()


if __name__ == "__main__":
    main()
