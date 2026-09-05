"""
forward_sim.py

Functions for generating a simulated reciprocal-space map.

Usage
-----
python forward_sim.py paramter_file.json
"""

import sys
import matplotlib.pyplot as plt

from data_io import load_parameters, save_csv

from model import generate_grid, simulate_image

from plotting import plot_simulation



def forward_sim():
    """
    Creates a simulated reciprocal-space map using a supplied
    parameter file.
    """

    # ----------------------------------------------------
    # Check command line input
    # ----------------------------------------------------

    if len(sys.argv) != 2:

        raise ValueError(
            "Usage: python forward_sim.py parameter_file.json"
        )


    parameter_filename = sys.argv[1]

    output_filename = parameter_filename.split('/')[-1]


    # ----------------------------------------------------
    # Load parameters
    # ----------------------------------------------------

    params = load_parameters(
        parameter_filename
    )


    # ----------------------------------------------------
    # Generate reciprocal-space grid
    # ----------------------------------------------------

    xs, ys = generate_grid(
        params
    )


    # ----------------------------------------------------
    # Simulate RSM on grid
    # ----------------------------------------------------

    image = simulate_image(
        xs,
        ys,
        params,
    )


    # ----------------------------------------------------
    # Plot simulated RSM
    # ----------------------------------------------------

    plot_simulation(
        params
    )


    # ----------------------------------------------------
    # Optional CSV output
    # ----------------------------------------------------

    save_csv_choice = input(
        "Save CSV file? (y/n): "
    )


    if save_csv_choice.lower() == "y":

        output_name_csv = (
            output_filename.replace(
                ".json",
                ".csv"
            )
        )

        save_csv(
            output_name_csv,
            xs,
            ys,
            image,
        )

        print(f"Saved {output_name_csv}")


    plt.savefig(f"forward_sim_{output_filename.replace('.json', '.png')}", dpi=150, bbox_inches='tight')

    plt.show()



if __name__ == "__main__":

    forward_sim()
