from pyeit.mesh.external import load_mesh, place_electrodes_equal_spacing
import matplotlib.pyplot as plt
from pyeit.visual.plot import create_mesh_plot_2, create_plot, mesh_plot
import pyeit.eit.protocol as protocol
from pyeit.eit.jac import JAC
from pyeit.eit.fem import EITForward
import numpy as np


def main():
    # The example file is oriented in the following manner:
    #   Left towards the X axis
    #   Anterior direction towards the Y axis
    # This allows the 2D mesh to be displayed in the radiological view with no transformations
    simulation_mesh_filename = (
        "example_data/mesha06_bumpychestslice_radiological_view_both_lungs_1_0-3.ply"
    )
    n_electrodes = 16

    sim_mesh = load_mesh(simulation_mesh_filename)
    electrode_nodes = place_electrodes_equal_spacing(sim_mesh, n_electrodes=16)
    sim_mesh.el_pos = np.array(electrode_nodes)

    mesh_plot(
        sim_mesh,
        el_pos=electrode_nodes,
        show_mesh=True,
        show_number=True,
        show_text=False,
        figsize=None,
    )

    fig, ax = plt.subplots()
    create_mesh_plot_2(
        ax, sim_mesh, electrodes=electrode_nodes, coordinate_labels="radiological"
    )

    protocol_obj = protocol.create(
        n_electrodes, dist_exc=int(n_electrodes / 2), step_meas=1, parser_meas="std"
    )
    fwd = EITForward(sim_mesh, protocol_obj)
    vh = fwd.solve_eit(perm=1)
    vi = fwd.solve_eit(perm=sim_mesh.perm)

    # Recon
    # Set up eit object
    pyeit_obj = JAC(sim_mesh, protocol_obj)
    pyeit_obj.setup(p=0.5, lamb=0.001, method="kotre", perm=1)

    # # Dynamic solve simulated data
    ds_sim = pyeit_obj.solve(vi, vh, normalize=False)
    solution = np.real(ds_sim)

    fig, ax = plt.subplots()
    create_plot(
        ax,
        solution,
        pyeit_obj.mesh,
        electrodes=electrode_nodes,
        coordinate_labels="radiological",
    )

    plt.show()


if __name__ == "__main__":
    main()
