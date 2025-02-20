from SlabModelStem import SlabModelStem
from ase.build import surface, bulk, add_vacuum
from ase.io import write, read
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
import os

# Save slab image with surface extension
def save_slab_image(slab, filename):
    ext_slab = slab * (4, 4, 1)
    # ext_slab = slab
    fig, axarr = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
    plot_atoms(ext_slab, axarr[0], rotation=('0x,0y,0z'))
    plot_atoms(ext_slab, axarr[1], rotation=('-90x,0y,0z'))
    plot_atoms(ext_slab, axarr[2], rotation=('-90x,-90y,0z'))
    plot_atoms(ext_slab, axarr[3], rotation=('30x,0y,0z'))
    fig.savefig(filename)

material_id = "mp-3901"

cif_filename = f"{material_id}.cif"

# Get and save structure from material id
if os.path.isfile(cif_filename):
    mybulk = read(cif_filename)
    # bulk = parser.get_structures()[0]
    # print(f"Local cif file: {cif_filename} has been loaded")
else:
    mybulk = bulk("OZn", crystalstructure="wurtzite", a=3.289, b=3.289, c=5.307, alpha=90.000, u=None)
    # mybulk = mybulk.repeat((3, 3, 2))

cell = mybulk
h = 1
k = 0
l = 0

slab_model_stem = SlabModelStem(cell, h, k, l)
slab_models = slab_model_stem.get_slab_models()
for i, slab_model in enumerate(slab_models):
    slab_model.set_thickness(2.0)
    # print(slab_model.__dict__)
    slab_model.set_vacuum(10.0)
    slab_model.set_scaleA(1)
    slab_model.set_scaleB(1)
    slab_cell = slab_model.to_atoms()
    print(slab_cell)
    # print(slab_cell.get_dipole_moment())
    # add_vacuum(slab_cell, 30.0)
    save_slab_image(slab_cell, f"{i}.png")

# myslab = surface(mybulk, (0, 1, 0), 3)
# myslab = myslab.repeat((3,4,1))