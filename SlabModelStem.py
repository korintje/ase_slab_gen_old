from ase import Atoms as ASE_Atoms
import numpy as np
from Lattice import get_cell_14, get_cell_dm_14
from SlabGenom import SlabGenom
from SlabModel import SlabModel
from SlabModelLeaf import SlabModelLeaf
from Cell import RealLattice, RealAtom


class AtomEntry:
    
    POSIT_THR = 0.01


    def __init__(self, name, lattice_vecs: np.ndarray):

        self.lattice: np.ndarray = lattice_vecs
        self.name: str = name
        self.abc: np.ndarray = np.zeros(3, dtype = float)
        self.xyz: np.ndarray = np.zeros(3, dtype = float)


    def __lt__(self, other):
        return self._compare_to_static(self, other) < 0


    def __eq__(self, other):
        if isinstance(other, AtomEntry):
            return self._equals_static(self, other)
        return False


    def __hash__(self):
        return hash(self.name)


    @staticmethod
    def _compare_to_static(entry1, entry2):
        if entry2 is None:
            return -1

        da, db, dc = entry1.abc - entry2.abc
        dxyz = dc * entry1.lattice[2]
        rr = np.sum(dxyz ** 2)
        if rr > AtomEntry.POSIT_THR ** 2:
            return -1 if dc > 0.0 else 1

        dxyz = db * entry1.lattice[1]
        rr = np.sum(dxyz ** 2)
        if rr > AtomEntry.POSIT_THR ** 2:
            return 1 if db > 0.0 else -1

        dxyz = db * entry1.lattice[0]
        rr = np.sum(dxyz ** 2)
        if rr > AtomEntry.POSIT_THR ** 2:
            return 1 if da > 0.0 else -1

        if entry1.name is None:
            return 1 if entry2.name is not None else 0

        return (entry1.name > entry2.name) - (entry1.name < entry2.name)


    @staticmethod
    def _equals_static(entry, obj):
        if entry is obj:
            return True
        if obj is None or not isinstance(obj, AtomEntry):
            return False

        other = obj
        if entry.name != other.name:
            return False

        ea, eb, ec = entry.abc
        oa, ob, oc = other.abc

        # print(abs(ea - oa + np.copysign(0.5 - ea, 1)))
        da = min(abs(ea - oa), abs(ea - oa + np.copysign(0.5 - ea, 1)))
        db = min(abs(eb - ob), abs(eb - ob + np.copysign(0.5 - eb, 1)))
        dc = min(abs(ec - oc), abs(ec - oc + np.copysign(0.5 - ec, 1)))

        dabc = np.array([da, db, dc])
        dxyz = np.dot(dabc, entry.lattice)
        rr = np.dot(dxyz, dxyz)

        return rr <= AtomEntry.POSIT_THR ** 2

"""
def adjust_atom_positions(abc: np.ndarray, threshold: float, c2z_scale: float):
    shifted_abc = abc - np.floor(abc)
    dc = 1.0 - shifted_abc[2]
    dz = dc * c2z_scale
    if dz < threshold:
        shifted_abc[2] -= 1.0
    return shifted_abc
"""

def get_auxi_props(
    offset: float,
    thickness: float,
    position_thr: float,
    max_for_cthick: int,
    step_for_cthick: float,
    unit_latt: np.ndarray, # 3 x 3 array
    unit_atoms # list of AtomEntry
):

    # Prepare lattice vectors
    auxi_latt: np.ndarray = unit_latt.copy()
    z_size_latt: float = auxi_latt[2][2]

    # Calc c_offset and thickness
    c_offset: int = offset % 1
    c_thick: float = max(0.0, thickness)

    for _istep in range(max_for_cthick):
        
        auxi_atoms = []  # List of AtomEntry

        n_thick = int(c_thick) + 1
        for i_thick in range(n_thick):

            atoms_1 = []
            atoms_2 = []
            for atom in unit_atoms:
                
                c = atom.abc[2] + c_offset
                c_org = c
                c -= int(c)
                dc = abs(c - 1.0)
                dz = dc * z_size_latt
                if dz < position_thr:
                    c -= 1.0
                zshift = abs(c - c_org) * z_size_latt
                atoms_buff = atoms_1 if zshift < position_thr else atoms_2
                c -= float(i_thick)
                dc = c - (1.0 - c_thick)
                dz = dc * z_size_latt
                if dz < (-2.0 * position_thr):
                    continue

                new_atom = AtomEntry(atom.name, unit_latt)
                new_atom.xyz = np.dot(
                    np.array([atom.abc[0], atom.abc[1], c]),
                    unit_latt,
                )

                atoms_buff.append(new_atom)

            auxi_atoms.extend(atoms_1)
            auxi_atoms.extend(atoms_2)

        if auxi_atoms:
            break
        
        c_thick += step_for_cthick

    # Scale lattice z vec
    zs = [atom.xyz[2] for atom in auxi_atoms]
    z_max = max(zs)
    z_min = min(zs)
    z_delta = max(z_max - z_min, 0.0)
    z_scale = 1.0 if z_size_latt == 0.0 else (z_delta / z_size_latt)
    auxi_latt[2] *= z_scale

    # Shift atoms to bottom of the cell
    has_thick = auxi_latt[2][2] > SlabModelStem.THICK_THR
    x_min = z_min * auxi_latt[2][0] / auxi_latt[2][2] if has_thick else 0.0
    y_min = z_min * auxi_latt[2][1] / auxi_latt[2][2] if has_thick else 0.0
    for atom in auxi_atoms:
        atom.xyz -= np.array([x_min, y_min, z_min])
    
    return auxi_atoms, auxi_latt


class SlabModelStem(SlabModel):

    DET_THR: float = 1.0e-8
    PACK_THR: float = 1.0e-6     # internal coordinate
    OFFSET_THR: float = 1.0e-12  # angstrom
    THICK_THR: float = 1.0e-12   # angstrom
    POSIT_THR: float = 1.0e-4    # angstrom
    VALUE_THR: float = 1.0e-12
    STEP_FOR_GENOMS: float = 0.50  # angstrom
    STEP_FOR_CTHICK: float = 0.05  # internal coordinate
    MAX_FOR_CTHICK: int = 20
    SLAB_FIX_THR: float = 0.1  # angstrom
    SLAB_FIX_RATE: float = 0.5 # internal coordinate


    def __init__(self, ase_atoms: ASE_Atoms, h: int, k: int, l: int):

        super().__init__()

        positions: np.ndarray = ase_atoms.get_positions()
        if positions.size == 0:
            raise ValueError("Given atoms object is blank.")

        self.setup_millers(ase_atoms, h, k, l)
        self.setup_unit_atoms_in_cell(ase_atoms)
        self.setup_unit_atoms_in_slab()

        self.latt_auxi = None   # List<List<float>>
        self.latt_slab = None   # List<List<float>>
        self.entry_auxi = []    # List<AtomEntry>
        self.entry_slab = []    # List<AtomEntry>


    def get_slab_models(self):

        if self.latt_unit is None or len(self.latt_unit) < 3:
            return [SlabModelLeaf(self, self.offset)]

        if self.latt_unit[2] is None or len(self.latt_unit[2]) < 3:
            return [SlabModelLeaf(self, self.offset)]

        nstep = int(self.latt_unit[2][2] / self.STEP_FOR_GENOMS)
        if nstep < 2:
            return [SlabModelLeaf(self, self.offset)]

        slab_genoms = {}

        for i in range(nstep):
            offset = i / nstep
            slab_genom = self.get_slab_genom(offset)
            if slab_genom is not None and slab_genom not in slab_genoms:
                slab_genoms[slab_genom] = offset

        if not slab_genoms:
            return [SlabModelLeaf(self, self.offset)]

        slab_models = [SlabModelLeaf(self, offset) for offset in slab_genoms.values()]

        return slab_models


    def get_slab_genom(self, offset):

        if self.latt_unit is None or len(self.latt_unit) < 3:
            return None
        if self.latt_unit[2] is None or len(self.latt_unit[2]) < 3:
            return None
        if not self.entry_unit:
            return None

        natom = len(self.entry_unit)
        iatom = natom

        names = []
        coords = []

        for entry in self.entry_unit:
            if entry is None:
                return None

            c1 = entry.abc[2] + offset
            c2 = c1 - int(c1)

            dc = abs(c2 - 1.0)
            dz = dc * self.latt_unit[2][2]
            if dz < self.POSIT_THR:
                c2 -= 1.0

            dz = abs(c1 - c2) * self.latt_unit[2][2]
            if iatom >= natom and dz < self.POSIT_THR:
                iatom = self.entry_unit.index(entry)

            names.append(entry.name)
            coords.append(c2 * self.latt_unit[2][2])   

        names2 = names[iatom:] + names[:iatom]
        coords2 = coords[iatom:] + coords[:iatom]

        return SlabGenom(names2, coords2)


    def calc_dipole(self, autocharge=True):

        vecA, vecB = self.latt_slab[1:]
        surface_normal = np.cross(vecA, vecB)
        
    
    def to_atoms(self, slab_model):

        self.setup_auxi_atoms(slab_model)
        self.setup_slab_atoms(slab_model)
        # self.calc_dipole()

        ase_atoms = ASE_Atoms()
        ase_atoms.set_cell(self.latt_slab)
        ase_atoms.set_pbc((True, True, False))

        entry_slab = self.entry_slab[:]
        entry_slab.sort()
        for entry in entry_slab:
            ase_atoms.append(entry.name)
            ase_atoms.positions[-1] = entry.xyz

        return ase_atoms


    def setup_slab_atoms(self, slab_model):

        # Prepare slab lattice vectors
        self.latt_slab = self.latt_unit.copy()

        # Expand slab XY axes
        a_scale = max(1, slab_model.scaleA)
        b_scale = max(1, slab_model.scaleB)
        self.latt_slab[0] = float(a_scale) * self.latt_slab[0]
        self.latt_slab[1] = float(b_scale) * self.latt_slab[1]

        # Expand slab Z axis
        z_slab = self.latt_slab[2][2]
        z_total = self.latt_auxi[2][2] + 2.0 * max(0.0, slab_model.vacuum)
        z_scale = 1.0 if z_slab == 0.0 else (z_total / z_slab)
        z_vector = self.latt_slab[2] = z_scale * self.latt_slab[2]

        # Create atoms
        self.entry_slab = []
        txyz = 0.5 * (z_vector - self.latt_auxi[2])
        for ia in range(a_scale):
            ra = ia / a_scale
            for ib in range(b_scale):
                rb = ib / b_scale
                rs = np.array([ra, rb, 0.0])
                vxyz = np.dot(rs, self.latt_slab)
                vxyz[2] = 0.0
                
                for entry in self.entry_auxi:
                    if entry is None:
                        continue
                    entry2 = AtomEntry(entry.name, self.latt_slab)
                    entry2.xyz = entry.xyz + txyz + vxyz 
                    # if not entry2.xyz_is_in(self.entry_slab):
                    self.entry_slab.append(entry2)


    def setup_auxi_atoms(self, slab_model: SlabModel):
        
        auxi_atoms = [RealAtom(entry.name, np.dot(entry.abc, self.latt_unit)) for entry in self.entry_unit]
        auxi_cell: RealLattice = RealLattice(self.latt_unit, auxi_atoms, [])
        auxi_cell.setup_bonds()
        new_atoms, new_latt = auxi_cell.to_slab(slab_model.offset, slab_model.thickness)
        self.entry_auxi = []
        for atom in new_atoms:
            entry = AtomEntry(atom.element, new_latt)
            entry.xyz = atom.get_coord()
            self.entry_auxi.append(entry)
        self.latt_auxi = new_latt
        
        # for atom in self.entry_unit:
            # print(f"ABC: {atom.abc[0]}, {atom.abc[1]}, {atom.abc[2]}")
            # print(f"XYZ: {atom.xyz[0]}, {atom.xyz[1]}, {atom.xyz[2]}")
            # inv_matrix = np.linalg.inv(atom.lattice)
            # coord = np.dot(atom.lattice, atom.abc)
            # print(f"xyz: {coord[0]}, {coord[1]}, {coord[2]}")
        """
        self.entry_auxi, self.latt_auxi = get_auxi_props(
            slab_model.offset,
            slab_model.thickness,
            self.POSIT_THR,
            self.MAX_FOR_CTHICK,
            self.STEP_FOR_CTHICK,
            self.latt_unit,
            self.entry_unit,
        )
        """             


    def setup_unit_atoms_in_cell(self, ase_atoms: ASE_Atoms):

        positions: np.ndarray = ase_atoms.get_positions()
        symbols: np.ndarray = ase_atoms.get_chemical_symbols()
        lattice_vecs: np.ndarray = ase_atoms.get_cell()
        rec_lattice = np.linalg.inv(lattice_vecs)
        self.entry_unit = []
        for position, name in zip(positions, symbols):
            # Create atom entry and convert position
            entry = AtomEntry(name, lattice_vecs)
            # entry.xyz = position
            entry.abc = np.dot(position, rec_lattice)
            self.entry_unit.append(entry)
            
        # Get connected atoms
        # unit_atoms = [RealAtom(element, position) for element, position in zip(symbols, positions)] 
        # unit_cell = RealLattice(lattice_vecs, unit_atoms, [])
        # unit_cell.setup_bonds()
        # self.unit_cell = unit_cell


    def setup_unit_atoms_in_slab(self):

        latt_int = np.stack([self.vector1, self.vector2, self.vector3])
        inv_latt = np.linalg.inv(latt_int)

        unit_atoms_set = set()

        add_count = 0

        for ia in range(self.bound_box[0][0], self.bound_box[0][1] + 1):
            for ib in range(self.bound_box[1][0], self.bound_box[1][1] + 1):
                for ic in range(self.bound_box[2][0], self.bound_box[2][1] + 1):
                    for entry in self.entry_unit:
                        abc = np.dot(entry.abc + np.array([ia, ib, ic]), inv_latt)
                        if (-SlabModelStem.PACK_THR <= abc[0] < 1.0 + SlabModelStem.PACK_THR and 
                            -SlabModelStem.PACK_THR <= abc[1] < 1.0 + SlabModelStem.PACK_THR and 
                            -SlabModelStem.PACK_THR <= abc[2] < 1.0 + SlabModelStem.PACK_THR):
                            atom = AtomEntry(entry.name, self.latt_unit)
                            # atom.abc = adjust_atom_positions(abc, SlabModelStem.POSIT_THR, self.latt_unit[2][2])
                            shifted_abc = abc - np.floor(abc)
                            dc = 1.0 - shifted_abc[2]
                            dz = dc * self.latt_unit[2][2]
                            if dz < SlabModelStem.POSIT_THR:
                                shifted_abc[2] -= 1.0
                            atom.abc = shifted_abc
                            
                            print(f"add count {add_count}")
                            add_count += 1

                            unit_atoms_set.add(atom)
        
        unit_atoms = list(unit_atoms_set)
        print("-----")
        print(f"Length: {len(unit_atoms)}")
        print("-----")
        self.entry_unit = sorted(unit_atoms)


    def setup_millers(self, ase_atoms: ASE_Atoms, h, k, l):

        positions: np.ndarray = ase_atoms.get_positions()
        if positions.size == 0:
            raise ValueError("Given atoms is blank.")

        if h == 0 and k == 0 and l == 0:
            raise ValueError("Miller indices [0, 0, 0] is not allowed.")

        self.miller1 = h
        self.miller2 = k
        self.miller3 = l

        self.setup_intercepts()
        self.setup_vectors()
        self.setup_boundary_box()
        self.setup_lattice(ase_atoms)


    def setup_intercepts(self):
        scale_min = 1
        scale_max = 1
        self.num_intercept = 0

        if self.miller1 != 0:
            scale_min = max(scale_min, abs(self.miller1))
            scale_max *= abs(self.miller1)
            self.num_intercept += 1
            self.has_intercept1 = True
        else:
            self.has_intercept1 = False

        if self.miller2 != 0:
            scale_min = max(scale_min, abs(self.miller2))
            scale_max *= abs(self.miller2)
            self.num_intercept += 1
            self.has_intercept2 = True
        else:
            self.has_intercept2 = False

        if self.miller3 != 0:
            scale_min = max(scale_min, abs(self.miller3))
            scale_max *= abs(self.miller3)
            self.num_intercept += 1
            self.has_intercept3 = True
        else:
            self.has_intercept3 = False

        if scale_min < 1:
            raise ValueError("scaleMin is not positive.")

        if scale_max < scale_min:
            raise ValueError("scaleMax < scaleMin.")

        if self.num_intercept < 1:
            raise ValueError("there are no intercepts.")

        scale = 0
        for i in range(scale_min, scale_max + 1):
            if self.has_intercept1 and (i % self.miller1) != 0:
                continue
            if self.has_intercept2 and (i % self.miller2) != 0:
                continue
            if self.has_intercept3 and (i % self.miller3) != 0:
                continue

            scale = i
            break

        if scale < 1:
            raise ValueError("cannot detect scale.")

        self.intercept1 = scale // self.miller1 if self.has_intercept1 else 0
        self.intercept2 = scale // self.miller2 if self.has_intercept2 else 0
        self.intercept3 = scale // self.miller3 if self.has_intercept3 else 0


    def setup_vectors(self):

        self.vector1: np.ndarray = np.zeros(3, dtype = int)
        self.vector2: np.ndarray = np.zeros(3, dtype = int)
        self.vector3: np.ndarray = np.zeros(3, dtype = int)

        if self.num_intercept <= 1:
            self.setup_vectors1()
        elif self.num_intercept <= 2:
            self.setup_vectors2()
        else:
            self.setup_vectors3()


    def setup_vectors1(self):
        if self.has_intercept1:
            if self.intercept1 > 0:
                self.vector1[1] = 1
                self.vector2[2] = 1
                self.vector3[0] = 1
            else:
                self.vector1[2] = 1
                self.vector2[1] = 1
                self.vector3[0] = -1

        elif self.has_intercept2:
            if self.intercept2 > 0:
                self.vector1[2] = 1
                self.vector2[0] = 1
                self.vector3[1] = 1
            else:
                self.vector1[0] = 1
                self.vector2[2] = 1
                self.vector3[1] = -1

        elif self.has_intercept3:
            if self.intercept3 > 0:
                self.vector1[0] = 1
                self.vector2[1] = 1
                self.vector3[2] = 1
            else:
                self.vector1[1] = 1
                self.vector2[0] = 1
                self.vector3[2] = -1


    def setup_vectors2(self):
        if not self.has_intercept3:  # cat in A-B plane
            sign1 = int((self.intercept1 > 0) - (self.intercept1 < 0))
            sign2 = int((self.intercept2 > 0) - (self.intercept2 < 0))
            self.vector1[2] = sign1 * sign2
            self.vector2[0] = self.intercept1
            self.vector2[1] = -self.intercept2
            self.vector3[0] = sign1
            self.vector3[1] = sign2

        elif not self.has_intercept2:  # cat in A-C plane
            sign1 = int((self.intercept1 > 0) - (self.intercept1 < 0))
            sign3 = int((self.intercept3 > 0) - (self.intercept3 < 0))
            self.vector1[1] = sign1 * sign3
            self.vector2[0] = -self.intercept1
            self.vector2[2] = self.intercept3
            self.vector3[0] = sign1
            self.vector3[2] = sign3

        elif not self.has_intercept1:  # cat in B-C plane
            sign2 = int((self.intercept2 > 0) - (self.intercept2 < 0))
            sign3 = int((self.intercept3 > 0) - (self.intercept3 < 0))
            self.vector1[0] = sign2 * sign3
            self.vector2[1] = self.intercept2
            self.vector2[2] = -self.intercept3
            self.vector3[1] = sign2
            self.vector3[2] = sign3


    def setup_vectors3(self):
        sign1 = (self.intercept1 > 0) - (self.intercept1 < 0)
        sign2 = (self.intercept2 > 0) - (self.intercept2 < 0)
        sign3 = (self.intercept3 > 0) - (self.intercept3 < 0)

        if sign3 > 0:
            self.vector1[1] = sign1 * self.intercept2
            self.vector1[2] = -sign1 * self.intercept3
            self.vector2[0] = -sign2 * self.intercept1
            self.vector2[2] = sign2 * self.intercept3
        else:
            self.vector1[0] = -sign1 * self.intercept1
            self.vector1[2] = sign1 * self.intercept3
            self.vector2[1] = sign2 * self.intercept2
            self.vector2[2] = -sign2 * self.intercept3

        self.vector3[0] = sign1
        self.vector3[1] = sign2
        self.vector3[2] = sign3


    def setup_boundary_box(self):

        self.bound_box = [[0, 0] for _ in range(3)]

        for i in range(3):
            if self.vector1[i] < 0:
                self.bound_box[i][0] += self.vector1[i]
            else:
                self.bound_box[i][1] += self.vector1[i]

            if self.vector2[i] < 0:
                self.bound_box[i][0] += self.vector2[i]
            else:
                self.bound_box[i][1] += self.vector2[i]

            if self.vector3[i] < 0:
                self.bound_box[i][0] += self.vector3[i]
            else:
                self.bound_box[i][1] += self.vector3[i]


    def setup_lattice(self, ase_atoms: ASE_Atoms):

        latt_int = np.stack([self.vector1, self.vector2, self.vector3])
        lattice: np.ndarray = ase_atoms.cell[:]
        latt_unit0 = np.dot(latt_int, lattice)
        self.latt_const = None if latt_unit0.size == 0 else get_cell_dm_14(latt_unit0)
        if self.latt_const is None or len(self.latt_const) < 6:
            raise ValueError("Lattice constants are invalid.")

        self.latt_unit = get_cell_14(self.latt_const)
        if self.latt_unit is None or self.latt_unit.size == 0 or len(self.latt_unit) < 3:
            raise ValueError("Lattice vectors are invalid.")
        for i in range(3):
            if self.latt_unit[i] is None or len(self.latt_unit[i]) < 3:
                raise ValueError(f"Lattice vector {i} is invalid.")
