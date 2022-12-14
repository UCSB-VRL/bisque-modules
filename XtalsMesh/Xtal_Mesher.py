import sys
import os
import time
import numpy as np
import meshio
from tqdm import tqdm
import glob
import re
import igl
import time

start = time.time()

PURPLE = "\033[1;35m"
GREEN = "\033[1;32m"
CYAN = "\033[0;36m"
END = "\033[0m"

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def write_inp(mesh, grain_dict):
    nodes = mesh.points
    cells = mesh.cells[0][1].astype(int)
    f = open('/work/XtalMesh.inp','w')
    f.write('** Generated by XtalMesh **\n\n')
    f.write('*Node, nset=ALLNODES\n')
    for i, n in enumerate(nodes):
        f.write('%i, %s\n' % (i + 1, ", ".join(map(str, n))))
    f.write('*Element, type=C3D10\n')
    for i, e in enumerate(cells):
        f.write('%i, %s\n' % (i + 1, ", ".join(map(str, e+1))))
    f.write('*Elset, elset=ALLELEMENTS\n')
    for i in range(len(cells)):
        f.write('%i,\n' % (i + 1))
    for keys, vals in grain_dict.items():
        f.write('*Elset, elset=%s\n' % keys)
        for v in vals:
            f.write('%i,\n' % (v + 1))
    f.close()


print(PURPLE, '\n\n ================', END)
print(GREEN, ' XTAL_MESHER', END)
print(PURPLE, '================\n', END)

"""
Volume Mesh with fTetWild
"""

print('(1/4) Begin Meshing with fTetWild')
edge_length = sys.argv[1]
print(CYAN)
os.system('/fTetWild/build/./FloatTetwild_bin --input /work/Whole.stl --output /work/Volume_1.msh --disable-filtering -e 5e-4 -l' + str(edge_length))

while os.path.exists('/work/Volume_1.msh') == False:
    time.sleep(10)
print(END)
print('(1/4) Meshing Complete')


"""
Convert From Linear to Quadratic Tetrahedrons
"""

print('(2/4) Converting to Quadratic Tets')
mesh = meshio.gmsh.read('/work/Volume_1.msh')
nodes, elements = mesh.points, mesh.cells[0][1]
np.savetxt('/work/lin_nodes.txt', nodes)
np.savetxt('/work/lin_elements.txt', elements, fmt='%i')

os.system('./tet_mesh_l2q lin')

while os.path.exists('/work/lin_l2q_elements.txt') == False:
    time.sleep(10)

"""
Gather Raw Mesh and Reassociate Materials
"""

nodes = np.loadtxt('/work/lin_l2q_nodes.txt')
elements = np.loadtxt('/work/lin_l2q_elements.txt')
# element order from l2q needs changing
elements = elements[:,[0,1,2,3,4,7,5,6,8,9]]

# Calculate centroids for each element for Winding Number
cells = [("tetra10", elements)]
mesh = meshio.Mesh(nodes, cells)
numcells = len(cells[0][1])
cells = cells[0][1].astype(int)
centroid = np.mean(nodes[cells],1)

mesh.cell_data = {}
mesh.points_data = {}
grain_ids = np.zeros((numcells,))

# Load STLs for each Grain
files = glob.glob("/work/STL/*.stl")
files = natural_sort(files)

# Grain Meshes
count = 0
Gdict = {}
for ii,f in tqdm(enumerate(files), leave=True, total=len(files), ncols=0, desc="(3/4) Assigning Material Sets"):
    fname = os.path.basename(f)
    fsplit = str.split(fname, '.')
    numStr = fsplit[0]
    grnNum = int(numStr)
    numStr = numStr.zfill(4)
    V, F = igl.read_triangle_mesh(f)
    P = igl.winding_number(V, F, centroid)
    grain_ids[np.where(P > 0.01)[0]] = grnNum
    Gdict['GRAIN_'+str(grnNum)] = np.where(P > 0.01)[0]

"""
Write Output Files
"""

print("(4/4) Writing .VTK and .INP")
write_inp(mesh, Gdict)
mesh.cell_data['GrainIds']=grain_ids
meshio.vtk.write('/work/XtalMesh.vtk',mesh)


os.system('rm /work/Volume_1.msh')
os.system('rm /work/Volume_1.msh_.csv')
os.system('rm /work/Volume_1.msh__sf.obj')
os.system('rm /work/lin_nodes.txt')
os.system('rm /work/lin_elements.txt')
os.system('rm /work/lin_l2q_nodes.txt')
os.system('rm /work/lin_l2q_elements.txt')

end = time.time()

print(GREEN + 'FINISHED - Total processing time: ', int(end-start),'s\n', END)
