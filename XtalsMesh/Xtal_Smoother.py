import igl
import numpy as np
import meshio
from tqdm import tqdm
import trimesh
import glob
import re
import os
import pymesh
from pymeshfix import _meshfix
import time
import networkx as nx

start = time.time()

PURPLE = "\033[1;35m"
GREEN = "\033[1;32m"
END = "\033[0m"

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

print(PURPLE, '\n\n ================', END)
print(GREEN, ' XTAL_SMOOTHER', END)
print(PURPLE, '================', END)


"""
Read in Data
"""
print('(1/9) Reading Triangle Data')

workdir = '/work/'
# Mesh vertices (Nv , 3)
V = np.loadtxt(workdir+'nodes.txt', skiprows=5)

# Mesh faces  (Nf , 3)
F = np.loadtxt(workdir+'triangles.txt', skiprows=9).astype(int)

# Node types (2, 3, 4, 12, 13, 14)
ntype = np.loadtxt(workdir+'nodetype.txt', skiprows=1).astype(int)
flabel = np.loadtxt(workdir+'facelabels.txt', skiprows=1, delimiter=',').astype(int)

# Setting up constraints for nodes on faces/edges/corners
xdim =[np.min(V[:,0]), np.max(V[:,0])]
ydim =[np.min(V[:,1]), np.max(V[:,1])]
zdim =[np.min(V[:,2]), np.max(V[:,2])]

xCond = 1*(V[:,0]==xdim[0]) + 1*(V[:,0]==xdim[1])
yCond = 1*(V[:,1]==ydim[0]) + 1*(V[:,1]==ydim[1])
zCond = 1*(V[:,2]==zdim[0]) + 1*(V[:,2]==zdim[1])
xV = np.where(xCond > 0)[0]
yV = np.where(yCond > 0)[0]
zV = np.where(zCond> 0 )[0]

plane = np.transpose(np.concatenate([[1*(xCond == 0)],
                                     [1*(yCond == 0)],
                                     [1*(zCond == 0)]], axis=0))

# Smoothing Parameters
itr = 20
lamda = 0.2
mu = -0.21


"""
Exterior Triple Lines
"""
# Find Triple Line Vertices on Model Boundaries
condition = 1*(ntype == 13) + 1*(ntype == 14)
triple_verts = np.where(condition > 0)[0]

# Form Graph of Mesh for Fast Neighborhood Lookup
G = nx.Graph()
for f in F:
    G.add_edges_from([(f[0], f[1]), (f[0],f[2]), (f[1],f[2])])


n13 = np.where(ntype == 13)[0]
n14 = np.where(ntype == 14)[0]

# List of Triple Line Neighbors + Weights [[v1, v2, v3], [w1, w2, w3]]
#   weights control smoothing,
#   QPs are given large weights making triple lines stable near QPs.
triple_sets = []

for v in tqdm(triple_verts, leave=True, ncols=0, desc='(2/9) Finding Exterior Triple Lines'):
    nbrs = np.array(list(G.neighbors(v)) + [v])
    tset = []
    weight = []
    triple = nbrs[np.isin(nbrs, n13)]
    quad = nbrs[np.isin(nbrs, n14)]
    for n in triple:
        tset.append(n)
        weight.append(1)
    for n in quad:
        tset.append(n)
        weight.append(100)
    triple_sets.append([tset, weight])

# Node-Wise Laplacian + Taubin Smoothing
for m in tqdm(range(itr), leave=True, ncols=0, desc='(3/9) Smoothing Exterior Triple Lines'):
    Vnew = V.copy()
    for i, v in enumerate(triple_verts):
        nbrs = V[triple_sets[i][0]]
        weights = triple_sets[i][1]
        Vnew[v] += plane[v]*lamda*((np.dot(weights, nbrs)/np.sum(weights)) - V[v])
        Vnew[v] -= plane[v]*mu*((np.dot(weights, nbrs)/np.sum(weights)) - Vnew[v])
    V = Vnew.copy()


"""
Interior Triple Lines
"""
# Find Triple Line Vertices Inside Model
condition = 1*(ntype == 3) + 1*(ntype == 4)
triple_verts = np.where(condition > 0)[0]

ext_triples = np.where(ntype == 13)[0]
int_triples = np.where(ntype == 3)[0]
ext_quads = np.where(ntype ==14)[0]
int_quads = np.where(ntype == 4)[0]

# List of Triple Line Neighbors + Weights [[v1, v2, v3], [w1, w2, w3]]
#   weights control smoothing,
#   QPs are given large weights making triple lines stable near QPs.
triple_sets = []

for v in tqdm(triple_verts, position=0, ncols = 0, leave=True, desc='(4/9) Finding Interior Triple Lines'):
    nbrs = np.unique(list(G.neighbors(v)) + [v])
    tset = []
    weight = []
    for n in nbrs:
        if n in ext_triples or n in int_triples:
            tset.append(n)
            weight.append(1)
        if n in ext_quads or n in int_quads:
            tset.append(n)
            weight.append(100)
    triple_sets.append([tset, weight])

# Node-Wise Laplacian + Taubin Smoothing
for m in tqdm(range(itr), leave=True, ncols=0, desc='(5/9) Smoothing Interior Triple Lines'):
    for i, v in enumerate(triple_verts):
        nbrs = V[triple_sets[i][0]]
        weights = triple_sets[i][1]
        Vnew[v] += plane[v]*lamda*((np.dot(weights, nbrs)/np.sum(weights)) - V[v])
        Vnew[v] -= plane[v]*mu*((np.dot(weights, nbrs)/np.sum(weights)) - Vnew[v])
    V = Vnew.copy()

"""
Boundary Smoothing
"""
# Pin Non-Grain Boundary Nodes
cond = 1*(ntype==12) + 1*(ntype==2)
pinned = 1*(cond>0) # pinned=0, free=1
pinned = np.transpose(np.tile(pinned,[3,1]))

# Laplacian Matrix Using TriMesh
mesh = trimesh.Trimesh(V, F, process = False)
L = trimesh.smoothing.laplacian_calculation(mesh, equal_weight=True)

# Vectorized Laplacian + Taubin Smoothing
for m in tqdm(range(itr), position=0, ncols=0, desc='(6/9) Smoothing Boundaries'):
    Vnew += plane*pinned*lamda*(L.dot(V) - V)
    Vnew -= plane*pinned*mu*(L.dot(Vnew) - Vnew)
    V = Vnew.copy()

"""
Write Surface Meshes
"""
if not os.path.exists('/work/STL'):
    os.makedirs('/work/STL')

grain_ids = np.unique(flabel)
for i in tqdm(range(len(grain_ids)), leave=True, ncols=0, desc="(7/9) Writing Surface Meshes"):
    gid = grain_ids[i]
    if gid == -1:
        continue
    condition = 1*(flabel[:, 0] == gid) + 1*(flabel[:, 1] == gid)
    grainF = F[condition > 0,:]
    grain_mesh = trimesh.Trimesh(V, grainF)
    trimesh.repair.fix_normals(grain_mesh)
    mesh = pymesh.form_mesh(grain_mesh.vertices, grain_mesh.faces)
    pymesh.save_mesh(workdir + 'STL/' + str(gid)+'.stl', mesh)


"""
Stitch Meshes
"""
# Load STLs for each Grain
mesh_list = []
files = glob.glob('/work/STL/*.stl')
for i in tqdm(range(len(files)), leave=True, ncols=0, desc="(8/9) Stitching Surface Meshes"):
    f = files[i]
    mesh_list.append(pymesh.load_mesh(f))
merge_mesh = pymesh.merge_meshes(mesh_list)
print('(9/9) Writing Whole Surface Mesh')
pymesh.save_mesh('/work/Whole.stl', merge_mesh)

end = time.time()

print(GREEN + 'FINISHED - Total processing time: ', int(end - start), 's\n', END)
