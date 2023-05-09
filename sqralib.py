import openmm
from tqdm import tqdm
import molgri.molecules.parsers as mmp
import molgri.space.fullgrid as msf
import molgri.molecules.transitions as mmt
from scipy.sparse.linalg import eigs
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix, diags, coo_matrix
from scipy.constants import k as kB
from scipy.constants import N_A
import numpy.linalg as npl


def sqra_for_pt(sqra, energy_type, D=1, T=300):
    '''
    Calculates adjacency matrix and rate matrix Q for a pseudotrajectory.

    parameters:
    sqra (Molgri SQRA object=: 
        The based on the grid that you want Q for.
    
    energy_type (string):
        of energy type to be passed to sqra.sim_hist.parsed_trajectory.get_all_energies

    D (float):
        Diffusion constant for the SQRA flux

    T (float):
        Temperature in Kelvin for boltzmann probabilities used in SQRA 

    '''


    print('Creating Voronoi grid...')
    voronoi_grid = sqra.sim_hist.full_grid.get_full_voronoi_grid()
    Npoints = len(voronoi_grid.flat_positions)
    
    print('Calculating grid volumes...')
    all_volumes = voronoi_grid.get_all_voronoi_volumes()

    adjacency = get_PT_adjacency(voronoi_grid)
    
    obtained_energies = sqra.sim_hist.parsed_trajectory.get_all_energies(energy_type=energy_type)
    
    all_energies = np.empty(shape=(sqra.num_cells,))
    for a, e in zip(sqra.assignments, obtained_energies):
        if not np.isnan(a):
            #convert to joules from kJ/mole
            all_energies[int(a)] = e * (1000/N_A)

    Q = map_SQRA(voronoi_grid, adjacency, all_energies, all_volumes,  T, D)

    #Might be a problem:
    for i, q in enumerate(Q.data):
        if np.isinf(q):
            Q.data[i] = 0
    
    sums = -1*Q.sum(axis=1)
    Q = Q + diags(sums.getA1())

    return Q, adjacency

def single_sqra(i,j,grid, adjacency, energies, volumes,T, D):

    if j>i:
        
        dist = grid.get_distance_between_centers(i,j, print_message=False)
        surf = grid.get_division_area(i,j, print_message=False)
    
        flux = np.longdouble((D*surf) / dist )
        sqr = np.exp( (energies[i]-energies[j]) / (2*kB*T) , dtype=np.longdouble)
        
        q1 = (flux*sqr) / volumes[i]
        q2 = flux / (sqr*volumes[j])

        return True, q1, q2, i, j
    else:
        return False, 0, 0, 0, 0
    

def map_SQRA(grid, adjacency, energies, volumes,T, D):
    
    f = lambda ij : single_sqra(ij[0],ij[1], grid, adjacency, energies, volumes,  T, D)
    qit = map(f, zip(adjacency.row, adjacency.col))
    k = 0
    N = len(adjacency.row)
    qdata = np.zeros(N)
    rows = np.zeros(N)
    cols = np.zeros(N)
    for b, q1, q2, i, j in tqdm(qit, total=N):
        if b:
            qdata[k], qdata[k+1] = q1, q2
            rows[k], rows[k+1] = i, j
            cols[k], cols[k+1] = j, i
            k += 2

    N2 = len(grid.flat_positions)
    return csr_matrix((qdata, (rows,  cols)) , shape=(N2,N2))


def get_PT_adjacency(grid):
    gridpoints = grid.flat_positions
    translations = grid.full_grid.t_grid.trans_grid
    rotations = [x/npl.norm(x) for x in grid.full_grid.o_positions]
    indexgrid = np.zeros([len(rotations), len(translations)], dtype=np.int)

    A = dok_matrix((len(gridpoints),len(gridpoints)))

    print('Indexing grid...')
    for i,gp in enumerate(tqdm(gridpoints)):
        d = npl.norm(gp)
        rot = gp/d
        
        ti = 0
        ri = 0

        #probably needs some kind of check
        #to prevent an infinite loop if there is a problem with the grid.
        while not np.isclose(translations[ti], d):
            ti += 1

        while not np.isclose(np.dot(rot, rotations[ri]), 1):
            ri += 1
                
        indexgrid[ri, ti] = i

            
    print('Building adjacency matrix...')
    for i in tqdm(range(len(rotations))):
        for j in range(len(translations)-1):
            ki = indexgrid[i,j]
            kj = indexgrid[i,j+1]
            A[ki,kj] = 1
            A[kj,ki] = 1
            
    for i in tqdm(range(len(rotations)-1)):
        gridi = indexgrid[i, 0]
        point_1 = msf.Point(gridi, grid)

        for j in range(i+1, len(rotations)):
            gridj = indexgrid[j, 0]
            point_2 = msf.Point(gridj, grid)
                
            if grid._are_neighbours(point_1, point_2):
                for ki,kj in zip(indexgrid[i,:], indexgrid[j,:]):
                    A[ki,kj] = 1
                    A[kj,ki] = 1
                    
    print('Number of nonzero elements in adjacency matrix:', A.nnz)
    return A.tocoo()
