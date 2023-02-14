'''
Module containing the ModelledTrajectory class

Classes
-------
ModelledTrajectory:
    Combines openmm simulation and mda universe to model frames in provided trajectory

ModelIterator:
    Allows for iteration over a ModelledTrajectory


Functions
---------
array2vec3:
    translates coordinates from MDA to OpenMM format


Variables
---------
_nbmethods:
    Dictionary to allow nonbondedMethods to be specified as a string when the class is initialized.
    

'''
import numpy as np
from typing import Tuple, Optional, List

from openff.toolkit import Molecule as offMol
from openff.interchange import Interchange

import openmm
import openmm.app as app
from openmm.unit import *
from openmmforcefields.generators import GAFFTemplateGenerator

import MDAnalysis as mda
from MDAnalysis.analysis import distances




'''
Classes
-------
'''


class ModelledTrajectory:
    '''
    Keep a simulation of an MD system to evaluate points on a trajectory

    Combines an MDAnalysis universe and OpenMM simulation. Allows for evaluation of each frame in the trajectory loaded into the MDA universe. Changes the coordinates of the simulation box when a new frame is called, so only one frame is loaded at a time. Specifically designed to work with pseudotrajectories generated by the MolGri package. On initializing loads the trajectory and starts a simulation based on the files provided. 


    Parameters
    ----------
    protein (str): 
        filename of the static molecule .pdb file in the molgri pseudotrajectory

    smallmol (str): 
        filename of the moving small molecule .sdf structure file. 

    trajTop (str): 
        filename of the first trajectory file, containing at least the topology, e.g. gromacs .top file or .pdb

    trajectory (str): 
        optional (list of) trajectory coordinate file(s) matching the topology in trajtop. Optional because some topology files (like .pdb) also contain all the coordinates of the trajectory. 

    xyz ( (float, float, float) ): 
        Tuple of periodic box dimensions for the simulation

    forces ( [str] ): 
        list forcefield xml files to be passed to OpenMM when setting up the ForceField

    nonbondedMethod (str): 
        the nonbonded method to be passed to the OpenMM system, as a string. Standard value is the same as in the createSystem() function.
    
    nonbondedCutoff (float): 
        float to be passed to the openMM system as the cutoff for nonbonded interactions, in nanometers. Standard value is the same as in the createSystem() function.


    Properties
    ----------
    _tu (MDAnalysis.core.universe):
        Internal property that stores all the trajectory frames and coordinates to be read into the simulation when needed.

    _simulation (openmm.app.simulation.Simulation):
        Internal property that contains the simulation box used for evaluation of trajectory frames 
        
    _system (openmm.openmm.System):
        The system created to start the simulation box. Kept in case any properties need to be changes that require a restart of the simulation box

    _reporters ( {openmm.app.pdbreporter.PDBReporter} ):
        Dictionary of openmm pdbreporter objects created for output. This dict is used to keep track of which reporters are active, indexed by the output file they are writing to.

    '''
    
    def __init__(self, protein: str, smallmol: str, trajTop: str, trajectory:Optional[str] = None, periodicBox:Tuple[float, float, float] = (10, 10, 10), forces:List[str] = ['amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml'], nonbondedMethod:str = 'NoCutoff', nonbondedCutoff:float = 1):
        #load the trajectory
        if trajectory == None:
            self._tu = mda.Universe(trajTop)
        else:
            self._tu = mda.Universe(trajTop, trajectory)
        self._reporters = {}

        #load the molecules and add a template generator for the small molecule
        #the protein is expected to be covered by the normal forcefield specified
        mol = offMol.from_file(smallmol)
        offTop = mol.to_topology()
        smalltop = offTop.to_openmm()
        gaff = GAFFTemplateGenerator(molecules=mol)
        pdbProt = app.PDBFile(protein)

        #Create a topology of the molecules combined
        mod = app.Modeller(pdbProt.getTopology(), pdbProt.positions)
        mod.add(smalltop, array2vec3(offTop.get_positions()))
        x = openmm.Vec3(periodicBox[0],0,0)
        y = openmm.Vec3(0,periodicBox[1],0)
        z = openmm.Vec3(0,0,periodicBox[2])
        topology = mod.getTopology()
        topology.setPeriodicBoxVectors([x, y, z])

        #Set up a Forcedield for the system, adding the small model parameters from GAFF
        forcefield = app.ForceField()
        for force in forces:
            forcefield.loadFile(force)
        forcefield.registerTemplateGenerator(gaff.generator)
        self._system = forcefield.createSystem(topology, nonbondedMethod=_nbmethods[nonbondedMethod], nonbondedCutoff=nonbondedCutoff*nanometer)
        integrator = openmm.LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.0004*picoseconds)
        #Create the simulation box that provides the environment for measurements along the trajectory
        self._simulation = app.Simulation(topology, self._system, integrator)

        


    def __getitem__(self, index:int):
        '''Move the simulation box to frame index'''

        self._tu.trajectory[index]
        pos = array2vec3([atm.position for atm in self._tu.atoms])
        self._simulation.context.setPositions(pos)


    def __len__(self):
        '''The length of the class is the length of the trajectory'''

        return len(self._tu.trajectory)
            

    def frames(self, select:List[int]=[]):
        '''To iterate over all, or an ordered selection specified by select, of the frames'''

        return ModelIterator(self._simulation, self._tu, select)

    
    def getPotentialEnergy(self, returnQuantity:bool = False):
        '''Return the potential energy of the current state of the simulation'''

        state = self._simulation.context.getState(getEnergy=True)
        pot = state.getPotentialEnergy()
        if returnQuantity:
            return pot
        else:
            return pot.value_in_unit(kilojoule_per_mole)

        
    def minimizeEnergy(self, tolerance:float = 10, maxIterations:int = 0):
        '''Passes call for energy minimization in the simulation'''

        self._simulation.minimizeEnergy(tolerance = Quantity(value=tolerance, unit=kilojoule/mole), maxIterations=maxIterations)

        
    def pdbReporter(self, repfile:str = 'output.pdb'):
        '''
        Write current coordinates to pdb file.

        Uses the openm pdbreporter to write trajectories to a file. If a certain file is specified for the first time a new reporter will be created for it. Calling this again will add an extra frame to that file.
        '''

        state = self._simulation.context.getState(getPositions=True)
        if repfile not in self._reporters:
            self._reporters[repfile] = app.PDBReporter(repfile, 1)
        
        self._reporters[repfile].report(self._simulation, state)

        
    def constrainAtoms(self, selection:List[int]):
        '''
        Constrain selection of atoms in the simulation

        Atom ID provided in the input list this function will constrain its postion in the simulation box by setting their mass to 0. Applies to all frames in the simulation and not just the current one! As of yet non-reversible. 
        '''

        for i in selection:
                self._system.setParticleMass(i, 0*amu)
        self._simulation.context.reinitialize()

        
    def getForces(self, asNumpy:bool = True, returnQuantity:bool = False):
        '''Report on the current forces in the system'''

        state = self._simulation.context.getState(getForces=True)
        f = state.getForces(asNumpy)
        if returnQuantity:
            return f
        else:
            return f.value_in_unit(kilojoule / (nanometer * mole))
        
    def selectIDs(self, selection:str):
        '''Passes a string of MDA atom selection language to the universe and returns selected atom IDs'''

        selected = self._tu.select_atoms(selection)
        return [atm.ix for atm in selected]

        
class ModelIterator:
    '''
    Iterator over the specified frames in a modelledtrajectory.

    Will iterate over the entire trajectory and, at each iteration, set the simulation coordinates to that frame. Takes a list of integers as an optional argument, if provided will iterate over those frames and in that specific order.

    Arguments
    ---------
    sim (openmm.app.simulation.Simulation):
        The openMM simulation box that is changed for each iteration

    u (MDAnalysis.core.universe):
      the MDAnalysis universe that contains the coordinates to be loaded in the simulation box
    
    select ( [int] ): Optional list of integers. When specified will iterate over the specific frames in this list.
    '''
    
    def __init__(self, sim, u, select=[]):
        if select == []:
            self.numits = len(u.trajectory)
            self.selected=False
        else:
            self.numits = len(select)
            self.selected=True
        self.sim = sim
        self.u = u
        self.iteration = 0
        self.select = select
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration < self.numits:
            i = self.iteration
            if self.selected:
                self.u.trajectory[self.select[i]]
            else:
                self.u.trajectory[i]
            pos = array2vec3([atm.position for atm in self.u.atoms])
            self.sim.context.setPositions(pos)
            self.iteration += 1
            return i
        else:
            self.u.trajectory[0]
            pos = array2vec3([atm.position for atm in self.u.atoms])
            self.sim.context.setPositions(pos)
            raise StopIteration
    def __len__(self):
        return self.numits


'''
Functions
---------
'''

def array2vec3(positions):
    '''Translation between the mda and openMM coordinates

    MDAnalysis stores the coordinates as floats, in angstrom. Openmm needs the coordinates in its own Vec3 format with units explicitly provided.'''
    return [openmm.Vec3(r[0], r[1], r[2])*0.1*nanometer for r in positions]



'''
Variables
---------
'''

_nbmethods = {'NoCutoff':app.NoCutoff, 'CutoffNonPeriodic':app.CutoffNonPeriodic, 'CutoffPeriodic':app.CutoffPeriodic, 'Ewald':app.Ewald, 'PME':app.PME}
