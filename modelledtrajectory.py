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
    translates coordinates from normal arrays (from MDA) to OpenMM format

vec32array:
    translates coordinates from openMM format to normal arrays (generally for use in MDA)


Variables
---------
_nbmethods:
    Dictionary to allow nonbondedMethods to be specified as a string when the class is initialized.
    

'''
import numpy as np
import numpy.linalg as la
from typing import Tuple, Optional, List

from tqdm import tqdm
import openff.toolkit as off
from openff.interchange import Interchange

import openmm
import openmm.app as app
from openmm.unit import *
from openmmforcefields.generators import GAFFTemplateGenerator

import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.coordinates.memory import MemoryReader



'''
Classes
-------
'''

class ModelledTrajectory:
    '''
    Keep a simulation of an MD system to evaluate points on a trajectory

    Combines an MDAnalysis universe and OpenMM simulation. Allows for evaluation of each frame in the trajectory loaded into the MDA universe. Changes the coordinates of the simulation box when a new frame is called, so only one frame is loaded at a time. Specifically designed to work with pseudotrajectories generated by the MolGri package. On initializing loads the trajectory and starts a simulation based on the files provided. Requires openmm objects as input.


    Parameters
    ----------
    topology (openmm.Topology):
        Topology to be used in the simulation part of the class, in openmm topology format. 

    system (openmm.System):
        Openmm system to be used in the simulation part of the class. 

    Integrator (any openmm Integrator object):
        The integrator to be used for the simulation. 

    trajTop (str): 
        topology (either openmm or mda), or filename of the first trajectory/topology file for the MDA universe.
        Must be the same as, or describe the same system as, the topology and system  parameters.

    trajectory (str): 
        optional (list of) trajectory coordinate file(s) matching the topology in trajtop. Optional because some topology files (like .pdb) also contain all the coordinates of the trajectory. 


    Properties
    ----------
    _tu (MDAnalysis.core.universe):
        Internal property that stores all the trajectory frames and coordinates to be read into the simulation when needed.

    _topology (openmm.Topology):
        The input topology. kept as a parameter since many openmm functions require a reference to this object.

    _integrator (any openmm integrator object):
        Input integrator kept as reference.

    _system (openmm.System):
        The input system used for the simulation box. Kept in case any properties need to be changes that require a restart of the simulation box

    _simulation (openmm.app.simulation.Simulation):
        Internal property that contains the simulation box used for evaluation of trajectory frames 
        
    _reporters ( {openmm.app.pdbreporter.PDBReporter} ):
        Dictionary of openmm pdbreporter objects created for output. This dict is used to keep track of which reporters are active, indexed by the output file they are writing to.

    '''
    
    def __init__(self, topology, system, integrator, trajTop, trajectory:List[str] = None):
        ´''' initilizes both an mda universe and an openmm simulation box of the same system to keep in parallel'''
        if trajectory == None:
            self._tu = mda.Universe(trajTop)
        else:
            self._tu = mda.Universe(trajTop, trajectory)
        self._reporters = {}
        self._slices = {}

        self._topology =  topology
        self._system = system
        self._integrator = integrator
        self._simulation = app.Simulation(self._topology, self._system, self._integrator)

    def __getitem__(self, index:int):
        '''Change the coordinates in the simulation box to frame index'''

        self._tu.trajectory[index]
        pos = array2vec3([atm.position/10 for atm in self._tu.atoms])
        self._simulation.context.setPositions(pos)


    def __len__(self):
        '''The length of the trajectory defines the length of this class'''

        return len(self._tu.trajectory)
            

    def frames(self, trajslice = None, select:List[int]=[]):
        '''Iterate over (a slice of ) the trajectory, or a different trajectory for the same system'''
        return ModelIterator(self, trajslice, select)

    
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
        
    
    def gridSim(self, steps:int, forcePicks:List[int], outputfolder:str = 'output', reportsteps:int = 0, select:List[int]=[], saveCheckpoint:bool = False):
        '''
        Run a Simulation on each (selected) point in the pseudotrajectory/grid. 

        Reports resulting trajectories and potential data in a separate file for each visited frame/gridpoint, in a folder specified in the input. 
        If any simulations fail (e.g. due to a too large timestep) the number of the failed sim is stored and the function moves on to the next frame.
        Will optionally output (average) forces on selected atoms. 
        
        '''

        if reportsteps == 0:
            reportsteps = steps

        success = []
        fail = []
        forcenames=[]
        for i, f in enumerate(self._system.getForces()):
                f.setForceGroup(i)
                forcenames.append(f.getName())

        with open(outputfolder+'/forces.txt', 'w') as ff:
            ff.write(str(forcenames))
        
        
        for i in self.frames(select=select):
            if select == []:
                n = i
            else:
                n = select[i]

            tfailed = False
            try:
                self._simulation.context.setVelocitiesToTemperature(self._integrator.getTemperature())
            except AttributeError:
                if not tfailed:
                    print('Integrator has no temperature to set velocities')
                    tfailed = True
                
                
            self._simulation.reporters.append(app.DCDReporter(outputfolder+'/trajectory_{}.dcd'.format(n), reportsteps))
            self._simulation.reporters.append(ProbeReporter(outputfolder+'/data_{}.csv'.format(n), reportsteps, forcePicks, len(self._system.getForces())))
            try:
                self._simulation.step(steps)
            except:
                fail.append(n)
            else:
                success.append(n)
                if saveCheckpoint:
                    self._simulation.saveCheckpoint(outputfolder+'/checkpoint_{}.chk'.format(n))
                    
            self._simulation.reporters = []
            self._simulation.currentStep = 0

            


        return success, fail


    def gridMinimize(self, maxIt:int = 0, tfile:str = 'minimized_trajectory.dcd', select:List[int]=[], printProgress = False):
        '''
        Run a Simulation on each (selected) point in the pseudotrajectory/grid. 

        Returns lists specifying which minimizations succeeded or failed. Saves a pseudotrajectory of all succesfully minimized frames.'''
        success = []
        fail = []

        rep = app.DCDReporter(tfile, 1)
        for i in self.frames(select=select):
            if select == []:
                n = i
            else:
                n = select[i]
            try:
                self._simulation.minimizeEnergy(maxIterations = maxIt)
            except:
                if printProgress:
                    print('Minimization of gridpoint {} failed'.format(n))
                fail.append(n)
            else:
                if printProgress:
                    print('Minimization of gridpoint {} completed'.format(n))
                state = self._simulation.context.getState(getPositions=True)
                rep.report(self._simulation, state)
                success.append(n)
            
        return success, fail

    def getForceNames(self):
        return [f.getName() for f in self._system.getForces()]

        
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

    def pickStateForSlice(self, slicename:str):
        '''Adds the positions in the current simulation state to a TrajSlice object'''
        if slicename not in self._slices:
            self._slices[slicename] = TrajSlice(len(self._tu.atoms))

        state = self._simulation.context.getState(getPositions=True)
        self._slices[slicename].addStateFrame(state)

    def getSlice(self, slicename:str):
        '''Finalize and return a trajectory slice, removing it from internal dict'''
        fetchSlice = self._slices.pop(slicename)
        fetchSlice.finalize()
        return fetchSlice

    def sliceTrajectory(self, select:List[int]):
        '''Extract coordinates for a selection of frames from the trajectory'''
        currentframe = self._tu.trajectory.frame
        newslice = TrajSlice(len(self._tu.atoms))
        for i in select:
            self._tu.trajectory[i]
            newslice.addUFrame(self._tu)

        self._tu.trajectory[currentframe]
        newslice.finalize()
        return newslice

    def sliceUniverse(self, trajectory = None, select:List[int]=[]):
        '''Take a trajslice object and create a copy of the universe with this trajectory'''
        if trajectory == None:
            trajectory = self.sliceTrajectory(select)

        newU = mda.Merge(self._tu.atoms)
        newU.load_new(trajectory[:]*10, format=MemoryReader, order='fac')

        return newU

   
            
        

    
class ModelIterator:
    '''
    Iterator over the specified frames in a modelledtrajectory.

    Will iterate over the entire trajectory and, at each iteration, set the simulation coordinates to that frame. Takes a list of integers as an optional argument, if provided will iterate over those frames and in that specific order.

    Arguments
    ---------
    sim (openmm.app.simulation.Simulation):
        The openMM simulation box that is changed for each iteration

    trajectory: numpy array of atom coordinates per frame to loop over.

    select ( [int] ): Optional list of integers. When specified will iterate over the specific frames in this list.
    '''
    
    def __init__(self, mt:ModelledTrajectory, trajectory = None, select:List[int] = []):
        self.numits = len(mt)
        self.sb = False
        self.tb = False
        self.mt = mt
        self.iteration = 0

        if trajectory != None:
            self.numits = len(trajectory)
            self.trajectory = trajectory
            self.tb = True
            
        if select != []:
            self.numits = len(select)
            self.select = select
            self.sb = True

        
    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration < self.numits:
            if self.sb:
                i = self.select[self.iteration]
            else:
                i = self.iteration
            
            if self.tb:
                curpos = array2vec3([vec for vec in self.trajectory[i]], nanometer)
            else:
                self.mt._tu.trajectory[i]
                curpos = array2vec3([atm.position/10 for atm in self.mt._tu.atoms])
                
            self.mt._simulation.context.setPositions(curpos)
            self.iteration += 1
            return (self.iteration - 1)
            
    
        else:
            self.mt[0]
            raise StopIteration
    def __len__(self):
        return self.numits






class TrajSlice:
    '''
    Storage for coordinates separate from the mda Universe
    
    Object used to store selected parts of the trajectory, or an altered version of the trajectory, 
    without the need to create completely new universe. Will first keep a list of frames where new
    frames can be appended. When finalizing all the frames are combined into one numpy array
    for efficiency.

    '''
    
    def __init__(self, numAtoms):
        self.numAtoms = numAtoms
        self.buildList = []
        self.trajectory = None
        self.fin = False

    def __getitem__(self, index):
        if self.fin:
            return self.trajectory[index, :, :]
        else:
            return self.buildList[index]

    def __len__(self):
        return len(self.trajectory)
    
    def addStateFrame(self, state):
        '''Append a new frame to the list building the trajectory'''
        if not self.fin:
            newframe = np.zeros([self.numAtoms, 3])
            for i, (x, y, z) in enumerate(state.getPositions(asNumpy=True)):
                newframe[i, :] = x.value_in_unit(nanometer), y.value_in_unit(nanometer), z.value_in_unit(nanometer)
                
            self.buildList.append(newframe)
        else:
            print('Trajectory has already been finalized')
        
    def addUFrame(self, u):
        
        if not self.fin:
            newframe = np.zeros([self.numAtoms, 3])
            for i, atm in enumerate(u.atoms):
                pos = atm.position * 0.1
                newframe[i, :] = pos[:]
            self.buildList.append(newframe)
        else:
            print('Trajectory has already been finalized')

            
    def finalize(self):
        if not self.fin:
            self.trajectory = np.zeros([len(self.buildList),self.numAtoms,3])
            for i,frame in enumerate(self.buildList):
                self.trajectory[i, :,:] = frame

            self.buildlist = []
            self.fin=True
            
        else:
            print('Trajectory has already been finalized')


class ProbeReporter():

    def __init__(self, f:str, reportInterval:int, selection:List[int], nforcegroups:int = 1):
        self._out = open(f, 'w')
        self._reportInterval = reportInterval
        self._probeIDs = selection
        self._nf = nforcegroups

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, False, False, None)

    def report(self, simulation, state):
        data = ['']*self._nf
        for i in range(self._nf):
            s2 = simulation.context.getState(getForces=True, getEnergy=True, groups = {i})
            forces = s2.getForces(asNumpy=True).value_in_unit(kilojoules/mole/nanometer)
            pot = s2.getPotentialEnergy().value_in_unit(kilojoules/mole)

            data[i] = str(pot)+','+str(np.array(np.mean(forces[self._probeIDs], 0)))
            
        self._out.write(','.join(data)+'\n')
        

            
'''
Functions
---------
'''

def array2vec3(positions):
    '''Translation between the mda and openMM coordinates

    MDAnalysis stores the coordinates as floats, Openmm needs the coordinates in its own Vec3 format with units explicitly provided.'''
    return [openmm.Vec3(r[0], r[1], r[2]) for r in positions]

def vec32array(positions, unit):
    out = np.zeros([len(positions), 3])
    for i,pos in enumerate(positions):
        out[i, :] = pos.value_in_unit(unit)

    return out

def draw_force_tcl(mt, df, colname, filename, posselect):
    mag = df[colname].apply(la.norm)
    scaled = (mag - mag.min()) / (mag.max() - mag.min())
    unit = df[colname] / mag
    
    
    sorting = np.argsort(-1*mag).tolist()
    with open(filename+'.tcl', 'w') as ftcl:
        
        ftcl.write('color scale method GWR\n')
        
        for n in tqdm(mt.frames(select = sorting)):
            i = sorting[n]
            r = mt._tu.atoms.positions[mt.selectIDs(posselect), :][0]
            
            cosr = np.dot(unit[i], r / la.norm(r))
            c= ( (cosr + 1)/2 )
            
            start = r - 0.6*unit[i]
            end = r + 0.6*unit[i]

            ftcl.write('graphics top color {}\n'.format(int(np.ceil(1023*c) + 33)))
            ftcl.write('graphics top cone {{{} {} {}}} {{{} {} {}}} radius {}\n'.format(
                start[0],start[1], start[2], end[0], end[1], end[2], 0.1 + 0.4*scaled[i]))


def draw_potential_tcl(mt, df, colname, filename, posselect, sel):

    sorting = np.argsort(df[colname]).tolist()
    pots = df[colname].iloc[sorting[:sel]] - df[colname].iloc[sorting[:sel]].min()
    med = pots.median()
    frac1 = 2*med
    frac2 = 2*(pots.max() - med)

    pots=pots.tolist()
    
    with open(filename+'.tcl', 'w') as ftcl:
        
        ftcl.write('color scale method GWR\n')
        
        for n in tqdm(mt.frames(select=sorting[:sel])):
            r = mt._tu.atoms.positions[mt.selectIDs(posselect), :][0]
            if pots[n] <= med:
                c = pots[n] / frac1
            else:
                c = 0.5 + ( (pots[n] - med) / frac2 ) 
                
            ftcl.write('graphics top color {}\n'.format(int(np.ceil(1023*c) + 33)))
            ftcl.write('graphics top sphere {{{} {} {}}} radius {}\n'.format(
                r[0], r[1], r[2], 0.03+0.7*(1-c)))



def solvatePT(newFile:str, forcefield, nSol:int, nosolTop, nosolU, solTop=None, model:str = 'tip3p', boxSize=None, boxVectors=None, padding=None, boxShape='cube', positiveIon='Na+', negativeIon='Cl-', ionicStrength=0*molar, neutralize=True, select:List[int] = []):
    '''
    Add water to (selected) frames in the pseudotrajectory and save the result
    '''

    if solTop == None:
        positions = array2vec3(ptU.atoms.positions/10)
        mod = app.Modeller(nosolTop, positions)
        mod.addSolvent(forcefield=forcefield, numAdded=nSol, model=model, boxSize=boxSize, boxVectors=boxVectors, padding=padding,
                       boxShape=boxShape, positiveIon=positiveIon, negativeIon=negativeIon, ionicStrength=ionicStrength,
                       neutralize=neutralize)
        solTop=mod.getTopology()

    solU=mda.Universe(solTop, trajectory=True)

    #TO FIX: get box dimensions from one of the topologies or input (needs to be multiplied because mda takes nm as angstrom)
    transform = transformations.boxdimensions.set_dimensions([30.0859, 30.0859, 30.0859, 90, 90, 90]) 

    with mda.Writer(newFile, solU.atoms.n_atoms) as W:

    for i in nosolU.trajectory[select]:
        positions = array2vec3(nosolU.atoms.positions/10)
        mod = app.Modeller(nosolTop, positions)
        mod.addSolvent(forcefield=forcefield, numAdded=nSol, model=model, boxSize=boxSize, boxVectors=boxVectors, padding=padding,
                       boxShape=boxShape, positiveIon=positiveIon, negativeIon=negativeIon, ionicStrength=ionicStrength,
                       neutralize=neutralize)
        
        solU.load_new(vec32array(mod.getPositions(), angstrom), order='fac')
        #TO FIX
        solU.trajectory.add_transformations(transform)
        cage_center = cage.center_of_mass()

        #ALso part of box dimension fix
        dim = i.triclinic_dimensions
        box_center = np.sum(dim, axis=0) / 2
        solU.atoms.translate(box_center - cage_center)
        sol.wrap(compound='residues')
        solU.atoms.translate(-cage.center_of_mass())
        W.write(solU.atoms)

    
    
    
    


'''
Variables
---------
'''

_nbmethods = {'NoCutoff':app.NoCutoff, 'CutoffNonPeriodic':app.CutoffNonPeriodic, 'CutoffPeriodic':app.CutoffPeriodic, 'Ewald':app.Ewald, 'PME':app.PME}
