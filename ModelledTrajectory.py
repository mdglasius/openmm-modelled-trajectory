from openff.toolkit import Molecule as offMol
from openff.interchange import Interchange
from openmmforcefields.generators import GAFFTemplateGenerator
from openmm.app import *
from openmm.unit import *
from openmm import *
import openmm
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances

class ModelledTrajectory:
    '''
    Class that combines an MDA universe and OpenMM simulation, designed for use with pseudotrajectories from the MolGri package. Allows for evaluating simulation properties and further minimization or simulation of the provided trajectory inside the openmm simulation box. Note that velocities and kinetic energy are not set or equilibrated for the model.
    '''
    def __init__(self, protein, smallmol, trajTop, trajectory=None, xyz = [10, 10, 10], forces=['amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml']):
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
        pdbProt = PDBFile(protein)

        #Create a topology of the molecules combined
        mod = Modeller(pdbProt.getTopology(), pdbProt.positions)
        mod.add(smalltop, array2vec3(offTop.get_positions()))
        x = Vec3(xyz[0],0,0)
        y = Vec3(0,xyz[1],0)
        z = Vec3(0,0,xyz[2])
        topology = mod.getTopology()
        topology.setPeriodicBoxVectors([x, y, z])

        #Set up a Forcedield for the system, adding the small model parameters from GAFF
        forcefield = ForceField()
        for force in forces:
            forcefield.loadFile(force)
        forcefield.registerTemplateGenerator(gaff.generator)
        self._system = forcefield.createSystem(topology, nonbondedMethod=PME)
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.0004*picoseconds)
        #Create the simulation box that provides the environment for measurements along the trajectory
        self._simulation = Simulation(topology, self._system, integrator)


    def __getitem__(self, index):
        #Set the simulation to frame by [index]
        self._tu.trajectory[index]
        pos = array2vec3([atm.position for atm in self._tu.atoms])
        self._simulation.context.setPositions(pos)


    def __len__(self):
        #Define the length of the class as the length of the trajectory
        return len(self._tu.trajectory)
            

    def frames(self, select=[]):
        #To iterate over all, or a selection specified by slice, of the frames
        return ModelIterator(self._simulation, self._tu, select)

    def getPotentialEnergy(self, returnQuantity = False):
        #Return the potential energy of the current state of the simulation
        #Optionally as openMM Quantity
        state = self._simulation.context.getState(getEnergy=True)
        pot = state.getPotentialEnergy()
        if returnQuantity:
            return pot
        else:
            return pot.value_in_unit(kilojoule_per_mole)

    def minimizeEnergy(self, tolerance=Quantity(value=10, unit=kilojoule/mole), maxIterations=0):
        #wrapper for energy minimization in openmm simulation
        self._simulation.minimizeEnergy(tolerance = tolerance, maxIterations = maxIterations)

    def pdbReporter(self, repfile='output.pdb'):
        #if no reporter for this file exists, create a new one, then write current state to that file.
        state = self._simulation.context.getState(getPositions=True)
        if repfile not in self._reporters:
            self._reporters[repfile] = PDBReporter(repfile, 1)
        
        self._reporters[repfile].report(self._simulation, state)
            
    def constrainAtoms(self, selections):
        #Select atoms with MDAnalysis selection language and constrain them in the openmm simulation by removing mass.
        #Requires the similation to reinitialize to apply changes
        for select in selections:
            bbgroup = self._tu.select_atoms(select, sorted=False)
            bbIDs = [atm.ix for atm in bbgroup]
            for bbid in bbIDs:
                self._system.setParticleMass(bbid, 0*amu)
        
        self._simulation.context.reinitialize()
            

class ModelIterator:
    '''
    Iterator for the ModelledTrajectory class. Will iterate over the trajectory and, at each iteration, set the simulation coordinates to that frame. Takes a select as an optional argument, if provided will iterate over the frames specified in the order they are provided.
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


def array2vec3(positions):
    return [Vec3(r[0], r[1], r[2])*0.1*nanometer for r in positions]
