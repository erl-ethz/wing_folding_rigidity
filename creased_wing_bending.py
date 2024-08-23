from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import itertools
import math
import numpy as np
from numpy import cos, sin, pi, sqrt, absolute, arcsin, arccos
import os
import sys
import scipy.io as sio
import traceback

# ==== Design Parameters ==== #
analysis_type = 'corrugated_wing_bending_force'
paper_thickness = 0.082		# [mm], paper thickness
wing_width = 39.			# [mm], wing width
wing_length = 200.			# [mm], wing length
fold_number = 5		# number of folds in the zigzag pattern
fold_angle = 30.	# [deg], fold angle
# crease_width = 0.	# [mm]
isotropic_elastic_modulus = 2.9e3	# [MPa], paper equivalent isotropic elastic modulus, https://doi.org/10.1016/j.ijsolstr.2017.05.028
poissons_ratio = 0.3	# Poisson's ratio of the paper, set to 0 for worst case assumed to be isotropic (poissons_ratio \simeq 0.3 [Szewczyk, W. (2008). Determination of Poisson’s ratio in the plane of the paper. Fibres & Textiles in Eastern Europe, 4, 117-120.])
paper = {'name': 'paper', 'density': 1.2e-6, 'elastic': (isotropic_elastic_modulus, poissons_ratio)}	# 

# ==== Simulation Parameters ==== #
mesh_element_size_fraction = 39.0
num_CPUs = 4	# number of CPUs to use for the analysis, set to 1 if parallel processing is not supported
nonlinear_geometry = False  # True for nonlinear geometry, False for linearized solution
initial_inc = 1.e-4
max_inc_nr = 1e3
min_inc_step = 2.23e-16
job_submission_policy = 'overwrite'	# 'prevent' or 'overwrite' or 'normal' to prevent, overwrite or allow job submission if the output database already exists


# ==== Functions ==== #
def create_wing_model():
	# geometry
	segment_length = wing_width / (fold_number + 1) 
	s = m.ConstrainedSketch(name='__profile__', sheetSize=200.0)
	unit_vect = np.array([cos(fold_angle / 2 / 180 * pi), sin(fold_angle / 2 / 180 * pi)])
	point2 = (0.0, 0.0)
	for i in range(fold_number + 1):
		point1 = point2
		point2 = point1 + segment_length * np.array([1,(-1)**i]) * unit_vect
		s.Line(point1=point1, point2=point2)
	p = m.Part(dimensionality=THREE_D, name='corrugated_wing', type=DEFORMABLE_BODY)
	p.BaseShellExtrude(depth=wing_length, sketch=s)
	del m.sketches['__profile__']

	# mesh part
	p.seedPart(size=wing_width / mesh_element_size_fraction, deviationFactor=0.1, minSizeFactor=0.1)  # global mesh size
	p.generateMesh()

	# material
	paper_abq = create_abaqus_material(paper)
	sect = m.HomogeneousShellSection(name='material_sect', material=paper_abq.name, thickness=paper_thickness)
	sect_ass = p.Set(name='sect_ass_set', faces=p.faces)
	p.SectionAssignment(region=sect_ass, sectionName=sect.name, offsetType=MIDDLE_SURFACE)
	# p.MaterialOrientation(axis=orientation_axis_variable, orientationType=GLOBAL, region=p.sets['section_assignment_set']) 	# rotate material orientation

	# sets
	edge_0 = p.Set(name='edge_0', edges=p.edges.getByBoundingBox(-0.1, -segment_length / 2, -0.1, 
																wing_width, segment_length / 2, 0.1))
	edge_L = p.Set(name='edge_L', edges=p.edges.getByBoundingBox(-0.1, -segment_length / 2, wing_length - 0.1, 
																wing_width, segment_length / 2, wing_length + 0.1))


	# Assembly
	a = m.rootAssembly
	p1 = a.Instance(name=p.name + '-1', part=p, dependent=ON)

	# Coupling RPs
	RP_0 = a.ReferencePoint(point=(wing_width * cos(fold_angle / 2 / 180 * pi) / 2, 0, 0))
	RP_L = a.ReferencePoint(point=(wing_width * cos(fold_angle / 2 / 180 * pi) / 2, 0, wing_length))
	a.features.changeKey(fromName='RP-1', toName='RP-0')
	a.features.changeKey(fromName='RP-2', toName='RP-L')

	C_0 = a.Set(name='C_0', referencePoints=(a.referencePoints[a.referencePoints.keys()[0]], ))  
	C_L = a.Set(name='C_L', referencePoints=(a.referencePoints[a.referencePoints.keys()[1]], ))  
	m.Coupling(name='cp_0', controlPoint=C_0, couplingType=KINEMATIC, influenceRadius=WHOLE_SURFACE, 	
				surface=p1.sets['edge_0'], u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
	m.Coupling(name='cp_L', controlPoint=C_L, couplingType=KINEMATIC, influenceRadius=WHOLE_SURFACE, 
				surface=p1.sets['edge_L'], u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
	
	# Analysis
	if nonlinear_geometry:
		nlgeom = ON
	else:
		nlgeom = OFF
	step = m.StaticStep(initialInc=initial_inc, maxNumInc=int(max_inc_nr), minInc=min_inc_step, 
					name=analysis_type, nlgeom=nlgeom, previous='Initial')  #, stabilizationMagnitude=1e-10, stabilizationMethod=DAMPING_FACTOR, continueDampingFactors=False)

	# Kinematic BCs
	m.EncastreBC(createStepName='Initial', name='BC-0', region=C_0)
	# BC_L = m.DisplacementBC(amplitude=UNSET, createStepName='Initial', name='BC-L', region=C_L, ur1=0.0, u1=0.0, ur3=0.0)
	# BC_L.setValuesInStep(stepName=step.name, ur1= - pi / 6.)
	BC_L = m.DisplacementBC(amplitude=UNSET, createStepName='Initial', name='BC-L', region=C_L, u1=0.0, u2=0.0, ur3=0.0)
	BC_L.setValuesInStep(stepName=step.name, u2 = - wing_length / 20.)

	# Modify Output Requests
	m.fieldOutputRequests['F-Output-1'].setValues(frequency=1)

def create_clean_abaqus_model(model_name):
	"""Create a clean Abaqus model to avoid mass and inertia issues. Deletes all other existing models"""
	if not('Model-1' in mdb.models):
		mdb.Model(name='Model-1', modelType=STANDARD_EXPLICIT)	
	for mod_key in mdb.models.keys():
		if mod_key != 'Model-1':
			del mdb.models[mod_key]
	if model_name in mdb.models:	# make sure you delete it
		del mdb.models[model_name]
	m = mdb.Model(name=model_name, modelType=STANDARD_EXPLICIT)
	del mdb.models['Model-1'] # To allow mass calculation, Model-1 needs to be deleted
	return m	


def create_abaqus_material(material):
	""" Creates a material in Abaqus with the given parameters."""
	if 'name' in material.keys():
		material_abaqus = m.Material(name=material['name'])
	else:	
		material_abaqus = m.Material(name='Material-1')
	
	if 'density' in material.keys():
		material_abaqus.Density(table=((material['density'],),))
	if 'elastic' in material.keys():
		if 'type' in material.keys():
			material_abaqus.Elastic(table=(material['elastic'],), type=material['type'])
		else:
			material_abaqus.Elastic(table=(material['elastic'],))
	if 'alpha' in material.keys():
		if material['alpha'] != 0.0:
			material_abaqus.Damping(alpha=material['alpha'])	
	return material_abaqus	


def job_submitter(job):
	if job_submission_policy == 'prevent':
		print("Job " + job_name + " was not submitted as user-defined submission policy is set to prevent.\n ")
	else: 
		print("Submitting job " + job_name + " ...")
		try: 
			job.submit(consistencyChecking=OFF)
			job.waitForCompletion()
			print("Simulation SUCCESSFUL for job " + job_name + "!!!\n ")
		except:
			print("WARNING: simulation UNSUCCESSFUL for job " + job_name + ". Check log data!\n ")


def job_handler():
	# creates a job and submits it only if requested in the settings
	if job_name in mdb.jobs:
		del mdb.jobs[job_name]		# delete it to make sure its re-created with the new parameters
	job = mdb.Job(name=job_name, model=m, type=ANALYSIS, numCpus=num_CPUs, numDomains=num_CPUs)
	# check whether output database related to this job exists:
	if os.path.isfile(os.getcwd() + '/' + model_name + '.odb'):
		if job_submission_policy == 'overwrite':
			print('ODB ' + job_name + '.odb already exists and overwrite request was given...')
			job_submitter(job)
		else:
			print('ODB ' + job_name + '.odb already exists and no overwrite request was given...')
			print('No job submission.\n ')
	else:
		job_submitter(job)
	return

'''
def postprocess():
	# Postprocess the results
	odb = session.openOdb(name=model_name + '.odb')
	session.viewports['Viewport: 1'].setValues(displayedObject=odb)
	session
'''

# ==== Main ==== #	
# creates a zig-zag folded wing model
if nonlinear_geometry:
	nlgeom = 'ON'
else:
	nlgeom = 'OFF'
model_name = analysis_type + '_fa' + str(int(fold_angle)) + '_n' + str(int(fold_number)) + '_t' + str(int(paper_thickness*1000)) + '_l' + str(int(wing_length)) + '_w' + str(int(wing_width)) + '_nl' + nlgeom
m = create_clean_abaqus_model(model_name)
create_wing_model()
job_name = model_name
job_handler()	# submit the job
'''
postprocess()	# postprocess the results
'''
