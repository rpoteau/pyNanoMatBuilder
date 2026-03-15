from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
from pathlib import Path
from .utils import hl, fg, bg

@dataclass

class pyNMBdata:
    clusters = ['regfccOh','regIco','regfccTd','regDD','fccCube','bccCube','fccCubo','fccTrOh','fccTrCube','bccrDD','fccdrDD','pbpy']

class pyNMBvar:
    dbFolder = 'resources/cif_database'
    _default_jmol = os.environ.get('JMOL_HOME')
    if _default_jmol and Path(_default_jmol).exists():
        path2Jmol = _default_jmol
    else:
        # Fallback to a placeholder string. 
        # Using a string instead of None prevents crashes in your Path() calls later.
        path2Jmol = "JMOL_NOT_FOUND"

class WulffShapes:
    data = ({
              'full name': {
                   'cube': 'cube',
                 'trcube': 'truncated cube',
                   'cubo': 'cuboctahedron',
                     'Oh': 'octahedron',
                   'trOh': 'truncated octahedron',
                     'Td': 'tetrahedron',
                 'dicoTd': 'deltoidal icositetrahedron',
                 'bccrDD': 'rhombic dodecahedron',
               'trbccrDD': 'truncated rhombic dodecahedron',
              'ttrbccrDD': 'tetratruncated rhombic dodecahedron',
                 'rhcubo': 'rhombicuboctahedron',
                 'hcpsph1': 'hcp sphere-like 6-fold symmetry',
                 'hcpsph2': 'hcp sphere-like 6-fold symmetry',
                'hcpwire': 'hcp nanowire along the c direction',
                 },
         'lattice system': {
                  'cube': 'cubic',
                'trcube': 'cubic',
                   'cubo': 'cubic',
                     'Oh': 'cubic',
                   'trOh': 'cubic',
                     'Td': 'cubic',
                 'dicoTd': 'cubic',
                 'bccrDD': 'cubic',
               'trbccrDD': 'cubic',
              'ttrbccrDD': 'cubic',
                 'rhcubo': 'cubic',
                 'hcpsph1': 'hexagonal',
                 'hcpsph2': 'hexagonal',
                'hcpwire': 'hexagonal',
                 },
        'Bravais lattice': {
                   'cube': 'bcc, fcc',
                 'trcube': 'fcc',
                   'cubo': 'fcc',
                     'Oh': 'fcc',
                   'trOh': 'fcc',
                     'Td': 'fcc',
                 'dicoTd': 'fcc',
                 'bccrDD': 'bcc',
               'trbccrDD': 'bcc',
              'ttrbccrDD': 'bcc',
                 'rhcubo': 'bcc',
                 'hcpsph1': 'hcp',
                 'hcpsph2': 'hcp',
                'hcpwire': 'hcp',
                 },
                 'planes': {
                   'cube': [[1,0,0]],
                 'trcube': [[1,0,0],[1,1,1]],
                   'cubo': [[1,0,0],[1,1,1]],
                     'Oh': [[1,1,1]],
                   'trOh': [[1,0,0],[1,1,1]],
                     'Td': [[1,1,1],[1,-1,-1],[-1,-1,1],[-1,1,-1]],    
                 'dicoTd': [[2,1,1]],
                 'bccrDD': [[1,1,0]],
               'trbccrDD': [[1,1,0],[1,1,1]],
              'ttrbccrDD': [[1,1,0],[0,0,1]],
                 'rhcubo': [[1,1,0],[0,0,1],[1,1,1]],
                 'hcpsph1': [[0,0,1],[1,0,0],[1,0,1]],
                 'hcpsph2': [[0,0,1],[1,0,0],[1,0,1],[2,0,1],[1,1,1],[2,1,0]],
                'hcpwire': [[1,0,0]],
                 },
         'apply symmetry': {
                   'cube': True,
                 'trcube': True,
                   'cubo': True,
                     'Oh': True,
                   'trOh': True,
                     'Td': False,
                 'dicoTd': True,
                 'bccrDD': True,
               'trbccrDD': True,
              'ttrbccrDD': True,
                 'rhcubo': True,
                 'hcpsph1': True,
                 'hcpsph2': True,
                'hcpwire': True,
                 },
      'relative energies': {
                   'cube': [1.0],
                 'trcube': [(1+np.sqrt(2))/2, np.sqrt((17+12*np.sqrt(2))/3)/2],
                   'cubo': [np.sqrt(2)/2, np.sqrt(6)/3],
                     'Oh': [1.0],
                   'trOh': [np.sqrt(2), np.sqrt(6)/2],
                     'Td': [1.0,1.0,1.0,1.0],
                 'dicoTd': [1.0],
                 'bccrDD': [1.0],
               'trbccrDD': [1.0,1.0],
              'ttrbccrDD': [1.0,1.0],
                 'rhcubo': [(1+np.sqrt(2))/2,(1+np.sqrt(2))/2,(np.sqrt((11+6*np.sqrt(2))/3))/2],
                 'hcpsph1': [34.6,39.9,40.9],
                 'hcpsph2': [2.76,3.07,3.15,3.51,3.39,3.41],
                'hcpwire': [1.0],
                 },
        
           'MOI for size': {
                       'cube': 'cube',
                     'trcube': 'cube',
                       'cubo': 'cube',
                         'Oh': 'Oh',
                       'trOh': 'sphere',
                         'Td': 'Td',
                     'dicoTd': 'dicoTd',
                     'bccrDD': 'bccrDD',
                   'trbccrDD': 'bccrDD',
                  'ttrbccrDD': 'bccrDD',
                     'rhcubo': 'sphere',
                     'hcpsph1': 'hcpsph',
                     'hcpsph2': 'hcpsph',
                    'hcpwire': 'wire',
                 },
                'comment': {
                   'cube': 'cutting length from the cube''s vertex c = 0',
                 'trcube': 'r3 = sqr((17+12sqr(2))/3)/2; r8 = (1+sqr(2))/3',
                   'cubo': 'cutting length from the cube''s vertex c = 0.5; r3 = sqr(6)/3; r4 = sqr(2)/2',
                     'Oh': 'cutting length from the cube''s vertex c = 1',
                   'trOh': 'r4 = sqr(2); r6 = sqr(6)/2',
                     'Td': None,
                 'dicoTd': '10.1063/1.4790368',
                 'bccrDD': None,
               'trbccrDD': None,
              'ttrbccrDD': 'chamfered cube or Goldberg polyhedron, r4=(3+4*sqr(3))/6, r6 = sqr(2)*(3+2*sqr(3))/6, 10.1039/c6dt00343e',
                 'rhcubo': 'yet another truncated rhombic dodecahedron = truncated cuboctahedral rhombus, r4 = (1+sqr(2))/2, r3 = (sqr((11+6*sqr(2))/3))/2, 10.1039/c6dt00343e',
                 'hcpsph1': '10.1039/c8cp06171h',
                 'hcpsph2': '10.1126/science.1106435',
                'hcpwire': None,
                 },
     })
    WSdf = pd.DataFrame(data)

class pyNMBcif:
    data = ({
        'cif file': {
            'NaCl': 'cod1000041-NaCl.cif',
            'TiO2 rutile': 'cod9015662-TiO2-rutile.cif',
            'TiO2 anatase': 'cod9015929-TiO2-anatase.cif',
            'Fe bcc': 'cod5000217-Fe_bcc.cif',
            'Mn alpha': 'cod9011068-Mn_alpha.cif',
            'Mn beta': 'cod1539039-Mn_beta.cif',
            'Co hcp': 'cod9008492-Co_hcp.cif',
            'Co fcc': 'cod9008466-Co_fcc.cif',
            'Co epsilon': 'cod9012884-Co_epsilon.cif',
            'Ru hcp': 'cod9008513-Ru_hcp.cif',
            'Pt fcc': 'cod9012957-Pt_fcc.cif',
            'Au fcc': 'cod9008463-Au_fcc.cif',
            'Fe beta': 'cod1539039-Fe_beta.cif',
            'Ag fcc': 'cod9008459-Ag_fcc.cif',
            'CsPbBr3 ortho' : 'CsPbBr3_ortho_14608.cif',
            'CsPbBr3 cubic' : 'CsPbBr3_cubic_231023.cif'
            }
            })
    CIFdf = pd.DataFrame(data)

class pyNMBimg:
    data = ({
        'png file': {
            'sphere': 'sphere-C.png',
            'ellipsoid': 'ellipsoid-C.png',
            'wire': 'underConstruction-C.png',
            'parallepiped': 'underConstruction-C.png',
            'Wulff': 'WS-C.png',
            'Wulff: cube': 'cubeWS-C.png',
            'Wulff: trcube': 'trcubeWS-C.png',
            'Wulff: cubo': 'cuboWS-C.png',
            'Wulff: Oh': 'OhWS-C.png',
            'Wulff: Oh': 'OhWS-C.png',
            'Wulff: trOh': 'trOhWS-C.png',
            'Wulff: dicoTd': 'dicoTdWS-C.png',
            'Wulff: bccrDD': 'bccrDDWS-C.png',
            'Wulff: trbccrDD': 'trbccrDDWS-C.png',
            'Wulff: ttrbccrDD': 'ttrbccrDDWS-C.png',
            'Wulff: rhcubo': 'rhcuboWS-C.png',
            'Wulff: hcpsph1': 'hcpsph1WS-C.png',
            'Wulff: hcpsph2': 'hcpsph2WS-C.png',
            'Wulff: hcpwire': 'underConstruction-C.png',
            }
            })
    IMGdf = pd.DataFrame(data)

