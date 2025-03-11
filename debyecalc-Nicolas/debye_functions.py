def calcSofQ(xyzfile:str,qmin:float=1,qmax:float=25,qstep:float=0.01,biso:float=0.1):
    from debyecalculator import DebyeCalculator
    import torch
    
    """
    xyz file: path to file produced by pyNMP- as this file contains 5 columns, it should be cleaned using the clean_xyz function
    qmin: min q value on which the Debye equation is computed
    qmax: max q value on which the Debye equation is computed
    qstep: step between successive q points
    biso: term for isotropic atomic displacement parameter
    """
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'  
    calc = DebyeCalculator(qmin=qmin,qmax=qmax,qstep=qstep,device=device,biso=biso)
    strufile=clean_xyz(xyzfile)
    Q, I =calc.iq(structure_source=strufile)
    os.system('rm %s'%strufile)
    return Q,I
    

def calcGofR(xyzfile:str,rmin:float=0,rmax:float=50,rstep:float=0.01,biso:float=0.1,qdamp:float=0):
    from debyecalculator import DebyeCalculator
    import torch
    
    """
    xyz file: path to file produced by pyNMP- as this file contains 5 columns, it should be cleaned using the clean_xyz function
    qmin: min q value on which the Debye equation is computed
    qmax: max q value on which the Debye equation is computed
    qstep: step between successive q points
    biso: term for isotropic atomic displacement parameter
    qdamp: parameter to accune for PDF damping linked to truncation 
    """
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'  
    calc = DebyeCalculator(rmin=rmin,rmax=rmax,rstep=rstep,device=device,biso=biso,qdamp=qdamp)
    strufile=clean_xyz(xyzfile)
    r, G =calc.gr(structure_source=strufile)
    os.system('rm %s'%strufile)
    return r,G 
