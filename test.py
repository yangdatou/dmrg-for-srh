import pickle, os, sys
import numpy as np
import pyscf
from pyscf import gto, scf, fci
from pyscf import mcscf, mrpt
from pyscf import lib

nproc = os.environ.get("SLURM_NPROCS", 1)
nproc = int(nproc)

from pyscf import dmrgscf
dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = 'mpirun -n %d --bind-to none' % nproc
max_m = 2000

mol = gto.Mole()
mol.verbose = 10
ene_nuc, h1, eri, ncipg, weightpg, Rpg, nao, NOccSO = pickle.load(open( "FC.p", "rb" ))
mol.nelectron = NOccSO
ovlp = np.eye(nao)
nso = 2 * nao

# RHF
hf_obj = scf.RHF(mol)
hf_obj.get_hcore = lambda *args: h1
hf_obj.get_ovlp = lambda *args: ovlp
hf_obj._eri = pyscf.ao2mo.restore(8, eri, nao)
hf_obj.kernel()
hf_obj.mo_coeff = np.eye(nao)

out = open("results.log","w")

cas_obj = mcscf.CASSCF(hf_obj,20,28)
cas_obj.fix_spin_(ss=0)
cas_obj.fcisolver = dmrgscf.DMRGCI(mol, maxM=max_m)
cas_obj.fcisolver.runtimeDir = lib.param.TMPDIR
cas_obj.fcisolver.scratchDirectory = lib.param.TMPDIR
cas_obj.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 1))
cas_obj.fcisolver.memory = int(mol.max_memory / 1000.0 / nproc)
cas_obj.canonicalization = True
cas_obj.natorb = True
cas_obj.kernel()
print("E(CAS)=",cas_obj.e_tot + ene_nuc)
out.write("E(CAS) = %1 12.8f\n" % (cas_obj.e_tot + ene_nuc))

# nevpt
pt_obj = mrpt.NEVPT(cas_obj)
pt_obj.kernel()
print("E(PT)=",pt_obj.e_corr + cas_obj.e_tot + ene_nuc)
out.write("E(PT)  = %1 12.8f\n" % (pt_obj.e_corr + cas_obj.e_tot + ene_nuc))
