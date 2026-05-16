"""
Workload Analyzer — RISC-V HPC Kernel Library
Based on Berkeley motifs (Asanovic et al., 2006) + extended taxonomy

Usage:
    python3 analyzer.py                          # classifies built-in APPS list
    python3 analyzer.py --csv path/to/apps.csv  # use a local CSV instead
    python3 analyzer.py --sheet                 # fetch from Google Sheet

Output:
    workload_report.md   — human-readable report
    workload_report.json — machine-readable, consumed by benchmark pipeline
"""

import csv
import json
import urllib.request
import argparse
from collections import defaultdict

sheet_url = (
    "https://docs.google.com/spreadsheets/d/"
    "1PIx_yXE7Kot-qf8TM1O7lhHAu5tDW4ULmihXElloSsk"
    "/export?format=csv&gid=0"
)

#  Berkeley Motif + Extended Kernel Taxonomy 
# Each motif entry:
#   ops          : core operations (for the report)
#   opt_focus    : what to optimise (for the report)
#   riscv_notes  : RISC-V / RVV implementation notes
#   lib_triggers : library names that map to this motif (substring match, lowercase)
#   kw_triggers  : keywords in app name / description / category (substring match)
#   cat_triggers : exact spreadsheet category strings that map here

motifs = {
    #  1. Dense Linear Algebra 
    "Dense Linear Algebra (DLA)": {
        "ops": "GEMM, GEMV, LU, Cholesky, QR, Eigensolvers",
        "opt_focus": "Register tiling, RVV vectorisation, L1/L2 cache blocking, BLAS-3 kernel reuse",
        "riscv_notes": "Primary RVV target; VLEN-512 achieves ~90 % peak on GEMM. Use RISC-V V intrinsics for inner loop.",
        "lib_triggers": [
            "blas", "cblas", "openblas", "lapack", "lapacke",
            "eigen", "armadillo", "mkl", "atlas", "scalapack",
            "plasma", "magma", "elpa",
        ],
        "kw_triggers": [
            "linear algebra", "matrix multiply", "dense matrix",
            "lu factor", "cholesky", "qr decomposition",
            "least squares", "linear system", "eigensolv",
            "dft", "quantum chemistry", "ab initio", "plane wave",
            "hartree-fock", "coupled cluster", "molecular orbital",
            "drug discovery", "force field",
        ],
        "cat_triggers": [
            "Comp. Chemistry", "Mol. Dynamics", "Quantum",
            "Materials", "Finance", "Math/Computing",
        ],
    },

    #  2. Stencil / PDE 
    "Stencil / PDE": {
        "ops": "N-point stencil averaging, halo exchange, 3-D/4-D grid iteration, AMR",
        "opt_focus": "Cache blocking, prefetching, memory-bandwidth bound, data alignment, MPI halo",
        "riscv_notes": "Bandwidth-bound; use RVV gather for irregular stencils. Benefit from large VLEN for unit-stride sweeps.",
        "lib_triggers": [
            "petsc", "trilinos", "hypre", "openfoam",
            "chombo", "amrex", "deal.ii", "fenics", "dolfin",
            "firedrake", "libmesh", "moose",
        ],
        "kw_triggers": [
            "finite element", "finite difference", "finite volume",
            "pde", "partial differential", "stencil",
            "poisson", "heat equation", "diffusion",
            "fluid dynamics", "cfd", "navier-stokes",
            "weather", "climate", "multiphysics",
            "structural", "thermomechanical", "fea",
            "adaptive mesh", "amr", "lattice boltzmann",
            "computational fluid", "elasticity",
            "hydrodynamics", "compressible", "incompressible",
            "turbulent", "seismic", "wave propagation",
            "ocean model", "atmospheric", "ice sheet",
            "reservoir", "subsurface", "geoscience",
            "reverse time migration",
        ],
        "cat_triggers": [
            "PDE / FEA", "CFD", "Electromagnetism",
            "Climate", "Simulation", "Oil & Gas",
        ],
    },

    #  3. Spectral / FFT 
    "Spectral / FFT": {
        "ops": "FFT, butterfly networks, convolution, spectral differentiation",
        "opt_focus": "Cache reuse, bit-reversal, twiddle-factor caching, RVV butterfly",
        "riscv_notes": "Implement Cooley-Tukey with RVV; target radix-8 for VLEN≥512. Mixed-radix for non-power-of-2.",
        "lib_triggers": [
            "fftw", "fft", "fftpack", "cufft", "vkfft",
            "pocketfft", "kissfft",
        ],
        "kw_triggers": [
            "fourier", "spectral", "fft", "signal processing",
            "frequency domain", "acoustics", "seismic processing",
            "convolution", "filter bank", "dsp",
            "photonic", "band structure", "dielectric",
        ],
        "cat_triggers": [],
    },

    #  4. Sparse Linear Algebra / Graph 
    "Sparse / Graph": {
        "ops": "SpMV, SpGEMM, BFS/DFS, gather-scatter, CSR/CSC format",
        "opt_focus": "Irregular memory access, gather/scatter latency, CSR row-balance, load balancing",
        "riscv_notes": "RVV indexed gather for SpMV; exploit non-temporal stores for write-once rows. Software prefetch for CSR indices.",
        "lib_triggers": [
            "superlu", "umfpack", "pardiso",
            "suitesparse", "metis", "scotch", "boost graph",
            "graph500", "graphblas", "graphlab",
        ],
        "kw_triggers": [
            "sparse", "graph", "network analysis", "spmv",
            "sparse matrix", "unstructured mesh",
            "graph traversal", "breadth first", "depth first",
            "connectivity", "topology", "phylogen",
            "connected components", "population genetics",
            "rna-seq", "assembly graph", "bioinformatics",
        ],
        "cat_triggers": ["Bioinformatics"],
    },

    #  5. N-Body / Particle 
    "N-Body / Particle": {
        "ops": "All-pairs force, tree traversal, neighbour lists, Ewald summation, particle-in-cell",
        "opt_focus": "Spatial locality, neighbour-list caching, tree-code vectorisation, SIMD force kernel",
        "riscv_notes": "Inner force loop vectorises cleanly with RVV FP32/FP64. Use masked ops for pair-exclusion lists.",
        "lib_triggers": [
            "gromacs", "namd", "lammps", "amber", "openmm",
            "fmm", "barnes-hut",
        ],
        "kw_triggers": [
            "molecular dynamics", "n-body", "particle simulation",
            "force field", "gravitational", "coulomb",
            "barnes hut", "fast multipole", "lj potential",
            "monte carlo", "md simulation",
            "cosmological", "astrophysics", "nbody",
            "plasma", "particle-in-cell", "langevin",
            "protein folding", "biomolecular",
            "lattice qcd", "quark", "gluon", "fermion",
        ],
        "cat_triggers": ["Astrophysics"],
    },

    #  6. MapReduce / Data Parallel 
    "MapReduce / Data Parallel": {
        "ops": "Map, Shuffle, Reduce, Sort, Scan, ETL",
        "opt_focus": "Memory bandwidth, cache-oblivious layout, vectorised reduction",
        "riscv_notes": "RVV reductions (vredsum, vredmax) directly accelerate map-reduce inner loops.",
        "lib_triggers": [
            "hadoop", "spark", "dask", "ray",
            "blazingsql", "gpudb",
        ],
        "kw_triggers": [
            "mapreduce", "data parallel", "distributed",
            "big data", "etl", "pipeline",
            "sql engine", "analytics", "data integration",
            "oltp", "brokerage",
        ],
        "cat_triggers": [],
    },

    #  7. Element-wise / Reduction 
    "Element-wise / Reduction": {
        "ops": "Add, Mul, ReLU, Softmax, Norm, Sum, Dot product, Transcendentals",
        "opt_focus": "Kernel fusion, loop unrolling, RVV FP reductions, mixed precision",
        "riscv_notes": "Fuse activation + normalisation into single RVV pass. Use vfmacc for fused multiply-accumulate.",
        "lib_triggers": [
            "numpy", "scipy", "pandas",
            "thrust", "tbb", "openmp",
            "cupy", "numba",
        ],
        "kw_triggers": [
            "element-wise", "reduction", "vector add",
            "dot product", "norm", "relu", "softmax",
            "activation", "normalisation", "normalization",
            "data analytics", "statistics",
            "deep learning", "neural network", "machine learning",
            "transformer", "encoder", "bert", "pytorch",
            "tensorflow", "keras", "mxnet", "jax",
            "gradient boosting", "catboost",
            "video codec", "transcod", "streaming",
            "render", "game engine",
        ],
        "cat_triggers": ["ML / AI", "Data Science", "Media/Video", "Video Games"],
    },

    #  8. Structured / Unstructured Mesh 
    "Mesh / Geometry": {
        "ops": "Mesh generation, ray-tracing, surface normals, BVH traversal, CAD kernel",
        "opt_focus": "Memory layout for irregular access, BVH stack, SIMD triangle intersection",
        "riscv_notes": "BVH traversal benefits from RVV masked compare; triangle batch intersection with fp32 RVV.",
        "lib_triggers": [
            "vtk", "cgal", "opencascade", "netgen",
        ],
        "kw_triggers": [
            "mesh generat", "mesher", "cad", "geometry",
            "visualiz", "paraview", "visit", "mayavi",
            "ray trac", "bvh", "surface reconstruct",
            "3d mesh", "unstructured 3d", "splotch",
        ],
        "cat_triggers": ["Pre/Post CAD", "Visualization"],
    },

    #  9. General / Infrastructure 
    "General / Infrastructure": {
        "ops": "Workload management, profiling, build, embedded control, protocol stacks",
        "opt_focus": "Latency, I/O throughput, OS scheduling, lightweight runtime",
        "riscv_notes": "Bare-metal RISC-V runtime; CLIC interrupt controller for real-time embedded. Use Zicsr for perf counters.",
        "lib_triggers": [],
        "kw_triggers": [
            "workload manager", "scheduler", "build system",
            "profil", "benchmark", "top500", "green500",
            "motor controller", "can bus", "embedded",
            "chatbot", "pocket", "risc-v binary",
            "agent-based", "parallel programming framework",
        ],
        "cat_triggers": ["HPC Tools", "Benchmarking", "Embedded", "Multi-body"],
    },
}

APPS = [
    #  PDE / FEA 
    ("Elmer",            "PDE / FEA",        "FEA Software for Multiphysics Problems",                    "GPL",          "Debian"),
    ("Code-Aster",       "PDE / FEA",        "Structural and thermomechanical software",                  "GPL",          "Debian"),
    ("CalculiX",         "PDE / FEA",        "3D Structural Finite Element Program",                      "GPL",          "Debian"),
    ("FreeFEM",          "PDE / FEA",        "Finite element software family",                            "GPL",          "Debian"),
    ("Impact",           "PDE / FEA",        "Explicit dynamic finite element program",                   "GPL",          None),
    ("XmdS",             "PDE / FEA",        "Extensible multi-dimensional simulator",                    "GPL",          None),
    ("GetDP",            "PDE / FEA",        "Generalized environment for discrete problems",             "GPL",          None),
    ("FElt",             "PDE / FEA",        "Solid mechanics FEA",                                       "GPL",          None),
    ("OOFEM",            "PDE / FEA",        "General finite element program",                            "GPL",          None),
    ("TOCHNOG",          "PDE / FEA",        "Free finite element program",                               "GPL",          "Debian"),
    ("FEniCS",           "PDE / FEA",        "Automated ODE/PDE solver",                                  "GPL+LGPL",     "Debian"),
    ("SLFFEA",           "PDE / FEA",        "Structural FEA solver",                                     "GPL",          None),
    ("FELyX",            "PDE / FEA",        "General FEM toolbox",                                       "GPL",          None),
    ("ALBERTA",          "PDE / FEA",        "General FEM library",                                       "GPL3",         "Debian"),
    ("Chombo",           "PDE / FEA",        "PDE solver on block-structured AMR grids",                  "BSD",          "Source"),
    ("FiPy",             "PDE / FEA",        "Python-based finite volume PDE solver",                     "Public domain",None),
    ("Julian",           "PDE / FEA",        "BEM for Laplace and elastic mechanics",                     "GPL",          None),
    #  CFD 
    ("OpenFOAM",         "CFD",              "General CFD toolbox with pre-processor",                    "GPL",          "Debian"),
    ("OpenFlower",       "CFD",              "CFD solver for turbulent incompressible Navier-Stokes",     "GPL",          None),
    ("Gerris",           "CFD",              "Variable density NS solver with AMR",                       "GPL",          "Debian"),
    ("Code_Saturne",     "CFD",              "General purpose CFD software",                              "GPL",          "Debian"),
    ("DUNS",             "CFD",              "Diagonalized Upwind Navier Stokes Code",                    "GPL",          None),
    ("SLFCFD",           "CFD",              "San Le's Free CFD",                                         "GPL",          None),
    ("PETSc-FEM",        "CFD",              "Multi-physics FEM based on PETSc",                          "GPL",          None),
    ("TYPHON",           "CFD",              "Platform for gas dynamics methods",                         "GPL",          None),
    ("OpenFVM",          "CFD",              "Finite volume method solver",                               "GPL",          None),
    ("Incompact3d",      "CFD",              "High-order flow solver for incompressible DNS",             "CeCILL-C",     "Source"),
    ("PyFR",             "CFD",              "Fluid flow on unstructured grids",                          "BSD",          "Source"),
    #  Electromagnetism ─
    ("Meep",             "Electromagnetism", "FDTD simulation for EM systems",                            "GPL",          "Debian"),
    ("MIT Photonic Bands","Electromagnetism","Band structures of periodic dielectric structures",          "GPL",          "Debian"),
    #  Computational Chemistry 
    ("abinit",           "Comp. Chemistry",  "DFT for molecules and crystals",                            "GPL",          "Debian"),
    ("AMBER",            "Comp. Chemistry",  "Molecular dynamics for biomolecular sims",                  "Proprietary",  "Binary"),
    ("Folding@Home",     "Comp. Chemistry",  "Distributed protein folding simulation",                    "GPL",          "Client"),
    ("GALAMOST",         "Comp. Chemistry",  "GPU-accelerated soft matter MD",                            "GPL",          "Source"),
    ("GENESIS",          "Comp. Chemistry",  "Generalized-Ensemble MD",                                   "GPL",          "Source"),
    ("HALMD",            "Comp. Chemistry",  "Large-scale MD for liquids",                                "GPL",          "Source"),
    ("HOOMD-Blue",       "Comp. Chemistry",  "Particle dynamics toolkit for GPUs",                        "BSD",          "Source"),
    ("MELD",             "Comp. Chemistry",  "Integrative MD plugin for OpenMM",                          "GPL",          "Source"),
    ("MOE",              "Comp. Chemistry",  "Integrated drug discovery suite",                           "Proprietary",  "Binary"),
    ("myPresto",         "Comp. Chemistry",  "Computational drug discovery suite",                        "Open-source",  "Source"),
    ("OpenMM",           "Comp. Chemistry",  "Molecular dynamics on HPC+GPU",                             "MIT",          "Source"),
    ("PolyFTS",          "Comp. Chemistry",  "Polymer self-assembly simulations",                         "Open-source",  "Source"),
    ("SOP-GPU",          "Comp. Chemistry",  "Langevin dynamics for SOP model",                           "GPL",          "Source"),
    ("GAMESS",           "Comp. Chemistry",  "Ab initio quantum chemistry",                               "Custom",       "Source+Binary"),
    ("QUICK",            "Comp. Chemistry",  "GPU ab initio for HF and DFT",                              "GPL",          "Source"),
    ("RMG",              "Comp. Chemistry",  "Real-space DFT for extreme scalability",                    "GPL",          "Source"),
    ("TeraChem",         "Comp. Chemistry",  "GPU quantum chemistry software",                            "Proprietary",  "Binary"),
    ("VASP",             "Comp. Chemistry",  "Ab initio quantum MD with plane waves",                     "Proprietary",  "Binary"),
    ("ACES III",         "Comp. Chemistry",  "High-performance quantum chemistry",                        "Open-source",  "Source"),
    ("ACES 4",           "Comp. Chemistry",  "Quantum chemistry with GPU infrastructure",                 "Open-source",  "Source"),
    ("ADF",              "Comp. Chemistry",  "DFT electronic structure calculations",                     "Proprietary",  "Binary"),
    ("BigDFT",           "Comp. Chemistry",  "Wavelet-based DFT for materials",                           "GPL",          "Source"),
    ("BrianQC",          "Comp. Chemistry",  "GPU quantum chemistry library",                             "Proprietary",  "Binary"),
    ("DIRAC",            "Comp. Chemistry",  "Relativistic quantum chemistry",                            "LGPL",         "Source"),
    ("LSDalton",         "Comp. Chemistry",  "Linear-scaling quantum chemistry",                          "GPL",          "Source"),
    ("NWChemEx",         "Comp. Chemistry",  "Exascale quantum chemistry",                                "Eclipse",      "Source"),
    ("Q-Chem",           "Comp. Chemistry",  "High-performance quantum chemistry",                        "Proprietary",  "Binary"),
    ("QBOX",             "Comp. Chemistry",  "First-principles MD with plane-wave DFT",                   "GPL",          "Source"),
    ("Quantum ESPRESSO", "Comp. Chemistry",  "Electronic structure and materials modeling",               "GPL",          "Source"),
    #  Mol. Dynamics 
    ("LAMMPS",           "Mol. Dynamics",    "Classical MD for materials modeling",                       "GPL",          "Source+Binary"),
    ("GROMACS",          "Mol. Dynamics",    "MD for biomolecular systems",                               "GPL",          "Source+Container"),
    ("NAMD",             "Mol. Dynamics",    "Parallel molecular dynamics",                               "Free noncomm", "Binary+Source"),
    ("Tinker-HP",        "Mol. Dynamics",    "High-performance MD package",                               "Academic",     "Source"),
    #  Quantum 
    ("CP2K",             "Quantum",          "Molecular simulations for various systems",                 "GPL",          "Source"),
    ("NWChem",           "Quantum",          "Quantum chemistry for molecular and periodic systems",      "ECL",          "Source"),
    ("CPMD",             "Quantum",          "Car-Parrinello Molecular Dynamics",                         "Academic",     "Source"),
    ("QMCPACK",          "Quantum",          "Quantum Monte Carlo package",                               "UIUC",         "Source"),
    ("MILC",             "Quantum",          "MIMD Lattice QCD",                                          "GPL",          "Source"),
    ("Octopus",          "Quantum",          "Real-space TDDFT code",                                     "GPL",          "Source"),
    #  Climate 
    ("NEMO",             "Climate",          "Ocean modelling framework",                                  "CeCILL",       "Source"),
    ("CAM",              "Climate",          "Global atmospheric model",                                   "Open Source",  "Source"),
    ("CESM",             "Climate",          "Coupled global climate model",                               "Open Source",  "Source"),
    ("ICON",             "Climate",          "Icosahedral non-hydrostatic atm model",                      "Academic",     "Source"),
    ("Elmer/Ice",        "Climate",          "Ice sheet flow modeling",                                    "GPL",          "Source"),
    ("WRF",              "Climate",          "Weather Research and Forecasting model",                     "Custom",       "Source"),
    #  Materials 
    ("Quantum Espresso", "Materials",        "Electronic structure calculations",                          "GPL",          "Source"),
    ("GPAW",             "Materials",        "Grid-based projector-augmented wave method",                 "GPL",          "Source"),
    ("ADCIRC",           "Materials",        "Advanced Circulation Model for coastal hydrodynamics",       "LGPL",         "Source"),
    ("NEMO5",            "Materials",        "Nanoelectronics Modeling Tool",                              "Proprietary",  "Source"),
    #  Benchmarking 
    ("HPCC",             "Benchmarking",     "HPC Challenge benchmark suite",                             "BSD",          "Source+Binary"),
    ("Graph500",         "Benchmarking",     "Large-scale graph analysis benchmark",                      "Open Source",  "Source"),
    ("CloverLeaf",       "Benchmarking",     "Hydrodynamics mini-app",                                    "Public domain","Source"),
    ("top500",           "Benchmarking",     "Top500 supercomputer benchmark list",                       "Custom",       None),
    ("green500",         "Benchmarking",     "Green500 energy-efficiency ranking",                        "Custom",       None),
    ("TPC-C",            "Benchmarking",     "OLTP benchmark",                                            "Custom",       None),
    ("TPC-DI",           "Benchmarking",     "Data Integration ETL benchmark",                            "Custom",       None),
    ("TPC-E",            "Benchmarking",     "OLTP brokerage workload benchmark",                         "Custom",       None),
    #  Bioinformatics 
    ("RAxML",            "Bioinformatics",   "Phylogenetic tool for large datasets",                      "GPL",          "Source"),
    ("POY",              "Bioinformatics",   "Phylogenetic analysis software",                            "GPL-2.0",      "Source+Binary"),
    ("BORN",             "Bioinformatics",   "Bayesian inference of species networks",                    "MIT",          "Source"),
    ("MrBayes",          "Bioinformatics",   "Bayesian inference of phylogeny",                           "GPL",          "Source"),
    ("Trinity",          "Bioinformatics",   "RNA-Seq de novo assembly",                                  "BSD",          "Source"),
    ("ParConnect",       "Bioinformatics",   "Connected components in large graphs",                      "Open Source",  "Source"),
    ("POP",              "Bioinformatics",   "Population genetics software",                              "GPL",          "Source"),
    #  Simulation 
    ("Splotch",          "Visualization",    "Visualization for astrophysical datasets",                  "GPLv3",        "Source"),
    ("SST",              "Simulation",       "Structural Simulation Toolkit",                             "BSD",          "Source"),
    ("PFLOTRAN",         "Simulation",       "Subsurface flow simulation",                                "BSD",          "Source"),
    ("Cardioid",         "Simulation",       "Cardiac simulation tool",                                   "GPL",          "Source"),
    ("VPIC",             "Simulation",       "Vector Particle-in-Cell for plasma",                        "BSD",          "Source"),
    ("AWP-ODC",          "Simulation",       "Anelastic Wave Propagation for earthquake sim",             "GPL",          "Source"),
    ("BQCD",             "Simulation",       "Lattice QCD fermion solver",                                "GPL",          "Source"),
    ("CADISHI",          "Simulation",       "Euclidean distance histograms on GPU",                      "GPL",          "Source"),
    ("CASTRO",           "Simulation",       "Compressible astrophysical hydrodynamics",                  "BSD",          "Source"),
    ("Chemora",          "Simulation",       "Differential equations on GPU clusters",                    "GPL",          "Source"),
    ("Cholla",           "Simulation",       "Astrophysical hydrodynamics cosmology",                     "GPL",          "Source"),
    ("Chroma",           "Simulation",       "Lattice QCD quark-gluon calculations",                      "GPL",          "Source"),
    ("CPS",              "Simulation",       "Lattice QCD modular code",                                  "GPL",          "Source"),
    ("GAMER",            "Simulation",       "GPU-accelerated AMR for astrophysics",                      "MIT",          "Source"),
    ("GENE",             "Simulation",       "Gyrokinetic plasma simulator for fusion",                   "GPL",          "Source"),
    ("GPU-AH",           "Simulation",       "Astrophysics code for cosmic strings",                      "GPL",          "Source"),
    ("GPUwalls",         "Simulation",       "Cosmology domain wall simulation",                          "GPL",          "Source"),
    ("WPP",              "Simulation",       "Seismic wave propagation code",                             "LGPL",         "Source"),
    #  Astrophysics 
    ("FLASH",            "Astrophysics",     "Multi-physics astrophysical simulation",                    "Academic",     "Source"),
    ("GADGET",           "Astrophysics",     "Cosmological N-body simulation",                            "GPL",          "Source"),
    ("ChaNGa",           "Astrophysics",     "N-body cosmology simulator with Charm++",                   "Academic",     "Source"),
    ("Enzo",             "Astrophysics",     "AMR code for astrophysics",                                 "BSD",          "Source"),
    #  Machine Learning 
    ("BERT-AI",          "ML / AI",          "Bidirectional Encoder Representations from Transformers",   "Apache 2.0",   "Source"),
    ("GraphLab",         "ML / AI",          "Graph-based high-performance ML",                           "Apache 2.0",   "Source"),
    ("Horovod",          "ML / AI",          "Distributed deep learning training framework",              "Apache 2.0",   "Source"),
    ("Apache MXNet",     "ML / AI",          "Open-source deep learning framework",                       "Apache 2.0",   "Python+Container"),
    ("PyTorch",          "ML / AI",          "Deep learning framework for research and production",       "BSD",          "Python+Container"),
    ("TensorFlow",       "ML / AI",          "ML/DL framework for CPU/GPU computation",                   "Apache 2.0",   "Python+Container"),
    ("JAX",              "ML / AI",          "High-performance ML research with XLA",                     "Apache 2.0",   "Python"),
    ("Keras",            "ML / AI",          "Minimalist modular neural networks library",                "Open Source",  None),
    ("MatConvNet",       "ML / AI",          "CNNs for MATLAB",                                           "Custom",       None),
    ("MXNet",            "ML / AI",          "Deep learning for efficiency and flexibility",              "Custom",       None),
    ("Neon",             "ML / AI",          "Fast scalable Python DL framework",                         "Custom",       None),
    ("NVCaffe",          "ML / AI",          "Caffe deep learning framework",                             "Custom",       None),
    ("PaddlePaddle",     "ML / AI",          "Easy-to-use scalable DL platform",                          "Custom",       None),
    ("Theano",           "ML / AI",          "Symbolic expression compiler for large-scale ML",           "Custom",       None),
    ("Caffe2",           "ML / AI",          "Fast deep learning framework",                              "Custom",       None),
    ("CNTK",             "ML / AI",          "Microsoft toolkit for deep neural networks",                "Custom",       None),
    ("CatBoost",         "ML / AI",          "Gradient boosting with categorical features",               "Custom",       None),
    ("Chainer",          "ML / AI",          "Flexible intuitive DL framework",                           "Custom",       None),
    ("Deeplearning4j",   "ML / AI",          "Deep learning for the JVM",                                 "Custom",       None),
    ("Torch7",           "ML / AI",          "ML and computer vision IDE",                                "Open Source",  None),
    #  Pre/Post CAD 
    ("Salome",           "Pre/Post CAD",     "Graphical FEA pre/post with CAD",                           "LGPL",         "Debian"),
    ("Gmsh",             "Pre/Post CAD",     "Graphical FEA CAD tool and mesher",                         "GPL",          "Debian"),
    ("NETGEN",           "Pre/Post CAD",     "Automatic 2D/3D mesh generator",                            "LGPL",         "Debian"),
    ("enGrid",           "Pre/Post CAD",     "Automatic mesh generator",                                  "GPL",          "Debian RFP"),
    ("MeshLab",          "Pre/Post CAD",     "Processing of unstructured 3D meshes",                      "GPL",          "Debian"),
    ("Paraview",         "Visualization",    "Parallel visualization application",                        "several",      "Debian"),
    ("VisIt",            "Visualization",    "Parallel visualization tool",                               "BSD",          "Debian ITP"),
    ("MayaVi",           "Visualization",    "Data visualization based on VTK",                           "BSD",          "Debian"),
    ("BRL-CAD",          "Pre/Post CAD",     "CSG CAD system for US military",                            "GPL",          "Debian RFP"),
    ("QCad",             "Pre/Post CAD",     "2D general CAD using Qt",                                   "GPL",          "Debian"),
    #  Multi-body / Physics 
    ("MBDyn",            "Multi-body",       "Command-line multi-body dynamics",                          "GPL",          None),
    ("ORSA",             "Multi-body",       "Orbit Reconstruction Simulation and Analysis",              "GPL",          "Debian"),
    #  Math / Computing 
    ("OpenMC",           "Math/Computing",   "Monte Carlo particle transport",                            "MIT",          "Source"),
    ("MATLAB",           "Math/Computing",   "Mathematical computing environment",                        "Proprietary",  "Binary"),
    ("HPC Repast",       "Math/Computing",   "Agent-based modeling for HPC",                              "New BSD",      "Binary+Source"),
    ("Charm++",          "Math/Computing",   "Parallel programming framework",                            "BSD",          "Source"),
    ("ArrayFire",        "Math/Computing",   "GPU-accelerated computing library",                         "BSD",          "Library"),
    ("Eigen",            "Math/Computing",   "C++ template library for linear algebra",                   "MPL2",         "Header-only"),
    ("Julia",            "Math/Computing",   "High-level language for technical computing",               "MIT",          "Binary+Source"),
    ("Mathematica",      "Math/Computing",   "Symbolic technical computing",                              "Proprietary",  "Binary"),
    ("MAGMA",            "Math/Computing",   "Dense linear algebra for heterogeneous arch",               "Custom",       None),
    #  Oil & Gas 
    ("6X",               "Oil & Gas",        "Reservoir Simulation on Tesla",                             "Proprietary",  "Source"),
    ("Echelon",          "Oil & Gas",        "GPU-based reservoir simulator",                             "Proprietary",  "Standalone"),
    ("DecisionSpace",    "Oil & Gas",        "E&P platform for geoscience and drilling",                  "Proprietary",  "Suite"),
    ("tNavigator",       "Oil & Gas",        "Integrated reservoir simulation",                           "Proprietary",  "Standalone"),
    ("GeoDepth",         "Oil & Gas",        "Seismic Interpretation Suite",                             "Proprietary",  "Suite"),
    ("Geoteric",         "Oil & Gas",        "3D seismic interpretation",                                 "Proprietary",  "Standalone"),
    ("AxRTM",            "Oil & Gas",        "Reverse Time Migration Software",                           "Proprietary",  "SDK"),
    ("HUESpace",         "Oil & Gas",        "Seismic compression and imaging SDK",                       "Proprietary",  "SDK"),
    #  Finance 
    ("Adaptiv Analytics","Finance",          "Pricing and risk engine",                                   "Custom",       None),
    ("Oneview",          "Finance",          "Forward Monte Carlo for capital markets",                   "Custom",       None),
    ("SciFinance",       "Finance",          "Derivative pricing",                                        "Custom",       None),
    ("NAG",              "Finance",          "RNG, Brownian bridges, PDE solvers",                        "Custom",       None),
    #  Media/Video 
    ("Handbrake",        "Media/Video",      "Open-source video transcoder",                              "GPL",          None),
    ("OBS",              "Media/Video",      "Video recording and live streaming",                        "Open Source",  None),
    ("Daniel2",          "Media/Video",      "CUDA accelerated video codec",                              "Custom",       None),
    #  Video Games 
    ("Unreal Engine",    "Video Games",      "Unreal engine (Epic Games/Fortnite)",                       "Custom",       None),
    ("Carmack games",    "Video Games",      "Open-source id Software games (Doom, Quake)",               "GPL",          None),
    #  HPC Tools 
    ("SLURM",            "HPC Tools",        "Configurable open source workload manager",                 "Open Source",  None),
    ("CMake",            "HPC Tools",        "Cross-platform build system",                               "Custom",       None),
    ("ELPA",             "HPC Tools",        "Eigensolvers for symmetric matrices",                       "Custom",       None),
    ("HPCtoolkit",       "HPC Tools",        "Performance analysis for parallel programs",                "Custom",       None),
    ("PAPI",             "HPC Tools",        "Portable hardware counter interface",                       "Custom",       None),
    ("TAU",              "HPC Tools",        "Profiling toolkit for parallel programs",                   "Custom",       None),
    #  Data Science 
    ("CuPy",             "Data Science",     "GPU-accelerated NumPy-compatible library",                  "Custom",       None),
    ("Numba",            "Data Science",     "Python array compiler for GPU speed",                       "Custom",       None),
    ("BlazingSQL",       "Data Science",     "GPU-accelerated SQL engine",                                "Custom",       None),
    ("GPUdb",            "Data Science",     "Multi-GPU distributed object store with SQL",               "Custom",       None),
    ("H2O4GPU",          "Data Science",     "GPU-accelerated ML platform",                               "Custom",       None),
    #  Embedded / RISC-V 
    ("VESC",             "Embedded",         "Open-source motor controller / robotics",                   "Custom",       None),
    ("CAN bus",          "Embedded",         "Controller Area Network protocol",                          "Custom",       None),
    ("Whisplay",         "Embedded",         "Pocket-sized AI chatbot on RISC-V/RPi",                    "GPLv3",        "RISC-V binary"),
]

#  Classification

def classify(name: str, description: str, category: str) -> str:
    text = f"{name} {description} {category}".lower()
    cat  = category.strip()

    for motif, meta in motifs.items():
        # 1. exact category match
        if cat in meta["cat_triggers"]:
            return motif
        # 2. library substring
        if any(lib in text for lib in meta["lib_triggers"]):
            return motif
        # 3. keyword substring
        if any(kw in text for kw in meta["kw_triggers"]):
            return motif

    return "General / Infrastructure"

#  Data loading from CSV 

def load_rows_csv(csv_path: str | None) -> list[dict]:
    if csv_path:
        with open(csv_path, newline="", encoding="utf-8") as f:
            lines = f.read().splitlines()
    else:
        print("Fetching spreadsheet from Google Sheets…")
        with urllib.request.urlopen(sheet_url) as resp:
            lines = resp.read().decode("utf-8").splitlines()

    reader = csv.DictReader(lines)
    rows = []
    for row in reader:
        name = row.get("Name", "").strip()
        if not name:
            continue
        rows.append({
            "name":        name,
            "link":        row.get("Link", "").strip(),
            "description": row.get("Description", "").strip(),
            "author":      row.get("Author", "").strip(),
            "license":     row.get("License", "").strip(),
            "category":    row.get("", "").strip(),
        })
    return rows

def load_rows_builtin() -> list[dict]:
    rows = []
    for entry in APPS:
        name, category, description, license_, _ = entry
        rows.append({
            "name":        name,
            "link":        "",
            "description": description,
            "author":      "",
            "license":     license_,
            "category":    category,
        })
    return rows

#  Analysis 

def analyze(rows: list[dict]) -> dict:
    results = defaultdict(list)
    for row in rows:
        motif = classify(row["name"], row["description"], row["category"])
        row["motif"] = motif
        results[motif].append(row)
    return results

#  Markdown report 

REPORT_HEADER = """\
# Workload Analysis Report
*Generated by `wokload_analyser.py` — Berkeley Motif classification (extended)*

## Summary

| # | Motif | Apps | % of Total |
|--:|-------|-----:|-----------:|
"""

def write_markdown(results: dict, total: int, path: str = "workload_report.md"):
    lines = [REPORT_HEADER]

    sorted_motifs = sorted(results.items(), key=lambda x: -len(x[1]))
    for i, (motif, apps) in enumerate(sorted_motifs, 1):
        pct = 100 * len(apps) / total if total else 0
        lines.append(f"| {i} | {motif} | {len(apps)} | {pct:.1f}% |")

    lines.append("\n---\n")
    lines.append("## Kernel Motif Reference\n")
    lines.append(
        "| Kernel Motif | Underlying Operations | Optimisation Focus | RISC-V Notes |\n"
        "|---|---|---|---|\n"
    )
    for motif, meta in motifs.items():
        lines.append(
            f"| **{motif}** | {meta['ops']} | {meta['opt_focus']} | {meta['riscv_notes']} |"
        )

    lines.append("\n---\n")
    lines.append("## Per-Motif App Listing\n")

    for motif, apps in sorted_motifs:
        pct = 100 * len(apps) / total if total else 0
        lines.append(f"### {motif} — {len(apps)} apps ({pct:.1f}%)\n")
        if motif in motifs:
            m = motifs[motif]
            lines.append(f"- **Operations:** {m['ops']}")
            lines.append(f"- **Optimisation focus:** {m['opt_focus']}")
            lines.append(f"- **RISC-V notes:** {m['riscv_notes']}\n")
        lines.append("| App | Category | Description | License |")
        lines.append("|-----|----------|-------------|---------|")
        for app in sorted(apps, key=lambda a: a["name"]):
            desc = app["description"][:70] + ("…" if len(app["description"]) > 70 else "")
            link = app.get("link", "")
            name = f"[{app['name']}]({link})" if link else app["name"]
            lines.append(
                f"| {name} | {app['category']} | {desc} | {app['license']} |"
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown report -> {path}")

#  Terminal output 

def print_report(results: dict, total: int):
    print(f"\n{'='*70}")
    print(f"  WORKLOAD ANALYSIS — {total} apps — Berkeley Motif Classification")
    print(f"{'='*70}")
    for motif, apps in sorted(results.items(), key=lambda x: -len(x[1])):
        pct = 100 * len(apps) / total if total else 0
        print(f"\n[{motif}] — {len(apps)} apps ({pct:.1f}%)")
        if motif in motifs:
            print(f"  ops   : {motifs[motif]['ops']}")
            print(f"  focus : {motifs[motif]['opt_focus']}")
        for app in sorted(apps, key=lambda a: a["name"]):
            desc = app["description"][:55] + ("…" if len(app["description"]) > 55 else "")
            print(f"  • {app['name']:<28} [{app['category']:<20}] {desc}")
    print(f"\n{'='*70}\n")

#  Entry point 

def main():
    parser = argparse.ArgumentParser(
        description="Classify HPC/science apps by Berkeley Motif"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--csv",   help="Path to local CSV file")
    group.add_argument("--sheet", action="store_true",
                       help="Fetch from Google Sheet (default: use built-in APPS list)")
    args = parser.parse_args()

    if args.csv or args.sheet:
        rows = load_rows_csv(args.csv)       # args.csv is None when --sheet
    else:
        rows = load_rows_builtin()

    total = len(rows)
    print(f"Loaded {total} apps.")

    results = analyze(rows)
    print_report(results, total)

    write_markdown(results, total)

    with open("workload_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {motif: apps for motif, apps in results.items()},
            f, indent=2
        )
    print("JSON report      -> workload_report.json")

if __name__ == "__main__":
    main()