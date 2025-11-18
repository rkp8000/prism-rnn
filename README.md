Code for Pang, Arbelaiz, Pillow 2025. "Learning neural dynamics through instructive signals"

Tested on: Python 3.12.8, matplotlib 3.10.0, numpy 2.1.3, scipy 1.14.1, pandas 2.3.2, scikit-learn 1.6.1, jupyterlab 4.2.5 ipykernel 6.29.5, torch 2.6.0+cu124, and MATLAB version R2025a (for Fig 4). Note that no GPU or CUDA installation is required for our instructor training examples, hence a cpu-only installation of pytorch should probably work as well.

Install automatically via `pip install -r requirements.txt'. Install time can be quite fast for Python packages, although pytorch installation can occasionally take up to half an hour. Most scripts are quite fast and will run in a few seconds/minutes. Simulations involving parameter sweeps and loops over multiple trials can take up to a few hours. The error-tracking simulations (Fig 4) can also take several hours. Optimization notebooks (Instructor Training, BPTT, e-prop) can also take up to a few hours to run, but may complete faster depending on hardware.

Note that in the notebooks above, the variable "L" is sometimes used for the instructive signal (previously termed "learning signal"), rather than "I".


## Figure key

The notebook associated with each specific figure panel in the manuscript is given below. There are also several notebooks for auxiliary/supporting analyses, but which did not directly produce any figures in the manuscript. 

2a-d: Notebook 1A

2e-f: Notebook 2A

3b: Notebook 3A

3c-e: Notebook 3B

3f: Notebook 3C

3g-h: Notebook 3D

3i: Notebook 3E

3j: Notebook 3D

4b: Produced in MATLAB using code in `fig_4_code` directory.

5b-c: Notebook 5D

5d: Notebook 5E

5f-h: Notebook 5C

5i: Notebook 5C1

S1a-b: Notebook 0C

S1c, left and middle column: Notebook 2D

S1c, right column: Notebook 2E

S2a-b: Notebook 2C

S2c-d: Notebook 2B

S2e-f: Notebook 3D

S3a-b: Notebook 3D

S7: Producted in MATLAB using code in `fig_4_code` directory

