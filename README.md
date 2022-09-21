# white-matter-ephys
A set of scripts, functions, and notes for analyzing data from the HSW system from White Matter LLC.

Installation instructions:

```
git clone https://github.com/ralphpeterson/white-matter-ephys.git
cd white-matter-ephys
conda env create -f environment.yml
conda activate wme
pip install -e .
```

Note: If installing on Windows, it may be necessary to create a new system environment variable: `CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1`.

More info on this issue here: https://github.com/scipy/scipy/issues/14002

windows installation error: solved by 'conda install scipy' within wme environnment
