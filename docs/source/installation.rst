Installation
======================================

FoamGraph requires PyTorch and PyTorch Geometric. It's recommended to check the documentation of these packages for intallation instructions. The following is an example for conda installation::

    conda install pytorch torchvision cudatoolkit={CUDA} -c pytorch
    conda install pyg -c pyg

where `{CUDA}` should be replaced by the desired CUDA version.

Use the package manager pip to install FoamGraph::

    pip install .