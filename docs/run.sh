#! /bin/bash

#conda install -c conda-forge sphinx sphinx-argparse sphinx_rtd_theme sphinx-jsonschema
#pip install sphinxcontrib.blockdiag

cwd=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}

if [ $2 == "True" ];
then
    rm -rf _build
    rm -rf _autosummary
    make clean html
fi

if [ $1 == "True" ];
then
    if [ ! -f "_build/html/index.html"  ];
    then
        make clean html
    fi
    if [ -f "_build/html/index.html"  ];
    then
        open _build/html/index.html
    fi
fi

cd ${cwd}
