W_ROOT := $(shell echo $$WEST_ROOT)
PYLANG_TARGET := $(shell readlink -f ../../shared/elasticnetwork-langevin)
PYQCP_TARGET := $(shell readlink -f ../../shared/pyqcprot)
CWD_TARGET := $(shell readlink -f .)

all: clean setup

setup: 
	@if [ -z "$${WEST_ROOT}" ]; then echo "The env variable WEST_ROOT must be specified"  && exit 1; fi
	mkdir data
	wget http://www.rcsb.org/pdb/files/1DC7.pdb -O data/1DC7.pdb
	wget http://www.rcsb.org/pdb/files/1DC8.pdb -O data/1DC8.pdb
	python bin/generate_data.py
	ln -s $(PYLANG_TARGET)/ElasticNetwork.so we_base/ElasticNetwork.so
	ln -s $(PYQCP_TARGET)/pyqcprot.so we_base/pyqcprot.so
	ln -s $(PYQCP_TARGET)/pyqcprot.so analysis/pyqcprot.so
	ln -s $(CWD_TARGET)/data/data.npz we_base/data.npz
	touch we_base/env.sh
	echo 'export WEST_ROOT=$(W_ROOT)' >> we_base/env.sh
	echo 'export WEST_PYTHONPATH=$${WEST_ROOT}/src:$${WEST_SIM_ROOT}' >> we_base/env.sh
	echo 'export WEST_PYTHON=python' >> we_base/env.sh

clean:
	rm -f we_base/env.sh we_base/*.so analysis/*.so we_base/data.npz
	rm -rf data

.PHONY: all clean setup 

