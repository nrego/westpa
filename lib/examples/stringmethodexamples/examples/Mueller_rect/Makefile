W_ROOT := $(shell echo $$WEST_ROOT)
PYLANG_TARGET := $(shell readlink -f ../../shared/pylangevin-integrator)
SRC_TARGET := $(shell readlink -f src)

all: clean setup

setup: 
	@if [ -z "$${WEST_ROOT}" ]; then echo "The env variable WEST_ROOT must be specified"  && exit 1; fi
	cd src && python setup.py build_ext --inplace
	ln -s $(PYLANG_TARGET)/ForceFields.so we_base/ForceFields.so
	ln -s $(PYLANG_TARGET)/cIntegratorSimple.so we_base/cIntegratorSimple.so
	ln -s $(SRC_TARGET)/utils.so we_base/utils.so
	ln -s $(SRC_TARGET)/utils.so analysis/utils.so
	touch we_base/env.sh
	echo 'export WEST_ROOT=$(W_ROOT)' >> we_base/env.sh
	echo 'export WEST_PYTHONPATH=$${WEST_ROOT}/src:$${WEST_SIM_ROOT}' >> we_base/env.sh
	echo 'export WEST_PYTHON=python' >> we_base/env.sh
	echo 'export WM_WORK_MANAGER=serial' >> we_base/env.sh

clean:
	rm -f we_base/env.sh we_base/*.so analysis/*.so

.PHONY: all clean setup 

