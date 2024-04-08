MAKEFILE_PATH := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

all:


frmt:
	cd include; ../scripts/frmt.sh
	cd lib; ../scripts/frmt.sh
	cd utils; ../scripts/frmt.sh
	cd tools; ../scripts/frmt.sh

fmt:
ifneq ($(DIR),)
	cd $(DIR); $(MAKEFILE_PATH)scripts/frmt.sh
endif

push:
	./scripts/leia-sync.sh push

pull:
	./scripts/leia-sync.sh pull
