OBJ=emf.o vec-mat.o

all: emf

CFLAGS += -Wall
CFLAGS += -O2
LOADLIBES += -lm

emf: $(OBJ)

emf.o: emf.h vec-mat.h
vec-mat.o: vec-mat.h

.PHONY: clean
clean:
	rm -f emf $(OBJ)
