all: emf
CFLAGS += -Wall
CFLAGS += -O2
LOADLIBES += -lm
emf: emf.o vec-mat.o
emf.o: emf.h vec-mat.h
vec-mat.o: vec-mat.h
