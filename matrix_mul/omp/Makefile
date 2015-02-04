PROJ = matrix_mul
CC = g++

CFLAGS = -c -Wall -I/opt/local/include -I$(HOME)/cppunit/include
LDFLAGS = -L/opt/local/lib -L$(HOME)/cppunit/lib
LIBS = -lcppunit -ldl
OBJS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))
OMPFLAGS = -fopenmp
SIMDFLAGS = -msse2 -msse3 -O3 -lm

all: $(PROJ)

$(PROJ): $(OBJS)
	$(CC) $(OMPFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS) -O3

%.o : %.cpp
	$(CC) $(OMPFLAGS) $(CFLAGS) $< -o $@ $(SIMDFLAGS) -O3

%.o : %.cpp %.h
	$(CC) $(OMPFLAGS) $(CFLAGS) $< -o $@ $(SIMDFLAGS) -O3

clean:
	rm -f $(PROJ) $(OBJS) *.xml
