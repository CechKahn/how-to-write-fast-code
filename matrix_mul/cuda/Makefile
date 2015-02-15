PROJ = matrix_mul
CC = g++
NVCC = nvcc

CFLAGS = -c -g -Wall -I/opt/local/include -I$(HOME)/cppunit/include -O3
LDFLAGS = -L/opt/local/lib -L$(HOME)/cppunit/lib
LIBS = -lcppunit -ldl
OBJS = matrix_mul.o tests.o

all: $(PROJ)

$(PROJ): $(OBJS) $(TEST_CASES)
	$(NVCC) $(LDFLAGS) $^ -o $@ $(LIBS) -O3

%.o : %.cu %.h
	$(NVCC) -c $< -o $@

%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@ 

clean:
	rm -f $(PROJ) $(OBJS) $(TEST_CASES) *.xml
