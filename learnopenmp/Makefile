CC = gcc
SRC = $(wildcard *.c)
PROGS = $(patubst %.c,%,$(SRC))

all: $(PROGS)

pi: serial_pi.c
	$(CC) -o pi -std=c99 -fopenmp serial_pi.c 
%: %.c
	$(CC) $(CFLAGS) -o $@ $<
clean:
	rm -rf $(PROGS)
