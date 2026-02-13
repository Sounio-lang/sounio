# Poseidon VM Build System

CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -pedantic -O2 -g
LDFLAGS =

SRC = main.c vm.c loader.c runtime.c
OBJ = $(SRC:.c=.o)
TARGET = poseidon

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)

test: $(TARGET)
	@echo "Running Poseidon VM tests..."
	@cd tests && ./run_tests.sh

# Dependencies
main.o: main.c loader.h vm.h runtime.h platform.h
vm.o: vm.c vm.h runtime.h opcodes.h platform.h
loader.o: loader.c loader.h vm.h platform.h
runtime.o: runtime.c runtime.h platform.h
