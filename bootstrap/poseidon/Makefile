# Poseidon VM Build System

CC = gcc
AR = ar
CFLAGS = -std=c99 -Wall -Wextra -pedantic -O2 -g
LDFLAGS =

CORE_SRC = poseidon.c vm.c loader.c runtime.c
CORE_OBJ = $(CORE_SRC:.c=.o)
CLI_SRC = main.c
CLI_OBJ = $(CLI_SRC:.c=.o)
TEST_OBJ = tests/library_tests.o
TARGET = poseidon
LIB_TARGET = libposeidon.a
LIB_TEST_TARGET = poseidon_lib_tests

.PHONY: all clean test fixtures

all: $(LIB_TARGET) $(TARGET)

$(LIB_TARGET): $(CORE_OBJ)
	$(AR) rcs $@ $^

$(TARGET): $(CLI_OBJ) $(LIB_TARGET)
	$(CC) $(CFLAGS) -o $@ $(CLI_OBJ) $(LIB_TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

tests/%.o: tests/%.c
	$(CC) $(CFLAGS) -I. -c -o $@ $<

$(LIB_TEST_TARGET): $(TEST_OBJ) $(LIB_TARGET)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJ) $(LIB_TARGET) $(LDFLAGS)

fixtures:
	@cd tests && python3 gen_test_soir.py

clean:
	rm -f $(CORE_OBJ) $(CLI_OBJ) $(TEST_OBJ) $(TARGET) $(LIB_TARGET) $(LIB_TEST_TARGET)

test: $(TARGET) $(LIB_TEST_TARGET)
	@echo "Running Poseidon VM tests..."
	@cd tests && ./run_tests.sh

# Dependencies
main.o: main.c poseidon.h
poseidon.o: poseidon.c poseidon.h loader.h vm.h platform.h
vm.o: vm.c vm.h runtime.h opcodes.h platform.h
loader.o: loader.c loader.h vm.h platform.h
runtime.o: runtime.c runtime.h platform.h
tests/library_tests.o: tests/library_tests.c poseidon.h
