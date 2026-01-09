CC=nvcc
CFLAGS=-O2
OMPFLAGS=-Xcompiler -fopenmp
LIBS=

DIR ?= .
BIN_DIR = $(DIR)/out

ifeq (run,$(firstword $(MAKECMDGOALS)))
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(RUN_ARGS):;@:)
endif

SRCS := $(wildcard $(DIR)/*.c)
BINS := $(patsubst $(DIR)/%.c,$(BIN_DIR)/%,$(SRCS))

.PHONY: all clean run

all: $(BINS)

$(BIN_DIR)/%: $(DIR)/%.c $(DIR)/%.cu
	@mkdir -p $(BIN_DIR)
	$(CC) -c $(word 2,$^) -o $@.o
	$(CC) $(OMPFLAGS) $(CFLAGS) $< $@.o -o $@ $(LIBS)
	@rm $@.o

run: all
	@for prog in $(BINS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog $(RUN_ARGS); \
	done

clean:
	rm -rf $(BIN_DIR)