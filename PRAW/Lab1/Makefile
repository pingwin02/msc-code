NVCC=nvcc
CUDAFLAGS=-O2

DIR ?= .
BIN_DIR = $(DIR)/out

ifeq (run,$(firstword $(MAKECMDGOALS)))
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(RUN_ARGS):;@:)
endif

SRCS := $(wildcard $(DIR)/*.cu)
BINS := $(patsubst $(DIR)/%.cu,$(BIN_DIR)/%,$(SRCS))

.PHONY: all clean run

all: $(BINS)

$(BIN_DIR)/%: $(DIR)/%.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CUDAFLAGS) $< -o $@

run: all
	@for prog in $(BINS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog $(RUN_ARGS); \
	done

clean:
	rm -rf $(BIN_DIR)