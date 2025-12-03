NVCC=nvcc

ifeq (run,$(firstword $(MAKECMDGOALS)))
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(RUN_ARGS):;@:)
endif

.PHONY: all clean run

all: raw shared

raw: raw.cu
	@mkdir -p out
	$(NVCC) $(CUDAFLAGS) $< -o out/$@

shared: shared.cu
	@mkdir -p out
	$(NVCC) $(CUDAFLAGS) $< -o out/$@

run:
	./out/raw $(RUN_ARGS)
	./out/shared $(RUN_ARGS)

clean:
	rm -rf out