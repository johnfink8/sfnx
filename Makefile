# fnx build: CUDA 12.9, sm_70, gcc-14
#   make            the engine and the picture encoder (bin/sfnx, bin/sfnx-vision; llama.cpp's libraries first if missing)
#   make install    the systemd unit (serve/sfnx.service.in for this checkout and user, port 8080), enabled at boot;
#                   CONFIG=path runs that config instead of serve/sfnx.json (it may live outside the checkout)
#   make test-bin   the test programs (bin/fnx_run, bin/batch_test, bin/cache_test, bin/sample_test, bin/gguf_test,
#                   bin/topk_test, bin/sfnx_input_test, bin/batch_plan_test, bin/stage_test,
#                   bin/pf_ring_plan_test, bin/rope_test)
#   make ref        llama.cpp's reference tools (ref/fnx_ref, ref/fnx_logref, ref/img_logref, ref/mmid_bench)
#   make trace      the CUPTI kernel tracer for `tools/lab.sh trace` and the *_trace.sh scripts
#   LLAMA=path      llama.cpp at a37cc46 somewhere other than ./llama.cpp
.DEFAULT_GOAL := build
.PHONY: build install test-bin ref trace
ref: ref/fnx_ref ref/fnx_logref ref/img_logref ref/mmid_bench
trace: tools/trace/libcupti_trace.so
CUPTI := /usr/local/cuda/extras/CUPTI
tools/trace/libcupti_trace.so: tools/trace/cupti_trace.cpp
	g++-14 -O2 -std=c++17 -shared -fPIC -I$(CUPTI)/include -I/usr/local/cuda/include -o $@ $< \
	    -L$(CUPTI)/lib64 -Wl,-rpath,$(CUPTI)/lib64 -lcupti -lpthread
# no built-in rules: with them make remakes an included build/kernels_p0.d by `%: %.o` from build/kernels_p0.d.o, which
# the kernels_p% rule then compiles with FNX_KPART=0.d ("floating constant in preprocessor expression")
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:
build: bin/sfnx bin/sfnx-vision
test-bin: bin/fnx_run bin/batch_test bin/cache_test bin/sample_test bin/gguf_test bin/topk_test bin/sfnx_input_test bin/batch_plan_test \
          bin/stage_test bin/pf_ring_plan_test bin/rope_test
CONFIG ?= serve/sfnx.json
# The config is checked as the front end checks it before loading anything (front.check_config), so a config it would
# refuse stops the install rather than a boot; what needs the model (the context against its trained length) the engine
# refuses when it starts. Its path is made absolute but a link is kept a link (repointing it then
# switches configs at the next restart), and it may hold only what sed and systemd's ExecStart take as is. The unit is
# written whole to build/ first and moved into place only once nothing in it is left unfilled.
install: build
	cfg=$$(realpath -s "$(CONFIG)") && [ -f "$$cfg" ] || { echo "install: no config $(CONFIG)" >&2; exit 1; }; \
	for p in "$$cfg" "$(CURDIR)"; do case "$$p" in *[!A-Za-z0-9._/+-]*) \
	    echo "install: $$p has characters a unit's ExecStart cannot hold as is (letters, digits, . _ / + - only)" >&2; exit 1;; esac; done; \
	.venv/bin/python -m front.check_config "$$cfg" && \
	sed -e "s|@ROOT@|$(CURDIR)|g" -e "s|@CONFIG@|$$cfg|" -e "s|@USER@|$$(id -un)|" -e "s|@GROUP@|$$(id -gn)|" \
	    serve/sfnx.service.in > build/sfnx.service && \
	if grep -n '@[A-Z]*@' build/sfnx.service; then echo "install: build/sfnx.service has unfilled fields" >&2; exit 1; fi && \
	sudo install -m 644 build/sfnx.service /etc/systemd/system/sfnx.service
	sudo systemctl daemon-reload
	sudo systemctl enable sfnx
	@echo "install: the unit runs $(CONFIG); a running sfnx keeps its old config until sudo systemctl restart sfnx"

NVCC := /usr/local/cuda/bin/nvcc
# llama.cpp at a37cc46 (its ggml headers; its libraries for the picture encoder): a checkout in llama.cpp/, or LLAMA=path
LLAMA ?= $(CURDIR)/llama.cpp
GG := $(LLAMA)/ggml
NVFLAGS := -DFNX_DIR='"$(CURDIR)"' -O3 --use_fast_math -std=c++17 -arch=sm_86 -ccbin g++-14 -Wno-deprecated-gpu-targets -I$(GG)/src/ggml-cuda -I$(GG)/src -I$(GG)/include --expt-relaxed-constexpr
# the cards every layer is split across (src/model.hpp): 3 (prod), or 2, 4, 6 or 8 (make FNX_NCARD=4 ...; the counts
# the model's shapes divide by, src/card_split.hpp; above three untested on hardware). build/ncard holds the count the
# objects in build/ were made for; a different count rewrites it, which rebuilds them all. It is written while make
# reads this file, so any make in this checkout restamps it to its count, -n and `make ref` too
FNX_NCARD ?= 3
$(if $(filter 2 3 4 6 8,$(FNX_NCARD)),,$(error FNX_NCARD: 2, 3, 4, 6 or 8 (the 24 query heads and 48 value heads split evenly, a 32-value FF block on every card, at most 8 for CUDA's peers a card), not '$(FNX_NCARD)'))
NVFLAGS += -DFNX_NCARD=$(FNX_NCARD)
$(shell mkdir -p build && { [ "$$(cat build/ncard 2>/dev/null)" = "$(FNX_NCARD)" ] || echo $(FNX_NCARD) > build/ncard; })

bin/gguf_test: tools/gguf_test.cpp src/gguf.cpp src/gguf.hpp src/rope_gguf.hpp
	mkdir -p bin && g++-14 -O2 -std=c++17 -o $@ tools/gguf_test.cpp src/gguf.cpp
bin/stage_test: tools/stage_test.cpp src/gguf.cpp src/gguf.hpp
	mkdir -p bin && g++-14 -O2 -std=c++17 -o $@ tools/stage_test.cpp src/gguf.cpp -lpthread
bin/sfnx_input_test: tools/sfnx_input_test.cpp tools/sfnx_input.hpp
	mkdir -p bin && g++-14 -O2 -std=c++17 -Wall -pthread -o $@ tools/sfnx_input_test.cpp
# the batched drafts' row plan, on the CPU alone
bin/batch_plan_test: tools/batch_plan_test.cpp src/batch_plan.hpp
	mkdir -p bin && g++-14 -O2 -std=c++17 -Wall -o $@ tools/batch_plan_test.cpp
# the prompt path's ring plan against the two- and three-card sequences, and every supported count's sum, on the CPU
bin/pf_ring_plan_test: tools/pf_ring_plan_test.cpp src/pf_ring_plan.hpp src/card_split.hpp src/model.hpp src/gguf.hpp build/ncard
	mkdir -p bin && g++-14 -O2 -std=c++17 -Wall -DFNX_NCARD=$(FNX_NCARD) -o $@ tools/pf_ring_plan_test.cpp
# YaRN's band, angles and magnitude against ggml's, on the CPU alone
bin/rope_test: tools/rope_test.cpp src/rope.hpp
	mkdir -p bin && g++-14 -O2 -std=c++17 -Wall -o $@ tools/rope_test.cpp

# one object per source in build/ (headers tracked by nvcc's own dependency files), so a change recompiles only the
# objects it touches; every engine program and the lab's library link the same objects (position independent).
# kernels.cu compiles as KPARTS objects in parallel (make -j), each with its share of the launchers (FNX_KPART)
KPARTS := 0 1 2 3 4 5 6 7 8
KERN_OBJ := $(foreach p,$(KPARTS),build/kernels_p$(p).o)
ENG_OBJ := build/gguf.o build/model.o build/store.o $(KERN_OBJ) build/moe_mma.o build/engine.o build/prefill.o build/session.o build/batch.o
PIC := -Xcompiler -fPIC
build/kernels_p%.o: src/kernels.cu build/ncard
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $(PIC) -DFNX_KPART=$* -MD -MF build/kernels_p$*.d -c -o $@ $<
build/%.o: src/%.cu build/ncard
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $(PIC) -MD -MF build/$*.d -c -o $@ $<
build/%.o: src/%.cpp build/ncard
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $(PIC) -MD -MF build/$*.d -c -o $@ $<
build/tool_%.o: tools/%.cpp build/ncard
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $(PIC) -MD -MF build/tool_$*.d -c -o $@ $<
build/tool_%.o: tools/%.cu build/ncard
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $(PIC) -MD -MF build/tool_$*.d -c -o $@ $<
-include $(wildcard build/*.d)

bin/load_probe: build/tool_load_probe.o build/gguf.o build/model.o build/store.o
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -o $@ $^ -lcuda
bin/kbench bin/moe_mma_bench bin/topk_test: bin/%: build/tool_%.o $(KERN_OBJ) build/moe_mma.o
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -o $@ $^
bin/sfnx: build/tool_sfnx.o $(ENG_OBJ)
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -o $@ $^ -lcublas -lcuda -lpthread
bin/fnx_run bin/sample_test bin/cache_test bin/lanes_test bin/batch_test: bin/%: build/tool_%.o $(ENG_OBJ)
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -o $@ $^ -lcublas -lcuda

# the lab (tools/fnx_lab.cpp): bin/fnx_lab keeps the model loaded and runs fnx_run's commands from build/libfnx_lab.so,
# which holds the engine and is reloaded after a rebuild; both use the shared CUDA runtime, and the lab exports the
# model's code (gguf, model) to the library
LAB_OBJ := $(KERN_OBJ) build/moe_mma.o build/engine.o build/prefill.o build/session.o build/batch.o build/pic/tool_fnx_run.o
build/pic/tool_fnx_run.o: tools/fnx_run.cpp build/ncard
	@mkdir -p build/pic
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -DFNX_LAB_LIB -MD -MF build/pic/tool_fnx_run.d -c -o $@ $<
-include $(wildcard build/pic/*.d)
build/libfnx_lab.so: $(LAB_OBJ)
	$(NVCC) $(NVFLAGS) -shared -cudart shared -o $@.new $^ -lcublas -lcuda && mv -f $@.new $@
bin/fnx_lab: build/tool_fnx_lab.o build/gguf.o build/model.o build/store.o build/libfnx_lab.so
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -cudart shared -Xlinker -export-dynamic -o $@ build/tool_fnx_lab.o build/gguf.o build/model.o build/store.o -ldl -lcuda
.PHONY: lab
# bin/fnx_hold: the lab's holder (the weights in shared memory, src/store.hpp; tools/lab.sh starts it)
bin/fnx_hold: build/tool_fnx_hold.o build/gguf.o build/model.o build/store.o
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -cudart shared -o $@ $^ -lcuda
lab: bin/fnx_lab bin/fnx_hold build/libfnx_lab.so
# the loader sources' hash, built into the holder and the worker: the holder refuses a worker built from other ones
# (tools/lab_proto.hpp). A new hash names a new stamp file, which rebuilds both objects. A count other than 3 joins the
# hash (a two-card worker refuses a three-card holder); three cards' hash is the sources' alone, so prod's holder
# stamp follows only the loader's code
LOADER_SRC := src/model.cpp src/model.hpp src/card_split.hpp src/gguf.cpp src/gguf.hpp src/store.cpp src/store.hpp tools/lab_proto.hpp
$(if $(filter-out $(wildcard $(LOADER_SRC)),$(LOADER_SRC)),$(error loader sources missing: $(filter-out $(wildcard $(LOADER_SRC)),$(LOADER_SRC))))
LOADER_REV := $(shell { cat $(LOADER_SRC); [ $(FNX_NCARD) = 3 ] || echo ncard $(FNX_NCARD); } | sha1sum | cut -c1-16)
build/loader_rev.$(LOADER_REV):
	@mkdir -p build && touch $@
build/tool_fnx_hold.o build/tool_fnx_lab.o: build/loader_rev.$(LOADER_REV)
build/tool_fnx_hold.o build/tool_fnx_lab.o: NVFLAGS += -DFNX_LOADER_REV='"$(LOADER_REV)"'

# the image encoder (vision/): llama.cpp's mtmd, built there with CUDA for sm_70
LU := $(LLAMA)
bin/sfnx-vision: vision/sfnx_vision.cpp vision/*.h $(LU)/build/bin/libmtmd.so
	mkdir -p bin && g++-14 -O2 -std=c++17 -I$(LU)/include -I$(GG)/include -I$(LU)/tools/mtmd -o $@ vision/sfnx_vision.cpp \
	    -L$(LU)/build/bin -lmtmd -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,$(LU)/build/bin -lpthread

# the llama.cpp reference tools (ref/): logits and tensor dumps the exactness checks compare against
LLFLAGS = -O2 -std=c++17 -I$(LU)/include -I$(GG)/include -L$(LU)/build/bin -Wl,-rpath,$(LU)/build/bin
ref/fnx_ref ref/fnx_logref: ref/%: ref/%.cpp ref/logref_yarn.hpp src/env.hpp $(LU)/build/bin/libmtmd.so
	g++-14 $(LLFLAGS) -o $@ $< -lllama -lggml -lggml-base
ref/img_logref: ref/img_logref.cpp ref/logref_yarn.hpp src/env.hpp $(LU)/build/bin/libmtmd.so
	g++-14 $(LLFLAGS) -I$(LU)/tools/mtmd -o $@ $< -lmtmd -lllama -lggml -lggml-base
ref/mmid_bench: ref/mmid_bench.cpp $(LU)/build/bin/libmtmd.so
	g++-14 $(LLFLAGS) -o $@ $< -lggml -lggml-base -lggml-cuda

# llama.cpp's libraries, built in its own tree with CUDA for sm_70
$(LU)/build/bin/libmtmd.so:
	test -d $(LU)/.git || { echo "clone llama.cpp at a37cc46 into $(LU) first (or make LLAMA=path)" >&2; exit 1; }
	cmake -S $(LU) -B $(LU)/build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
	   
	cmake --build $(LU)/build --target mtmd llama
