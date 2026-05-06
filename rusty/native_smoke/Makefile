ROOT ?= /opt/homebrew/Cellar/mlx/0.30.0
INCLUDE ?= $(ROOT)/include
LIB ?= $(ROOT)/lib
CXX ?= /usr/bin/clang++
SRC ?= mlx_smoke.cpp
PYTHON ?= /Users/jahbini/writeStory/.venv/bin/python

BIN ?= mlx_smoke
STD ?= -std=c++17
EXTRA_CXXFLAGS ?=
EXTRA_LDFLAGS ?=
VARIANT ?= default

LOCAL_ROOT := /Users/jahbini/development/mlx
LOCAL_INCLUDE := $(LOCAL_ROOT)/mlx
LOCAL_LIB_CANDIDATES := $(wildcard $(LOCAL_ROOT)/build/libmlx.dylib) \
	$(wildcard $(LOCAL_ROOT)/build/lib/libmlx.dylib) \
	$(wildcard $(LOCAL_ROOT)/build/*/libmlx.dylib)
LOCAL_LIB := $(strip $(firstword $(LOCAL_LIB_CANDIDATES)))

.PHONY: all build run clean otool matrix python brew-info pkg-config-info cmake-info \
	build-cxx17 build-cxx20 build-libcpp build-opt build-local \
	run-cxx17 run-cxx20 run-libcpp run-opt run-local

all: build

build:
	@echo $(CXX) $(STD) $(EXTRA_CXXFLAGS) -I$(INCLUDE) -DRUSTY_MLX_ROOT=\"$(ROOT)\" -DRUSTY_MLX_INCLUDE=\"$(INCLUDE)\" -DRUSTY_MLX_LIB=\"$(LIB)\" -DRUSTY_SMOKE_VARIANT=\"$(VARIANT)\" $(SRC) -L$(LIB) $(EXTRA_LDFLAGS) -lmlx -o $(BIN)
	@$(CXX) $(STD) $(EXTRA_CXXFLAGS) -I$(INCLUDE) \
		-DRUSTY_MLX_ROOT=\"$(ROOT)\" \
		-DRUSTY_MLX_INCLUDE=\"$(INCLUDE)\" \
		-DRUSTY_MLX_LIB=\"$(LIB)\" \
		-DRUSTY_SMOKE_VARIANT=\"$(VARIANT)\" \
		$(SRC) -L$(LIB) $(EXTRA_LDFLAGS) -lmlx -o $(BIN)

run: build
	@./$(BIN)

otool: build
	@otool -L $(BIN)

matrix:
	@$(MAKE) --no-print-directory run-cxx17
	@$(MAKE) --no-print-directory run-cxx20
	@$(MAKE) --no-print-directory run-libcpp
	@$(MAKE) --no-print-directory run-opt
	@$(MAKE) --no-print-directory run-local
	@$(MAKE) --no-print-directory python
	@$(MAKE) --no-print-directory brew-info
	@$(MAKE) --no-print-directory pkg-config-info
	@$(MAKE) --no-print-directory cmake-info

build-cxx17:
	@$(MAKE) --no-print-directory build STD=-std=c++17 VARIANT=cxx17_cellar

run-cxx17: build-cxx17
	@./$(BIN)

build-cxx20:
	@$(MAKE) --no-print-directory build STD=-std=c++20 VARIANT=cxx20_cellar

run-cxx20: build-cxx20
	@./$(BIN)

build-libcpp:
	@$(MAKE) --no-print-directory build STD=-std=c++17 EXTRA_CXXFLAGS='-stdlib=libc++' VARIANT=cxx17_libcpp_cellar

run-libcpp: build-libcpp
	@./$(BIN)

build-opt:
	@$(MAKE) --no-print-directory build ROOT=/opt/homebrew/opt/mlx INCLUDE=/opt/homebrew/opt/mlx/include LIB=/opt/homebrew/opt/mlx/lib VARIANT=cxx17_opt

run-opt: build-opt
	@./$(BIN)

build-local:
	@if [ -n "$(LOCAL_LIB)" ]; then \
		$(MAKE) --no-print-directory build ROOT=$(LOCAL_ROOT) INCLUDE=$(LOCAL_INCLUDE) LIB="$(LOCAL_LIB)" VARIANT=cxx17_local; \
	else \
		echo "[native_smoke] local repo build skipped: no libmlx.dylib under $(LOCAL_ROOT)/build"; \
	fi

run-local: build-local
	@if [ -n "$(LOCAL_LIB)" ]; then \
		./$(BIN); \
	else \
		true; \
	fi

python:
	@$(PYTHON) -c 'import subprocess, sys; cmd=[sys.executable, "-c", "import mlx.core as mx; print(mx.array([1,2,3]))"]; print("python_command=", " ".join(cmd)); proc=subprocess.run(cmd, capture_output=True, text=True); print("returncode=", proc.returncode); print("--- stdout ---") if proc.stdout else None; print(proc.stdout, end="") if proc.stdout else None; print("--- stderr ---") if proc.stderr else None; print(proc.stderr, end="") if proc.stderr else None'

brew-info:
	@brew info mlx || true

pkg-config-info:
	@pkg-config --modversion mlx || true

cmake-info:
	@find /opt/homebrew/Cellar/mlx/0.30.0 /Users/jahbini/development/mlx .venv/lib/python3.14/site-packages/mlx -path '*/MLXConfig.cmake' -o -path '*/mlx.metallib' -o -path '*/libmlx.dylib' | sort || true

clean:
	@rm -f $(BIN)
