#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/ops.h"

namespace {

constexpr const char* kSelectedRoot =
#ifdef RUSTY_MLX_ROOT
    RUSTY_MLX_ROOT;
#else
    "/opt/homebrew/Cellar/mlx/0.30.0";
#endif
constexpr const char* kSelectedInclude =
#ifdef RUSTY_MLX_INCLUDE
    RUSTY_MLX_INCLUDE;
#else
    "/opt/homebrew/Cellar/mlx/0.30.0/include";
#endif
constexpr const char* kSelectedLib =
#ifdef RUSTY_MLX_LIB
    RUSTY_MLX_LIB;
#else
    "/opt/homebrew/Cellar/mlx/0.30.0/lib";
#endif
constexpr const char* kVariant =
#ifdef RUSTY_SMOKE_VARIANT
    RUSTY_SMOKE_VARIANT;
#else
    "default";
#endif

template <typename Fn>
void run_case(const char* name, Fn&& fn) {
  std::cout << "[case] " << name << "\n";
  try {
    auto arr = fn();
    std::cout << "  constructor: ok\n";
    try {
      arr.eval();
      std::cout << "  eval: ok\n";
    } catch (const std::exception& err) {
      std::cout << "  eval std::exception: " << err.what() << "\n";
      return;
    } catch (...) {
      std::cout << "  eval unknown exception\n";
      return;
    }

    try {
      auto item = arr.template item<float>();
      std::cout << "  scalar/item: ok " << item << "\n";
    } catch (const std::exception& err) {
      std::cout << "  scalar/item std::exception: " << err.what() << "\n";
    } catch (...) {
      std::cout << "  scalar/item unknown exception\n";
    }
  } catch (const std::exception& err) {
    std::cout << "  constructor std::exception: " << err.what() << "\n";
  } catch (...) {
    std::cout << "  constructor unknown exception\n";
  }
}

}  // namespace

int main() {
  std::cout << "variant=" << kVariant << "\n";
  std::cout << "compiler_command=provided by Makefile\n";
  std::cout << "cplusplus=" << __cplusplus << "\n";
  std::cout << "DYLD_LIBRARY_PATH="
            << (std::getenv("DYLD_LIBRARY_PATH")
                    ? std::getenv("DYLD_LIBRARY_PATH")
                    : "")
            << "\n";
  std::cout << "selected_root=" << kSelectedRoot << "\n";
  std::cout << "selected_include=" << kSelectedInclude << "\n";
  std::cout << "selected_lib=" << kSelectedLib << "\n";

  try {
    const auto& device = mlx::core::default_device();
    std::cout << "default_device_available="
              << (mlx::core::is_available(device) ? "true" : "false") << "\n";
  } catch (const std::exception& err) {
    std::cout << "default_device std::exception: " << err.what() << "\n";
  } catch (...) {
    std::cout << "default_device unknown exception\n";
  }

  run_case("scalar int", []() { return mlx::core::array(7); });
  run_case("scalar float", []() { return mlx::core::array(7.0f); });
  run_case("vector float",
           []() { return mlx::core::array({1.0f, 2.0f, 4.0f}); });
  run_case("zeros({1})", []() { return mlx::core::zeros({1}); });
  run_case("ones({1})", []() { return mlx::core::ones({1}); });

  return 0;
}
