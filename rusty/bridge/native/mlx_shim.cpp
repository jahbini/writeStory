#include "mlx_shim.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"
#endif

namespace {

#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
struct TestArrayRecord {
  mlx::core::array value;
};

struct RuntimeProbeResult {
  bool primary_success = false;
  std::string primary_target = "default_device";
  std::string primary_failure_stage;
  std::string primary_exception;
  bool cpu_attempted = false;
  bool cpu_success = false;
  std::string cpu_failure_stage;
  std::string cpu_exception;
};

std::mutex& test_array_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::uint64_t, TestArrayRecord>& test_array_table() {
  static std::unordered_map<std::uint64_t, TestArrayRecord> table;
  return table;
}

std::uint64_t next_test_array_handle() {
  static std::uint64_t next = 1;
  return next++;
}

std::string json_escape(const std::string& input) {
  std::ostringstream out;
  for (char ch : input) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

mlx::core::array make_test_array() {
  return mlx::core::array(7.0f);
}

bool run_tiny_array_probe_once(bool use_cpu, RuntimeProbeResult& result) {
  std::string stage = "array construction";
  try {
    if (use_cpu) {
      mlx::core::set_default_device(
          mlx::core::Device(mlx::core::Device::cpu, 0));
    }

    auto handle = next_test_array_handle();
    {
      std::lock_guard<std::mutex> lock(test_array_mutex());
      test_array_table().emplace(handle, TestArrayRecord{make_test_array()});
    }

    stage = "eval";
    {
      std::lock_guard<std::mutex> lock(test_array_mutex());
      auto it = test_array_table().find(handle);
      if (it == test_array_table().end()) {
        throw std::runtime_error("test array handle disappeared before eval");
      }
      it->second.value.eval();
    }

    stage = "scalar extraction";
    {
      std::lock_guard<std::mutex> lock(test_array_mutex());
      auto it = test_array_table().find(handle);
      if (it == test_array_table().end()) {
        throw std::runtime_error(
            "test array handle disappeared before scalar extraction");
      }
      (void)it->second.value.item<float>();
    }

    stage = "device sync";
    mlx::core::synchronize();

    stage = "free handle";
    {
      std::lock_guard<std::mutex> lock(test_array_mutex());
      auto erased = test_array_table().erase(handle);
      if (erased != 1) {
        throw std::runtime_error("test array handle was not freed");
      }
    }

    return true;
  } catch (const std::exception& err) {
    if (use_cpu) {
      result.cpu_failure_stage = stage;
      result.cpu_exception = err.what();
    } else {
      result.primary_failure_stage = stage;
      result.primary_exception = err.what();
    }
    return false;
  } catch (...) {
    if (use_cpu) {
      result.cpu_failure_stage = stage;
      result.cpu_exception = "unknown exception";
    } else {
      result.primary_failure_stage = stage;
      result.primary_exception = "unknown exception";
    }
    return false;
  }
}

RuntimeProbeResult run_runtime_probe() {
  RuntimeProbeResult result;
  result.primary_success = run_tiny_array_probe_once(false, result);
  if (!result.primary_success) {
    result.cpu_attempted = true;
    result.cpu_success = run_tiny_array_probe_once(true, result);
  }
  return result;
}
#endif

}  // namespace

extern "C" {

const char* rusty_mlx_shim_version() {
  return "rusty-mlx-shim/0.1";
}

int rusty_mlx_shim_probe() {
  return 1;
}

int rusty_mlx_link_probe() {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    const auto& device = mlx::core::default_device();
    return mlx::core::is_available(device) ? 1 : 1;
  } catch (...) {
    return 0;
  }
#else
  return 0;
#endif
}

unsigned long long rusty_mlx_create_test_array() {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    std::lock_guard<std::mutex> lock(test_array_mutex());
    auto handle = next_test_array_handle();
    test_array_table().emplace(handle, TestArrayRecord{make_test_array()});
    return handle;
  } catch (...) {
    return 0;
  }
#else
  return 0;
#endif
}

double rusty_mlx_test_array_sum(unsigned long long handle) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    std::lock_guard<std::mutex> lock(test_array_mutex());
    auto it = test_array_table().find(handle);
    if (it == test_array_table().end()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(it->second.value.item<float>());
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
#else
  (void)handle;
  return std::numeric_limits<double>::quiet_NaN();
#endif
}

int rusty_mlx_free_test_array(unsigned long long handle) {
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  try {
    std::lock_guard<std::mutex> lock(test_array_mutex());
    auto erased = test_array_table().erase(handle);
    return erased == 1 ? 1 : 0;
  } catch (...) {
    return 0;
  }
#else
  (void)handle;
  return 0;
#endif
}

const char* rusty_mlx_runtime_diagnose_json() {
  static std::string output;
#if defined(RUSTY_MLX_HAVE_NATIVE_LINK)
  RuntimeProbeResult result = run_runtime_probe();
  std::ostringstream json;
  json << "{"
       << "\"primary_target\":\"" << json_escape(result.primary_target) << "\","
       << "\"primary_success\":" << (result.primary_success ? "true" : "false")
       << ","
       << "\"primary_failure_stage\":\""
       << json_escape(result.primary_failure_stage) << "\","
       << "\"primary_exception\":\"" << json_escape(result.primary_exception)
       << "\","
       << "\"cpu_attempted\":" << (result.cpu_attempted ? "true" : "false")
       << ","
       << "\"cpu_success\":" << (result.cpu_success ? "true" : "false") << ","
       << "\"cpu_failure_stage\":\"" << json_escape(result.cpu_failure_stage)
       << "\","
       << "\"cpu_exception\":\"" << json_escape(result.cpu_exception) << "\""
       << "}";
  output = json.str();
#else
  output =
      "{\"primary_target\":\"none\",\"primary_success\":false,"
      "\"primary_failure_stage\":\"array construction\","
      "\"primary_exception\":\"MLX native link unavailable\","
      "\"cpu_attempted\":false,\"cpu_success\":false,"
      "\"cpu_failure_stage\":\"\",\"cpu_exception\":\"\"}";
#endif
  return output.c_str();
}

}
