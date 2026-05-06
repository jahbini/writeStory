#pragma once

extern "C" {
const char* rusty_mlx_shim_version();
int rusty_mlx_shim_probe();
int rusty_mlx_link_probe();
unsigned long long rusty_mlx_create_test_array();
double rusty_mlx_test_array_sum(unsigned long long handle);
int rusty_mlx_free_test_array(unsigned long long handle);
const char* rusty_mlx_runtime_diagnose_json();
}
