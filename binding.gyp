{
  "targets": [
    {
      "target_name": "metal_llm",
      "sources": [
        "metal/metal_llm.cpp",
        "metal/metal_llm_node.cpp",
        "metal/direct_metal_probe.mm"
      ],
      "product_dir": "../metal",
      "cflags_cc": [
        "-std=c++17",
        "-fexceptions",
	"-fcxx-exceptions"
      ],
      "include_dirs": [
         "<!(node -p \"require('node-addon-api').include\")",
         "<!(node -p \"require('node-addon-api').include_dir\")",
         "<!(node -e \"var fs=require('fs');process.stdout.write(fs.existsSync('mlx/build/libmlx.a')?'mlx':'/opt/homebrew/include')\")",
         "./metal"
       ],
       "dependencies": [
         "<!(node -p \"require('node-addon-api').gyp\")"
       ],
      "libraries": [
        "<!@(node -e \"var fs=require('fs'),p=require('path');var a=p.resolve('mlx/build/libmlx.a');if(fs.existsSync(a))console.log(a);else{console.log('-L/opt/homebrew/lib');console.log('-lmlx');}\")",
        "-framework Metal",
        "-framework Foundation",
        "-framework Accelerate"
      ],
      "xcode_settings": {
	      "MACOSX_DEPLOYMENT_TARGET": "15.0",
	      "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
	      "CLANG_CXX_ENABLE_EXCEPTIONS": "YES",
	      "CLANG_CXX_LIBRARY": "libc++"
      },
      "defines": [
        "NAPI_CPP_EXCEPTIONS=1"
      ]
    }
  ]
}
