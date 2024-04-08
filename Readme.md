### Building the compiler:

The recommended LLVM commit to build the compiler: `da5a86b53e7d6e7ff7407b16c2c869894493ee99`.

To build the compiler, use:

```
mkdir build
cmake ../ -DCMAKE_PREFIX_PATH=${LLVM_ROOT}/lib/cmake/ -G Ninja
ninja
```

`${LLVM_ROOT}` is the path where LLVM is installed.

### Example:

Running:
```
./build/bin/xblang example.xb --load-extension=./build/lib/GPUPlugin.so --load-extension=./build/lib/OMPPlugin.so --dump  -par=nvptx --offload-llvm=false -cc=codegen
```
Should produce:
```
module {
  xb.func private @omp_get_thread_num() -> i32
  xb.func @main() {
    %0 = xb.constant(4 : i64) : i64
    %1 = arith.trunci %0 : i64 to i32
    %bsz = xb.var[local] @bsz : i32 -> !xb.ref<i32> [ = %1 : i32]
    %2 = xb.constant(2 : i64) : i64
    %3 = arith.trunci %2 : i64 to i32
    %gsz = xb.var[local] @gsz : i32 -> !xb.ref<i32> [ = %3 : i32]
    omp.parallel {
      %4 = xb.load %bsz : <i32> -> i32
      %bsz_0 = xb.var[local] @bsz : i32 -> !xb.ref<i32> [ = %4 : i32]
      %5 = xb.load %gsz : <i32> -> i32
      %gsz_1 = xb.var[local] @gsz : i32 -> !xb.ref<i32> [ = %5 : i32]
      xb.scope {
        %6 = xb.call @omp_get_thread_num() : () -> i32
        %ompId = xb.var[local] @ompId : i32 -> !xb.ref<i32> [ = %6 : i32]
        %7 = xb.load %bsz_0 : <i32> -> i32
        %8 = index.casts %7 : i32 to index
        %idx1 = index.constant 1
        %9 = xb.load %gsz_1 : <i32> -> i32
        %10 = index.casts %9 : i32 to index
        gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %10, %arg7 = %idx1, %arg8 = %idx1) threads(%arg3, %arg4, %arg5) in (%arg9 = %8, %arg10 = %idx1, %arg11 = %idx1) {
          xb.scope {
            %thread_id_x = gpu.thread_id  x
            %11 = index.castu %thread_id_x : index to i32
            %tid = xb.var[local] @tid : i32 -> !xb.ref<i32> [ = %11 : i32]
            %block_id_x = gpu.block_id  x
            %12 = index.castu %block_id_x : index to i32
            %bid = xb.var[local] @bid : i32 -> !xb.ref<i32> [ = %12 : i32]
            %13 = xb.load %ompId : <i32> -> i32
            %14 = xb.load %tid : <i32> -> i32
            %15 = xb.load %bid : <i32> -> i32
            gpu.printf "Host Thread ID: %d, Block ID: %d, Thread ID: %d\0A" %13, %15, %14 : i32, i32, i32
          }
          gpu.terminator
        }
      }
      omp.terminator
    }
    xb.return
  }
}

```
