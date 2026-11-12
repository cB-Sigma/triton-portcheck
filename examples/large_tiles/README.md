# Large tiles: timeouts on a real compile

One `tl.dot` over a 256×256×256 fp32 tile. It could never run on a device (the launch-time shared-memory check
would reject it, which this tool cannot see), but it makes the compiler and assembler work hard, so it is a good way
to watch the per-cell timeout.

```bash
cd examples/large_tiles
triton-portcheck check --times                 # 15 s limit from the manifest
triton-portcheck check --times --timeout 120   # let every cell finish
```

With Triton 3.9.0 on a 4-core machine the 256 tile took about 39 s for `cuda:80`, 10 s for `cuda:90` and
`hip:gfx942`, and 3 s for `cuda:100`; the 32 tile takes well under a second everywhere. With the 15 s limit the
`cuda:80` cell is reported as `timeout` in the stage that was running, and the other cells are unaffected.
