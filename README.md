# victory-rdna1

Our fork of [VIctory](https://github.com/Marvisak/VIctory), which lets an Xbox Series game run on a Windows PC. This one is
tuned for an **AMD RX 5700 XT**: an older card that has no mesh shaders and no barycentrics, which the game
expects to have.

**No game files are in here.** You need your own copy of the game. We're not affiliated with the game's developer or
publisher.

## Results on the two PCs we tested

Both run the Xbox Series S profile (`--console series-s`) with the [run settings](#run-settings) below. The console runs
this game at 30 fps, so it's playable, just slowly. Frame rates are measured in the world (more than 2,500 draws a frame),
after the shader cache has been built.

| | Our PC | Friend's PC |
|---|---|---|
| CPU | AMD Ryzen 5 3600 (6 cores / 12 threads) | AMD Ryzen 7 9800X3D (8 cores) |
| GPU | AMD Radeon RX 5700 XT (RDNA1), driver 32.0.21043.12001 | NVIDIA GeForce RTX 5070 Ti, driver 32.0.16.2124 |
| Mesh shaders / barycentrics | no / no (both worked around) | yes / yes |
| RAM | 32 GB (2 sticks), 3200 MT/s | 32 GB |
| OS | Windows 11 Pro (build 26200) | Windows 11 Pro (build 26300) |
| **Frame rate** | **6.6 fps median, 7.3 fps top 10%** (~141 ms a frame, run 103) | **8.6 fps average, up to 12-13 fps** in some areas (3,000-4,000 draws a frame) |
| Picture | full world | okay-ish |

Run 103's two speed-ups, `VICTORY_LOOKUP_CACHE=1` (158 → 147.5 ms) and `VICTORY_HEAP_SYNC_CUT=1` (147.5 → 141 ms),
took our PC from 158 to 141 ms a frame. On the friend's PC they raised the in-world average from 7.5 to 8.6 fps (+15%).

On a new install, the first run is **black for several minutes** while the shader pipelines compile. That's normal.
Leave it running, close it, and start it again.

## What we did

Things we fixed or worked around on the way, roughly in order:

- **Got a picture at all.** The game draws much of its geometry with mesh shaders, which RDNA1 doesn't have.
  Upstream runs those as compute passes. We fixed the parts that broke on AMD.
- **White screen from the upscaler.** The 5700 XT driver computes `Dot4AddI8Packed` wrong when an operand is a
  constant, so the FidelityFX upscaler output came out white. We worked around it in the shader translator.
- **Heads and hair missing.** Their pixel shaders need barycentrics, and RDNA1 refuses them. We emulate them with a
  geometry shader (`VICTORY_BARY_EMULATE`). An RT cap stays on with it.
- **"Kernel memory failure" in the world.** The pool of idle command lists ran dry. `VICTORY_IDLE_LISTS=32` fixes it.
- **Frame time from ~205 ms down to ~141 ms.** We profiled where the frame goes (card busy ~105 ms, the rest is our
  CPU-side translation and waiting) and cut it with a lookup cache and fewer heap syncs.
  We also tried lazy copy-back, card-side buffer copies and light refresh. None of them helped, so they're off.

Every change sits behind a `VICTORY_*` switch so it can be measured one run against another. We change several
switches per run, not one at a time.

## Building

```powershell
rustup default stable-x86_64-pc-windows-msvc
cargo build --release -p victory      # -> target\release\victory.exe
```

See `victory.example.toml` for the config and `docs/` for more. `docs/UPSTREAM-FORK-README.md` is the README of the
fork this one builds on.

## Run settings

The exe turns none of its speed switches on by itself. Set these environment variables before starting it:

```
VICTORY_TEXTURE_NO_WAIT=1  VICTORY_RELAY=2  VICTORY_JOIN_COPIES=1
VICTORY_ASYNC_SUBMIT=1  VICTORY_QUEUE2=1  VICTORY_INDIRECT_WINDOW=1  VICTORY_COPY_PLAIN=1
VICTORY_LEVELS_ON_DEMAND=1  VICTORY_MERGE_GAP=64  VICTORY_MERGE_RUNS=4096  VICTORY_STAGING_POOL=0
VICTORY_JOIN_PARTS=1  VICTORY_BACK_OWN_VIEW=1  VICTORY_INDIRECT_GATHER=1  VICTORY_CARD_REHASH_RUNS=1
VICTORY_CARD_BUFFERS_MIN_KB=64  VICTORY_CARD_BUFFERS_MB=1280  VICTORY_CARD_HASH_LEN=1
VICTORY_BACK_PRIVATE=1  VICTORY_ABSENT_KEEP=1  VICTORY_CARD_HASH_WIDE=1  VICTORY_CARD_CHANGED_LIST=1
VICTORY_HEAP_PASS_EVERY=1          # 4 blacked out the picture on an RTX 5070 Ti
VICTORY_LOOKUP_CACHE=1  VICTORY_HEAP_SYNC_CUT=1   # run 103
VICTORY_UPSCALER=plain  VICTORY_IDLE_LISTS=32
VICTORY_FRAME_LOG=1  VICTORY_LOG=warn,victory_gpu::host=info   # fps lines in the log
```

Then run: `victory.exe run <game folder> --execute --console series-s`

## Safety notes

- Block the network for `victory.exe` with a firewall rule before the first run. The game has its own crash
  uploader.
- The first start translates shaders as it needs them, so expect stutter. The cache makes the second run the one
  that counts.

## Credits

- **[VIctory](https://github.com/Marvisak/VIctory)** by Marvisak and its
  contributors is the base of everything here: the loader, the runtime
  services, the Direct3D 12 back end and the shader recompiler. Licensed
  GPL-3.0-or-later; so is this fork. This repo is one fresh commit; see
  upstream for the full history.
- **[XCL](https://github.com/ivall/xcl)** by ivall and its contributors
  (GPL-2.0-only) was used as a reference for how the same title behaves under
  another layer. Fixes inspired by it were written anew in VIctory's code.
- The guard over every view of a memory page is code by **myokex**
  ([victory-amd](https://github.com/myokex/victory-amd), commit `c6dc4d8`), which
  adapts pull request #115 here by Spexivus, its correction by joroyt and a part of
  VIctory #1122 by tightfall. The read-beside fix for kernel reads into title
  memory is by **Joshua Odom** (commit `10ea38e` of the same fork family). Both
  are carried here unchanged apart from one merge conflict.
- This fork's changes were developed with [Claude Code](https://claude.com/claude-code).

## License

GPL-3.0-or-later, same as VIctory. See `LICENSE`.
