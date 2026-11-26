# victory-rdna1

Our fork of [VIctory](https://github.com/Marvisak/VIctory), which lets an Xbox Series game run on a Windows PC. This one is
tuned for an **AMD RX 5700 XT**: an older card that has no mesh shaders and no barycentrics, which the game
expects to have.

**No game files are in here.** You need your own copy of the game. We're not affiliated with the game's developer or
publisher.

## Our best run: run 103

| | |
|---|---|
| Frame rate in the world | **6.6 fps median, 7.3 fps at the top 10%**, ~141 ms a frame (406 samples) |
| Console profile | Xbox Series S (`-Console series-s`). Series X gives 1-2 fps on this card |
| Settings that made it | `VICTORY_LOOKUP_CACHE=1` (147.5 vs 158.1 ms) and `VICTORY_HEAP_SYNC_CUT=1` (141 vs 147.5 ms), both on by default now |

The same code on a friend's PC with a newer card that has mesh shaders runs at 9-11 fps walking and 12-15 fps at its
best.

For reference, the console runs this game at 30 fps. So it's playable, but slowly.

## Our PC

| | |
|---|---|
| CPU | AMD Ryzen 5 3600 (6 cores / 12 threads) |
| GPU | AMD Radeon RX 5700 XT (RDNA1), driver 32.0.21043.12001 |
| RAM | 32 GB (2 sticks), 3200 MT/s |
| OS | Windows 11 Pro (build 26200) |

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
