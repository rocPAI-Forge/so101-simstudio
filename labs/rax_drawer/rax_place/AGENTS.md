# RAX place agent (SO-101 drawer)

You are the coding agent for one simulated SO-101 arm. The robot service is the only process that moves the arm. In this session call tools through:

```python
from labs.rax_drawer.rax_place.tools import Session

with Session(split="explore") as ctx:
    ctx.reset()
    obs = ctx.observe(include_cameras=True)
    ctx.step({"vy": 0.02}, duration_s=0.25)
    print(ctx.check_success())
```

## Rules

1. Use only `reset`, `observe`, `step`, and `check_success`.
2. Explore only with `split="explore"`. Never open `manifests/test.json` for tuning.
3. Do not read MuJoCo drawer joints, cube poses, handle geoms, or contact arrays for control.
4. After a successful episode, write notes under `memory/` and compile `code/open_place_close/controller.py` with `main(ctx)`.
5. Self-test the compiled program on explore scenes before any locked evaluation.
6. Locked test is one frozen run after the program is frozen.

## Loop

1. Explore with short steps. Prefer cameras when deciding where to grasp or place.
2. Record working sequences in `memory/NOTES.md`.
3. Compile a controller that takes fresh observations and does not hardcode a single layout.
4. Run unattended explore self-tests. If they fail, return to exploration.
