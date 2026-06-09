# The Orbital Mechanic

A 2D gravity simulator built with Python and Pygame. It starts with a visual Earth and moon, then lets you launch either the satellite or the moon with a specific velocity vector and watch whether the body impacts, escapes, or settles into orbit.

Update 2.0 adds a start screen with the Earth-Moon Sim mode and a disabled Coming Soon mode, while the simulation animates in the background.

Planet impacts now create a visual plume with bright fire particles and slower smoke puffs.

Impacts also kick the camera with a quick shake, scaled by the force of the hit.

A faint gray ghost path predicts the selected launch trajectory before launch.

The ghost path now marks likely impacts, escapes, and bound-orbit predictions before launch, including impact force meters and catastrophic Earth-rupture warnings.

Close moon passes now generate a fiery atmospheric streak behind the moon.

Moon impacts now permanently scar Earth with a dented crater, red-hot impact site, cooling glow, and subtle fracture cracks.

If Earth takes five impacts, or one impact above the catastrophic force threshold, it explodes into fire, smoke, and debris while keeping its crater scars visible on the shattered remnant.

## Run

```bash
python -m pip install -r requirements.txt
python orbital_mechanic.py
```

## Controls

Most common actions are now clickable in the **Quick Controls** panel:

- On the start screen, click `Earth-Moon Sim` or press `Enter` / `Space` to begin

- `Sat` / `Moon`: choose the launch target
- `Launch`, `Orbit`, `Rev`: launch normally, auto-orbit, or reverse auto-orbit
- `Pause`, `Step`, `Reset`, `Clear`, `Ghost`, `Center`: simulation and view actions
- `Hit Pause`: toggle automatic pause when the moon or satellite impacts another body
- `Time -`, `1x`, `Time +`: scrub simulation speed
- `Earth`, `Moon`, `Sat`, `G -`, `G 1x`, `G +`: choose and adjust gravity

Keyboard and mouse shortcuts still work:

- `H` or the `Hide` / `HUD` button: hide or restore menus, readouts, prediction, and launch controls
- Mouse wheel: zoom the camera in or out around Earth
- Right-click drag: pan the camera
- `F`: recenter the camera on Earth
- `Tab`: switch between satellite launch mode and moon launch mode
- `S` / `M`: select the satellite or moon directly
- `O`: place the selected body into a near-circular orbit around Earth
- `Shift` + `O`: place the selected body into a reverse near-circular orbit
- `1` / `2` / `3`: choose whose gravity multiplier to edit: Earth, moon, or satellite
- `[` / `]`: lower or raise the selected gravity multiplier
- `Shift` + `[` / `]`: make a larger gravity adjustment
- `0`: reset the selected gravity multiplier to `1.00x`
- `G`: show or hide the orbit prediction ghost path
- `I`: toggle pause on impact
- `,` / `.`: scrub the simulation time speed down or up
- `Shift` + `,` / `.`: scrub time in larger jumps
- `T`: return to normal `1.00x` time
- `N`: while paused, advance one physics frame
- `Left` / `Right`: rotate the selected launch vector
- `Up` / `Down`: increase or decrease the selected launch speed
- `Shift` + arrow keys: fine adjustment
- Click and drag the moving moon: change its live velocity immediately
- Mouse drag from the selected launch point: set a specific `vx, vy` vector visually
- `Space`: launch or relaunch the selected body from its starting point
- `P`: pause
- `R`: reset the planet-moon system
- `C`: clear trails
- `+` / `-`: also scrub the simulation time speed

## Physics

The simulator uses Newtonian N-body gravity:

```text
F = G * m1 * m2 / r^2
```

Each body accelerates toward every other live body. The integrator is velocity Verlet, which is more stable for orbital motion than a basic Euler step. A small softening term prevents numerical spikes when two bodies get very close.

The planet, moon, and satellite all interact gravitationally. The planet-moon pair starts with barycentric velocities so the system begins close to a stable orbit instead of relying on a fixed central body.
