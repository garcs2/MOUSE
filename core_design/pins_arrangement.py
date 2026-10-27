# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
# Define placeholders for the variables
fuel_pin = 'FUEL'
moderator_pin = 'MODERATOR'

# Your list structure with placeholders
LTMR_pins_arrangement = [

[moderator_pin, fuel_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin,\
 fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin] * 6,

[fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin,\
 moderator_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin] *6,

[moderator_pin, fuel_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, fuel_pin,\
 fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin] *6,
 
[fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin,\
 moderator_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, fuel_pin] *6, 

[moderator_pin, fuel_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, fuel_pin,\
 fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin] *6, 

[fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin,\
 moderator_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin] *6,
 
[moderator_pin, fuel_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, fuel_pin,\
 fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin] *6, 
[fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin,\
 moderator_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin] *6,
[moderator_pin, fuel_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, fuel_pin,\
 fuel_pin, fuel_pin, fuel_pin, fuel_pin] *6, 

[fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin,\
 moderator_pin, fuel_pin, moderator_pin] *6,

[moderator_pin, fuel_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, fuel_pin,\
 fuel_pin, fuel_pin] *6,

[fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin,\
 moderator_pin] * 6,
    



[moderator_pin, fuel_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, fuel_pin] * 6,
[fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin] * 6,
[moderator_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin] * 6,

[fuel_pin, fuel_pin, moderator_pin, fuel_pin, fuel_pin, fuel_pin, moderator_pin, fuel_pin] * 6,
[moderator_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin] * 6,
[fuel_pin, fuel_pin, moderator_pin, fuel_pin, moderator_pin, fuel_pin] * 6,

[moderator_pin, fuel_pin, fuel_pin, fuel_pin, fuel_pin] * 6,
[fuel_pin, fuel_pin, moderator_pin, fuel_pin] * 6,
[moderator_pin, fuel_pin, fuel_pin] * 6,

[fuel_pin, moderator_pin] * 6,
[fuel_pin] * 6,
[moderator_pin]
]





# ----------------------------------------------------------------------------------
#  Regular moderator sub-lattice pattern (60 rings) and symmetric moderator thinning
#
#  LTMR_pins_arrangement_sublattice: moderator pins on every 2nd lattice site along both
#  lattice directions, centred on the middle pin, so each moderator pin is surrounded by
#  six fuel pins. This sub-lattice has the full symmetry of the hexagonal lattice (60-degree
#  rotations and mirrors), so every core cut from it is fully symmetric and every core size
#  has the same layout (~24-29 % moderator; small cores are a little higher because of their
#  edges). Rings are listed outermost first and each ring repeats one side ('unit') 6 times,
#  in OpenMC HexLattice ring order, like LTMR_pins_arrangement (which is kept unchanged).
#
#  thin_moderator_symmetric(arrangement, fraction): keeps a fraction of the moderator pins
#  (the rest become fuel) without breaking the symmetry. For 0.5 it uses an exact regular
#  rule (see below); 1.0 keeps all; 0.0 makes the whole lattice fuel. Other fractions drop
#  whole symmetric groups of pins, ring by ring, holding a running total at the target.
# ----------------------------------------------------------------------------------
SUBLATTICE_RINGS = 60

_RING_STEPS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]   # walk one ring (axial coords)


def _ring_coords(k):
    """Axial coordinates (q, r) of the 6k positions of ring k, starting at one corner."""
    if k == 0:
        return [(0, 0)]
    q, r, pts = 0, -k, []
    for dq, dr in _RING_STEPS:
        for _ in range(k):
            pts.append((q, r))
            q, r = q + dq, r + dr
    return pts


def _on_sublattice(q, r):
    return q % 2 == 0 and r % 2 == 0


def _sublattice_ring(k):
    row = [moderator_pin if _on_sublattice(q, r) else fuel_pin for q, r in _ring_coords(k)]
    if k:
        assert row == row[:k] * 6, f"ring {k} is not 6-fold symmetric"
    return row


LTMR_pins_arrangement_sublattice = [_sublattice_ring(k) for k in range(SUBLATTICE_RINGS - 1, -1, -1)]


def _keep_half(q, r):
    """Exact fraction-0.5 rule for moderator sites (q, r) of the sub-lattice. With
    (a, b) = (q/2, r/2) the site's position on the moderator grid, keep it if it lies on the
    grid of every 2nd moderator site (a, b both even) OR on the 1-in-3 grid through the
    centre (a = b mod 3). Both grids are fully symmetric, so their union is too, and it
    holds exactly 1/4 + 1/3 - 1/12 = 1/2 of the moderator sites."""
    a, b = q // 2, r // 2
    return (a % 2 == 0 and b % 2 == 0) or (a - b) % 3 == 0


def _symmetry_images(q, r):
    out, p = set(), (q, r)
    for _ in range(6):
        p = (-p[1], p[0] + p[1])           # 60-degree rotation
        out.add(p)
        out.add((p[1], p[0]))              # and its mirror image
    return out


def thin_moderator_symmetric(arrangement, fraction):
    """Copy of `arrangement` (rings outermost first) keeping `fraction` of its moderator
    pins as moderator and turning the rest into fuel, preserving full hexagonal symmetry.
    Works ring by ring from the centre, so the inner N rings of a thinned pattern are the
    same as thinning an N-ring pattern."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"moderator fraction must be in [0, 1], got {fraction}")
    n = len(arrangement)
    coords = [_ring_coords(n - 1 - i) for i in range(n)]
    if fraction == 1.0:
        return [list(row) for row in arrangement]
    if fraction == 0.0:
        return [[fuel_pin] * len(row) for row in arrangement]

    sites = {p: pin for row, pts in zip(arrangement, coords) for p, pin in zip(pts, row)}
    mods = [p for p, pin in sites.items() if pin == moderator_pin]
    if fraction == 0.5 and all(_on_sublattice(*p) for p in mods):
        keep = {p for p in mods if _keep_half(*p)}
    else:
        # general fraction: symmetric groups, ring by ring, running total at the target
        groups, seen = [], set()
        for p in mods:
            if p in seen:
                continue
            grp = sorted(_symmetry_images(*p) & set(sites))
            seen.update(grp)
            q, r = grp[0]
            groups.append((max(abs(q), abs(r), abs(q + r)), grp))
        groups.sort(key=lambda g: (g[0], g[1]))
        keep, kept, total = set(), 0, 0
        for _, grp in groups:
            w = len(grp)
            total += w
            if kept + w <= fraction * total + w / 2:
                keep.update(grp)
                kept += w
    return [[moderator_pin if (pin == moderator_pin and p in keep) else fuel_pin
             for p, pin in zip(pts, row)] for row, pts in zip(arrangement, coords)]
