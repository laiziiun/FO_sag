
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

G = 9.80665

# ============================================================
# Vector / coordinate helpers
# ============================================================

def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray, fallback=None) -> np.ndarray:
    n = norm(v)
    if n < 1e-12:
        if fallback is None:
            return np.array([1.0, 0.0, 0.0])
        return np.asarray(fallback, dtype=float)
    return v / n


def heading_vectors(heading_deg: float):
    """
    Coordinate convention:
      X/Y/Z are metres.
      Z is up.
      Heading 0 deg = +X.
      Heading 90 deg = +Y.
    """
    th = math.radians(heading_deg)
    forward = np.array([math.cos(th), math.sin(th), 0.0])
    left = np.array([-math.sin(th), math.cos(th), 0.0])
    return forward, left


def local_to_world_xy(x_local, y_local, origin_xy, heading_deg):
    forward, left = heading_vectors(heading_deg)
    p = np.array([origin_xy[0], origin_xy[1], 0.0]) + x_local * forward + y_local * left
    return float(p[0]), float(p[1])


def world_to_local_xy(point_xy, origin_xy, heading_deg):
    forward, left = heading_vectors(heading_deg)
    rel = np.array([point_xy[0] - origin_xy[0], point_xy[1] - origin_xy[1], 0.0])
    along = float(np.dot(rel, forward))
    lateral = float(np.dot(rel, left))
    return along, lateral


def bearing_deg(dx, dy):
    """0 deg = +X, 90 deg = +Y."""
    return (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0


# ============================================================
# Convex hull / protected volume helpers
# ============================================================

def clean_points(points_df):
    pts = []
    for _, row in points_df.iterrows():
        try:
            x = float(row["x_m"])
            y = float(row["y_m"])
            if np.isfinite(x) and np.isfinite(y):
                pts.append((x, y))
        except Exception:
            pass

    # Remove exact duplicates while preserving first occurrence.
    unique = []
    seen = set()
    for x, y in pts:
        key = (round(x, 9), round(y, 9))
        if key not in seen:
            unique.append((x, y))
            seen.add(key)
    return unique


def convex_hull(points):
    """
    Andrew monotone chain convex hull.
    Input may be unordered. Output is CCW hull vertices without repeated first point.
    """
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return hull


def polygon_centroid(poly):
    if len(poly) == 0:
        return (0.0, 0.0)

    area2 = 0.0
    cx = 0.0
    cy = 0.0
    n = len(poly)

    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    if abs(area2) < 1e-9:
        return (float(np.mean([p[0] for p in poly])), float(np.mean([p[1] for p in poly])))

    return (cx / (3 * area2), cy / (3 * area2))


def point_in_polygon(x, y, poly):
    if len(poly) < 3:
        return False

    inside = False
    n = len(poly)

    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        # Treat boundary as inside.
        dx = x2 - x1
        dy = y2 - y1
        seg_len2 = dx * dx + dy * dy
        if seg_len2 > 1e-12:
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg_len2))
            px = x1 + t * dx
            py = y1 + t * dy
            if math.hypot(x - px, y - py) < 1e-7:
                return True

        if ((y1 > y) != (y2 > y)):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1
            if x <= x_cross:
                inside = not inside

    return inside


def row_inside_volume(row, poly, z_min, z_max):
    return (
        point_in_polygon(float(row["x_world_m"]), float(row["y_world_m"]), poly)
        and z_min <= float(row["z_world_m"]) <= z_max
    )


def interp_row(row0, row1, frac):
    return row0 + frac * (row1 - row0)


def find_first_volume_entry(df, poly, z_min, z_max):
    """Find first point where cable enters the extruded protected volume."""
    if len(poly) < 3:
        return None

    prev_inside = row_inside_volume(df.iloc[0], poly, z_min, z_max)
    if prev_inside:
        return df.iloc[0].copy()

    for i in range(1, len(df)):
        curr_inside = row_inside_volume(df.iloc[i], poly, z_min, z_max)

        if curr_inside and not prev_inside:
            lo = 0.0
            hi = 1.0
            r0 = df.iloc[i - 1]
            r1 = df.iloc[i]

            # Bisection between outside and inside sampled points.
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                rm = interp_row(r0, r1, mid)
                if row_inside_volume(rm, poly, z_min, z_max):
                    hi = mid
                else:
                    lo = mid

            return interp_row(r0, r1, hi).copy()

        prev_inside = curr_inside

    return None


def find_height_crossings_before_s(df, z_abs, s_limit):
    crossings = []

    for i in range(1, len(df)):
        s0 = float(df["s_m"].iloc[i - 1])
        s1 = float(df["s_m"].iloc[i])

        if s0 > s_limit:
            break

        z0 = float(df["z_world_m"].iloc[i - 1])
        z1 = float(df["z_world_m"].iloc[i])

        if abs(z0 - z_abs) < 1e-9 and s0 <= s_limit:
            crossings.append(df.iloc[i - 1].copy())
            continue

        if (z0 - z_abs) * (z1 - z_abs) <= 0 and abs(z1 - z0) > 1e-12:
            frac = (z_abs - z0) / (z1 - z0)
            s_cross = s0 + frac * (s1 - s0)

            if s_cross <= s_limit + 1e-9:
                row = interp_row(df.iloc[i - 1], df.iloc[i], frac)
                row["z_world_m"] = z_abs
                crossings.append(row.copy())

    unique = []
    for row in crossings:
        p = np.array([row["x_world_m"], row["y_world_m"], row["z_world_m"]])
        duplicate = False
        for old in unique:
            q = np.array([old["x_world_m"], old["y_world_m"], old["z_world_m"]])
            if norm(p - q) < 1e-6:
                duplicate = True
                break
        if not duplicate:
            unique.append(row)

    return unique


# ============================================================
# Ground-contact / lift-off cable model
# ============================================================

def force_per_length(
    tension_vec,
    mass_kg_per_m,
    diameter_m,
    cd,
    rho,
    cable_velocity_vec,
    wind_velocity_vec,
    include_drag=True,
):
    tangent = unit(tension_vec)
    f_g = np.array([0.0, 0.0, -mass_kg_per_m * G])

    if not include_drag or cd <= 0 or diameter_m <= 0 or rho <= 0:
        return f_g

    v_rel = np.asarray(cable_velocity_vec, dtype=float) - np.asarray(wind_velocity_vec, dtype=float)
    v_parallel = np.dot(v_rel, tangent) * tangent
    v_perp = v_rel - v_parallel
    vmag = norm(v_perp)

    if vmag < 1e-9:
        return f_g

    f_d = -0.5 * rho * cd * diameter_m * vmag * v_perp
    return f_g + f_d


def rhs(
    state,
    mass_kg_per_m,
    diameter_m,
    cd,
    rho,
    cable_velocity_vec,
    wind_velocity_vec,
    include_drag,
):
    T = state[3:6]
    tangent = unit(T)
    f = force_per_length(
        T,
        mass_kg_per_m,
        diameter_m,
        cd,
        rho,
        cable_velocity_vec,
        wind_velocity_vec,
        include_drag,
    )
    dT_ds = -f
    return np.r_[tangent, dT_ds]


def rk4_step(state, ds, *args):
    k1 = rhs(state, *args)
    k2 = rhs(state + 0.5 * ds * k1, *args)
    k3 = rhs(state + 0.5 * ds * k2, *args)
    k4 = rhs(state + ds * k3, *args)
    return state + ds * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def simulate_ground_liftoff_local(
    H0_N,
    drone_height_above_liftoff_m,
    mass_kg_per_m,
    diameter_m,
    cd,
    rho,
    drone_speed_mps,
    headwind_mps,
    crosswind_mps,
    climb_rate_mps,
    initial_lateral_tension_N,
    ds_m,
    max_s_m,
    include_drag=True,
):
    """
    Ground-contact model.
    Starts at the ground lift-off point:
      r(0) = [0,0,0]
      T(0) = [H0, Ty0, 0]
    This is only valid if a real ground-contact / lift-off point exists.
    """
    cable_velocity_vec = np.array([drone_speed_mps, 0.0, climb_rate_mps])
    wind_velocity_vec = np.array([-headwind_mps, crosswind_mps, 0.0])

    state = np.array([0.0, 0.0, 0.0, H0_N, initial_lateral_tension_N, 0.0], dtype=float)
    n_steps = int(max_s_m / ds_m) + 1
    rows = []
    reached_height = False

    args = (
        mass_kg_per_m,
        diameter_m,
        cd,
        rho,
        cable_velocity_vec,
        wind_velocity_vec,
        include_drag,
    )

    for i in range(n_steps):
        s = i * ds_m
        x, y, z, tx, ty, tz = state
        rows.append([s, x, y, z, tx, ty, tz, math.sqrt(tx * tx + ty * ty + tz * tz)])

        if z >= drone_height_above_liftoff_m and i > 2:
            reached_height = True
            break

        state = rk4_step(state, ds_m, *args)

        if not np.all(np.isfinite(state)) or norm(state[3:6]) < 1e-6:
            break

    df = pd.DataFrame(
        rows,
        columns=["s_m", "x_local_m", "y_local_m", "z_local_m", "Tx_N", "Ty_N", "Tz_N", "T_N"],
    )

    if reached_height and len(df) >= 2:
        z1, z2 = df["z_local_m"].iloc[-2], df["z_local_m"].iloc[-1]
        if abs(z2 - z1) > 1e-9:
            frac = (drone_height_above_liftoff_m - z1) / (z2 - z1)
            prev = df.iloc[-2]
            last = df.iloc[-1]
            interp = prev + frac * (last - prev)
            interp["z_local_m"] = drone_height_above_liftoff_m
            df = pd.concat([df.iloc[:-1], pd.DataFrame([interp])], ignore_index=True)

    return df, reached_height


def solve_H0_for_ground_range(
    target_x_m,
    drone_height_above_liftoff_m,
    mass_kg_per_m,
    diameter_m,
    cd,
    rho,
    drone_speed_mps,
    headwind_mps,
    crosswind_mps,
    climb_rate_mps,
    initial_lateral_tension_N,
    ds_m,
    max_s_m,
    include_drag,
    H_min,
    H_max,
    iterations=50,
):
    """
    Ground-contact solve mode.
    Finds H0 such that the ground-lift-off curve reaches drone height at x=target_x_m.
    This only constrains x. If crosswind is present, prefer solve_H0_Ty0_for_ground_endpoint().
    """

    def run(H):
        df, ok = simulate_ground_liftoff_local(
            H,
            drone_height_above_liftoff_m,
            mass_kg_per_m,
            diameter_m,
            cd,
            rho,
            drone_speed_mps,
            headwind_mps,
            crosswind_mps,
            climb_rate_mps,
            initial_lateral_tension_N,
            ds_m,
            max_s_m,
            include_drag,
        )
        if not ok or len(df) == 0:
            return None, ok, df
        return float(df["x_local_m"].iloc[-1]), ok, df

    x_lo, ok_lo, df_lo = run(H_min)
    x_hi, ok_hi, df_hi = run(H_max)

    if x_lo is None or x_hi is None:
        return None, None, None, "Could not reach drone height within max cable length for the selected H0 bracket."

    if not (x_lo <= target_x_m <= x_hi):
        return None, None, None, (
            f"Target drone ground range is outside the ground-contact H0 bracket. "
            f"At H0={H_min:.3g} N, x≈{x_lo:.1f} m; "
            f"at H0={H_max:.3g} N, x≈{x_hi:.1f} m. "
            f"This often means the ground-contact/lift-off assumption is invalid. "
            f"Try Auto or Fully suspended source-to-drone mode."
        )

    lo, hi = H_min, H_max
    best_H, best_x, best_df = None, None, None

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        x_mid, ok_mid, df_mid = run(mid)

        if x_mid is None:
            lo = mid
            continue

        best_H, best_x, best_df = mid, x_mid, df_mid

        if x_mid < target_x_m:
            lo = mid
        else:
            hi = mid

    return best_H, best_x, best_df, None


def solve_H0_for_ground_range_given_Ty(
    target_x_m,
    drone_height_above_liftoff_m,
    mass_kg_per_m,
    diameter_m,
    cd,
    rho,
    drone_speed_mps,
    headwind_mps,
    crosswind_mps,
    climb_rate_mps,
    Ty0_N,
    ds_m,
    max_s_m,
    include_drag,
    H_min,
    H_max,
    iterations=45,
):
    """
    Inner solve for crosswind endpoint matching.
    Given Ty0, find H0 such that the curve reaches drone height at x=target_x_m.
    Returns final y as well, so the outer solve can adjust Ty0.
    """

    def run(H):
        df, ok = simulate_ground_liftoff_local(
            H,
            drone_height_above_liftoff_m,
            mass_kg_per_m,
            diameter_m,
            cd,
            rho,
            drone_speed_mps,
            headwind_mps,
            crosswind_mps,
            climb_rate_mps,
            Ty0_N,
            ds_m,
            max_s_m,
            include_drag,
        )
        if not ok or len(df) == 0:
            return None, None, ok, df
        return float(df["x_local_m"].iloc[-1]), float(df["y_local_m"].iloc[-1]), ok, df

    x_lo, y_lo, ok_lo, df_lo = run(H_min)
    x_hi, y_hi, ok_hi, df_hi = run(H_max)

    if x_lo is None or x_hi is None:
        return None, None, None, None, "Could not reach drone height within max cable length for this Ty0."

    if not (min(x_lo, x_hi) <= target_x_m <= max(x_lo, x_hi)):
        return None, None, None, None, (
            f"For Ty0={Ty0_N:.3g} N, target x is outside H0 bracket. "
            f"At H0={H_min:.3g} N, x≈{x_lo:.1f} m; "
            f"at H0={H_max:.3g} N, x≈{x_hi:.1f} m."
        )

    lo, hi = H_min, H_max
    best_H, best_x, best_y, best_df = None, None, None, None

    # Support either monotonic direction, though x normally increases with H.
    increasing = x_hi >= x_lo

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        x_mid, y_mid, ok_mid, df_mid = run(mid)

        if x_mid is None:
            lo = mid
            continue

        best_H, best_x, best_y, best_df = mid, x_mid, y_mid, df_mid

        if increasing:
            if x_mid < target_x_m:
                lo = mid
            else:
                hi = mid
        else:
            if x_mid > target_x_m:
                lo = mid
            else:
                hi = mid

    return best_H, best_x, best_y, best_df, None


def solve_H0_Ty0_for_ground_endpoint(
    target_x_m,
    target_y_m,
    drone_height_above_liftoff_m,
    mass_kg_per_m,
    diameter_m,
    cd,
    rho,
    drone_speed_mps,
    headwind_mps,
    crosswind_mps,
    climb_rate_mps,
    ds_m,
    max_s_m,
    include_drag,
    H_min,
    H_max,
    Ty_min,
    Ty_max,
    iterations_ty=40,
    iterations_h=45,
):
    """
    Ground-contact 3D endpoint solve.
    Finds H0 and Ty0 such that the ground-lift-off curve reaches the drone height at:
      x = target_x_m
      y = target_y_m

    This fixes the crosswind issue where the cable bows sideways but should still end at the drone.
    The solve is nested:
      1. For each Ty0, solve H0 to match x.
      2. Adjust Ty0 until final y matches target_y_m.
    """
    if Ty_min >= Ty_max:
        return None, None, None, None, None, (
            "Ty0 lower bracket must be less than Ty0 upper bracket."
        )

    def run_ty(Ty):
        H, x_end, y_end, df, err = solve_H0_for_ground_range_given_Ty(
            target_x_m,
            drone_height_above_liftoff_m,
            mass_kg_per_m,
            diameter_m,
            cd,
            rho,
            drone_speed_mps,
            headwind_mps,
            crosswind_mps,
            climb_rate_mps,
            Ty,
            ds_m,
            max_s_m,
            include_drag,
            H_min,
            H_max,
            iterations=iterations_h,
        )
        if err:
            return None, None, None, None, err
        return H, x_end, y_end, df, None

    # Scan Ty bracket first to find a valid sign change in final y error.
    scan_Ty = np.linspace(Ty_min, Ty_max, 25)
    scan = []
    failed = []

    for Ty in scan_Ty:
        H, x_end, y_end, df, err = run_ty(float(Ty))
        if err:
            failed.append((float(Ty), err))
            continue

        y_err = y_end - target_y_m
        scan.append({
            "Ty": float(Ty),
            "H": H,
            "x": x_end,
            "y": y_end,
            "y_err": y_err,
            "df": df,
        })

        if abs(y_err) < 1e-4:
            return H, float(Ty), x_end, y_end, df, None

    if len(scan) == 0:
        return None, None, None, None, None, (
            "Could not find any Ty0 value where the inner H0 solve succeeded. "
            "Widen H0 bracket, widen Ty0 bracket, increase max cable length, or try Auto/Fully suspended mode."
        )

    # Find a sign-changing pair. Prefer the pair nearest target y.
    bracket = None
    for a, b in zip(scan[:-1], scan[1:]):
        if a["y_err"] == 0:
            return a["H"], a["Ty"], a["x"], a["y"], a["df"], None
        if a["y_err"] * b["y_err"] <= 0:
            bracket = (a, b)
            break

    if bracket is None:
        y_values = [r["y"] for r in scan]
        Ty_values = [r["Ty"] for r in scan]
        closest = min(scan, key=lambda r: abs(r["y_err"]))
        return None, None, None, None, None, (
            f"Could not bracket final y={target_y_m:.2f} m within Ty0 range. "
            f"Valid scan final-y range was {min(y_values):.2f} m to {max(y_values):.2f} m "
            f"over Ty0={min(Ty_values):.2f} N to {max(Ty_values):.2f} N. "
            f"Closest was Ty0={closest['Ty']:.3g} N giving y≈{closest['y']:.2f} m. "
            "Widen the Ty0 bracket."
        )

    lo = bracket[0]
    hi = bracket[1]

    # Bisection on Ty0 to match y.
    best = min([lo, hi], key=lambda r: abs(r["y_err"]))

    for _ in range(iterations_ty):
        mid_Ty = 0.5 * (lo["Ty"] + hi["Ty"])
        H, x_end, y_end, df, err = run_ty(mid_Ty)

        if err:
            # If one mid-run fails, stop gracefully with the best valid result.
            break

        mid = {
            "Ty": mid_Ty,
            "H": H,
            "x": x_end,
            "y": y_end,
            "y_err": y_end - target_y_m,
            "df": df,
        }

        if abs(mid["y_err"]) < abs(best["y_err"]):
            best = mid

        if lo["y_err"] * mid["y_err"] <= 0:
            hi = mid
        else:
            lo = mid

    return best["H"], best["Ty"], best["x"], best["y"], best["df"], None


# ============================================================
# Fully suspended source-to-drone approximate model
# ============================================================

def simulate_suspended_two_point_approx(
    H_N,
    drone_ground_range_m,
    drone_height_above_liftoff_m,
    mass_kg_per_m,
    diameter_m,
    cd,
    rho,
    drone_speed_mps,
    headwind_mps,
    crosswind_mps,
    ds_m,
):
    """
    Fully suspended source-to-drone approximation.

    This model always terminates at the actual drone point, so it fixes the previous
    'cable overshoots the drone' issue for close/high/tensioned cases.

    It uses a parabolic cable approximation between fixed endpoints:
      x = D*u
      z = h*u - sag_mid*4*u*(1-u)
      y = lateral_bow_mid*4*u*(1-u)

    Vertical sag is estimated from cable self-weight and horizontal tension:
      sag_mid ≈ w*D^2/(8H)

    Crosswind lateral bow is estimated similarly using signed lateral drag.

    It is less rigorous than a full two-point boundary-value cable solver, but it is
    stable, transparent and suitable for checking whether the no-ground-contact
    regime is plausible.
    """
    D = max(float(drone_ground_range_m), 1e-9)
    h = float(drone_height_above_liftoff_m)
    H = max(float(H_N), 1e-6)
    w = mass_kg_per_m * G

    chord_len = math.sqrt(D * D + h * h)
    n = max(40, int(chord_len / max(ds_m, 0.05)) + 1)
    u = np.linspace(0.0, 1.0, n)
    shape = 4.0 * u * (1.0 - u)

    sag_mid = w * D * D / (8.0 * H)

    # Crosswind: wind +Y pushes cable +Y in the middle.
    # Endpoints remain fixed at y=0 because the cable is connected to source and drone.
    q_lat = 0.5 * rho * cd * diameter_m * abs(crosswind_mps) * crosswind_mps
    lateral_bow_mid = q_lat * D * D / (8.0 * H)

    # Forward drag is not allowed to move endpoints, but it increases estimated tension.
    v_forward_rel = drone_speed_mps + headwind_mps
    q_forward = 0.5 * rho * cd * diameter_m * abs(v_forward_rel) * v_forward_rel

    x = D * u
    y = lateral_bow_mid * shape
    z = h * u - sag_mid * shape

    # Arc length and approximate tension from slopes.
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)
    ds_seg = np.sqrt(dx * dx + dy * dy + dz * dz)
    s = np.cumsum(ds_seg)
    s -= s[0]

    dzdx = np.gradient(z, x, edge_order=1)
    dydx = np.gradient(y, x, edge_order=1)
    Tz = H * dzdx
    Ty = H * dydx
    Tx = np.full_like(x, H) + q_forward * D * (1.0 - u)
    T = np.sqrt(Tx * Tx + Ty * Ty + Tz * Tz)

    return pd.DataFrame({
        "s_m": s,
        "x_local_m": x,
        "y_local_m": y,
        "z_local_m": z,
        "Tx_N": Tx,
        "Ty_N": Ty,
        "Tz_N": Tz,
        "T_N": T,
    })


def attach_world_coordinates(df_local, liftoff_xyz, heading_deg):
    out = df_local.copy()
    origin_xy = [liftoff_xyz[0], liftoff_xyz[1]]

    xw, yw, zw = [], [], []
    for x_local, y_local, z_local in zip(out["x_local_m"], out["y_local_m"], out["z_local_m"]):
        x, y = local_to_world_xy(x_local, y_local, origin_xy, heading_deg)
        xw.append(x)
        yw.append(y)
        zw.append(liftoff_xyz[2] + z_local)

    out["x_world_m"] = xw
    out["y_world_m"] = yw
    out["z_world_m"] = zw
    return out


# ============================================================
# Cutter geometry
# ============================================================

def local_path_direction_at_s(df, s_value, fallback_heading_deg):
    s_arr = df["s_m"].to_numpy()
    idx = int(np.argmin(np.abs(s_arr - s_value)))
    i0 = max(0, idx - 1)
    i1 = min(len(df) - 1, idx + 1)

    dx = float(df["x_world_m"].iloc[i1] - df["x_world_m"].iloc[i0])
    dy = float(df["y_world_m"].iloc[i1] - df["y_world_m"].iloc[i0])

    if math.hypot(dx, dy) < 1e-9:
        return fallback_heading_deg % 360.0

    return bearing_deg(dx, dy)


def cutter_segment_from_center(center_x, center_y, center_z, orientation_deg, length_m):
    th = math.radians(orientation_deg)
    ux = math.cos(th)
    uy = math.sin(th)
    half = 0.5 * length_m

    p1 = (center_x - half * ux, center_y - half * uy, center_z)
    p2 = (center_x + half * ux, center_y + half * uy, center_z)
    return p1, p2


# ============================================================
# Plotting
# ============================================================

def add_protected_volume(fig, hull, z_min, z_max, raw_points=None):
    if len(hull) < 3:
        return

    cx, cy = polygon_centroid(hull)

    x = [cx]
    y = [cy]
    z = [z_min]

    for px, py in hull:
        x.append(px)
        y.append(py)
        z.append(z_min)

    top_center_idx = len(x)
    x.append(cx)
    y.append(cy)
    z.append(z_max)

    top_start_idx = len(x)
    for px, py in hull:
        x.append(px)
        y.append(py)
        z.append(z_max)

    n = len(hull)
    I, J, K = [], [], []

    # Bottom fan.
    for i in range(n):
        I.append(0)
        J.append(1 + ((i + 1) % n))
        K.append(1 + i)

    # Top fan.
    for i in range(n):
        I.append(top_center_idx)
        J.append(top_start_idx + i)
        K.append(top_start_idx + ((i + 1) % n))

    # Side faces.
    for i in range(n):
        b0 = 1 + i
        b1 = 1 + ((i + 1) % n)
        t0 = top_start_idx + i
        t1 = top_start_idx + ((i + 1) % n)
        I.extend([b0, b0])
        J.extend([b1, t1])
        K.extend([t1, t0])

    fig.add_trace(go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=I,
        j=J,
        k=K,
        opacity=0.22,
        name="Protected convex-hull volume",
        hoverinfo="name",
    ))

    closed = hull + [hull[0]]
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in closed],
        y=[p[1] for p in closed],
        z=[z_min] * len(closed),
        mode="lines",
        line=dict(width=4),
        name="Convex hull footprint",
    ))
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in closed],
        y=[p[1] for p in closed],
        z=[z_max] * len(closed),
        mode="lines",
        line=dict(width=3),
        name="Protected volume top",
    ))

    for px, py in hull:
        fig.add_trace(go.Scatter3d(
            x=[px, px],
            y=[py, py],
            z=[z_min, z_max],
            mode="lines",
            line=dict(width=2),
            showlegend=False,
        ))

    if raw_points is not None and len(raw_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in raw_points],
            y=[p[1] for p in raw_points],
            z=[z_min] * len(raw_points),
            mode="markers+text",
            marker=dict(size=5),
            text=[f"P{i+1}" for i in range(len(raw_points))],
            textposition="top center",
            name="Raw footprint points",
        ))


def make_3d_plot(df, hull, raw_points, z_min, z_max, liftoff_xyz, drone_xyz, entry_row, cutter, model_used):
    fig = go.Figure()

    add_protected_volume(fig, hull, z_min, z_max, raw_points=raw_points)

    fig.add_trace(go.Scatter3d(
        x=df["x_world_m"],
        y=df["y_world_m"],
        z=df["z_world_m"],
        mode="lines",
        line=dict(width=6),
        name=f"Sagging cable ({model_used})",
    ))

    fig.add_trace(go.Scatter3d(
        x=[liftoff_xyz[0]],
        y=[liftoff_xyz[1]],
        z=[liftoff_xyz[2]],
        mode="markers+text",
        marker=dict(size=6),
        text=["Source / take-off"],
        textposition="top center",
        name="Source / take-off",
    ))

    fig.add_trace(go.Scatter3d(
        x=[drone_xyz[0]],
        y=[drone_xyz[1]],
        z=[drone_xyz[2]],
        mode="markers+text",
        marker=dict(size=7),
        text=["Drone"],
        textposition="top center",
        name="Drone",
    ))

    if entry_row is not None:
        fig.add_trace(go.Scatter3d(
            x=[entry_row["x_world_m"]],
            y=[entry_row["y_world_m"]],
            z=[entry_row["z_world_m"]],
            mode="markers+text",
            marker=dict(size=8),
            text=["Cable enters protected volume"],
            textposition="top center",
            name="Volume entry",
        ))

    if cutter is not None:
        p1 = cutter["p1"]
        p2 = cutter["p2"]
        c = cutter["center"]

        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode="lines+markers",
            line=dict(width=8),
            marker=dict(size=4),
            name="Recommended cutter span",
        ))

        fig.add_trace(go.Scatter3d(
            x=[c[0]],
            y=[c[1]],
            z=[c[2]],
            mode="markers+text",
            marker=dict(size=8),
            text=["Cutter centre"],
            textposition="top center",
            name="Cutter centre",
        ))

        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[0, 0],
            mode="lines",
            line=dict(width=4, dash="dash"),
            name="Cutter ground projection",
        ))

        fig.add_trace(go.Scatter3d(
            x=[c[0], c[0]],
            y=[c[1], c[1]],
            z=[0, c[2]],
            mode="lines",
            line=dict(width=3, dash="dot"),
            name="Cutter centre mast",
        ))

    fig.update_layout(
        title="3D protected volume, cable path and recommended cutter deployment. Hold left click to rotate, hold right click to pan, mousewheel to zoom.",
        scene=dict(
            xaxis_title="World X (m)",
            yaxis_title="World Y (m)",
            zaxis_title="World Z / height (m)",
            aspectmode="data",
        ),
        height=740,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h"),
    )
    return fig


def make_plan_plot(df, hull, raw_points, liftoff_xyz, drone_xyz, entry_row, cutter):
    fig = go.Figure()

    if len(hull) >= 3:
        closed = hull + [hull[0]]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in closed],
            y=[p[1] for p in closed],
            mode="lines",
            fill="toself",
            name="Protected convex hull",
        ))

    if len(raw_points) > 0:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in raw_points],
            y=[p[1] for p in raw_points],
            mode="markers+text",
            text=[f"P{i+1}" for i in range(len(raw_points))],
            textposition="top center",
            name="Raw points",
        ))

    fig.add_trace(go.Scatter(
        x=df["x_world_m"],
        y=df["y_world_m"],
        mode="lines",
        name="Cable plan path",
    ))

    fig.add_trace(go.Scatter(
        x=[liftoff_xyz[0]],
        y=[liftoff_xyz[1]],
        mode="markers+text",
        text=["Source"],
        textposition="top center",
        name="Source",
    ))

    fig.add_trace(go.Scatter(
        x=[drone_xyz[0]],
        y=[drone_xyz[1]],
        mode="markers+text",
        text=["Drone ground projection"],
        textposition="top center",
        name="Drone ground projection",
    ))

    if entry_row is not None:
        fig.add_trace(go.Scatter(
            x=[entry_row["x_world_m"]],
            y=[entry_row["y_world_m"]],
            mode="markers+text",
            text=["Volume entry"],
            textposition="top center",
            name="Volume entry",
        ))

    if cutter is not None:
        p1 = cutter["p1"]
        p2 = cutter["p2"]
        c = cutter["center"]

        fig.add_trace(go.Scatter(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            mode="lines+markers",
            line=dict(width=6),
            name="Recommended cutter span",
        ))

        fig.add_trace(go.Scatter(
            x=[c[0]],
            y=[c[1]],
            mode="markers+text",
            text=["Cutter centre"],
            textposition="top center",
            name="Cutter centre",
        ))

    fig.update_layout(
        title="Plan view: convex hull footprint and cutter orientation",
        xaxis_title="World X (m)",
        yaxis_title="World Y (m)",
        yaxis_scaleanchor="x",
        height=600,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_side_plot(df, heading_deg, liftoff_xyz, entry_row, cutter, z_min, z_max):
    along = []
    for x, y in zip(df["x_world_m"], df["y_world_m"]):
        a, _ = world_to_local_xy([x, y], [liftoff_xyz[0], liftoff_xyz[1]], heading_deg)
        along.append(a)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=along,
        y=df["z_world_m"],
        mode="lines",
        name="Cable side profile",
    ))

    fig.add_hrect(y0=z_min, y1=z_max, opacity=0.18, line_width=0, annotation_text="Protected height band")

    if entry_row is not None:
        e_along, _ = world_to_local_xy(
            [entry_row["x_world_m"], entry_row["y_world_m"]],
            [liftoff_xyz[0], liftoff_xyz[1]],
            heading_deg,
        )
        fig.add_trace(go.Scatter(
            x=[e_along],
            y=[entry_row["z_world_m"]],
            mode="markers+text",
            text=["Volume entry"],
            textposition="top center",
            name="Volume entry",
        ))

    if cutter is not None:
        c = cutter["center"]
        c_along, _ = world_to_local_xy([c[0], c[1]], [liftoff_xyz[0], liftoff_xyz[1]], heading_deg)
        fig.add_trace(go.Scatter(
            x=[c_along],
            y=[c[2]],
            mode="markers+text",
            text=["Cutter"],
            textposition="top center",
            name="Cutter",
        ))
        fig.add_hline(y=c[2], line_dash="dash", annotation_text=f"Cutter height {c[2]:.2f} m")

    fig.update_layout(
        title="Side view: cable height versus along-track distance",
        xaxis_title="Along-track distance from source / lift-off (m)",
        yaxis_title="World Z / height (m)",
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Cable sag model selector", layout="wide")

st.title("Drone fibre-optic cable sag model: model selector + protected convex-hull volume")

st.markdown(
    """
Best viewed on laptop / PC, not on phone. App models a drone fibre-optic cable and recommends a finite-length cutter deployment.
The protected asset is defined by raw ground points; the app automatically forms a **convex hull** and extrudes it into a transparent 3D volume.

The key fix in this version is the **cable regime selector**. Close/high/tensioned cases often have no ground contact, so the app should not force a ground-lift-off catenary.

**Default scenario:** liftoff at ground level, drone at 900 m range and 120 m altitude, protected footprint around 650–850 m downrange, 200 m protected volume height, light fibre cable, and light head/crosswind. These defaults are for visualisation and sensitivity testing. Normal for site to take a minute to load.
"""
)

with st.sidebar:
    st.header("Cable regime")
    cable_regime = st.radio(
        "Model selector",
        [
            "Auto: suspended first, ground-contact if needed",
            "Fully suspended source-to-drone",
            "Ground-contact / lift-off",
        ],
        help=(
            "Auto first checks a source-to-drone suspended curve. If it stays above ground, that is used. "
            "If it drops below ground, the app switches to the ground-contact/lift-off model."
        ),
    )

    st.header("Source / take-off")
    liftoff_x = st.number_input("Source X (m)", value=0.0, step=10.0)
    liftoff_y = st.number_input("Source Y (m)", value=0.0, step=10.0)
    liftoff_z = st.number_input("Source Z (m)", value=0.0, step=0.5)
    ground_z = st.number_input("Ground plane Z (m)", value=0.0, step=0.5)

    st.header("Drone current state")
    drone_ground_range_m = st.number_input(
        "Drone ground range from source (m)", min_value=1.0, max_value=20000.0, value=620.0, step=50.0
    )
    drone_height_above_liftoff_m = st.number_input(
        "Drone height above source (m)", min_value=0.1, max_value=2000.0, value=30.0, step=10.0
    )
    drone_heading_deg = st.number_input(
        "Drone heading (deg, 0=+X, 90=+Y)", min_value=-360.0, max_value=360.0, value=0.0, step=5.0
    )

    st.header("Protected volume")
    protected_z_min = st.number_input("Protected volume base Z (m)", value=0.0, step=0.5)
    protected_height_m = st.number_input(
        "Protected volume height (m)", min_value=1.0, max_value=500.0, value=200.0, step=1.0
    )
    protected_z_max = protected_z_min + protected_height_m

    st.header("Cutter")
    cutter_height_m = st.number_input(
        "Cutter active height, absolute Z (m)",
        value=3.0,
        step=1.0,
        format="%.3f",
        help="Enter any integer or decimal height. Default is 3m"
    )
    cutter_length_m = st.number_input("Cutter active length / span (m)", min_value=0.1, max_value=500.0, value=20.0, step=1.0)

    st.header("Tension")
    tension_mode = st.radio(
        "Ground-contact tension mode",
        ["Use specified H0 / Ty0", "Solve H0 and Ty0 to match drone X/Y endpoint"],
        index=1, # default to endpoint solve
        help=(
            "The solve option only applies to the ground-contact/lift-off model. "
            "It solves H0 and Ty0 together so the cable reaches drone height at the selected ground range and final lateral y=0. "
            "Fully suspended and Auto-suspended modes use the specified H0/source tension."
        ),
    )
    H0_N = st.number_input("Specified H0 / source horizontal tension (N)", min_value=0.001, max_value=5000.0, value=10.0, step=1.0)
    H_min = st.number_input("H0 lower bracket for ground-contact solve (N)", min_value=0.001, max_value=1000.0, value=0.1, step=0.1)
    H_max = st.number_input("H0 upper bracket for ground-contact solve (N)", min_value=0.001, max_value=20000.0, value=300.0, step=10.0)

    st.caption("Ty0 is lateral tension at the lift-off point. It is solved automatically in endpoint-solve mode.")
    initial_lateral_tension_N = st.number_input("Specified Ty0 for ground-contact model (N)", min_value=-1000.0, max_value=1000.0, value=0.0, step=0.1)
    Ty_min = st.number_input("Ty0 lower bracket for endpoint solve (N)", min_value=-5000.0, max_value=5000.0, value=-50.0, step=1.0)
    Ty_max = st.number_input("Ty0 upper bracket for endpoint solve (N)", min_value=-5000.0, max_value=5000.0, value=50.0, step=1.0)

    st.header("Cable properties")
    cable_mass_kg_per_km = st.number_input(
        "Cable mass / linear density (kg/km)", min_value=0.001, max_value=50.0, value=0.14, step=0.01, format="%.4f"
    )
    cable_diameter_mm = st.number_input("Cable diameter (mm)", min_value=0.01, max_value=20.0, value=0.8, step=0.1)

    st.header("Drone motion and wind")
    include_drag = st.checkbox("Include aerodynamic drag", value=True)
    drone_speed_mps = st.number_input("Drone forward speed (m/s)", min_value=0.0, max_value=80.0, value=0.0, step=1.0)
    climb_rate_mps = st.number_input("Drone climb rate / vertical cable speed (m/s)", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)
    headwind_mps = st.number_input("Headwind (m/s, positive = against drone)", min_value=-40.0, max_value=40.0, value=0.0, step=1.0)
    crosswind_mps = st.number_input("Crosswind (m/s, positive = local +Y)", min_value=-40.0, max_value=40.0, value=4.0, step=1.0)
    cd = st.number_input("Cable drag coefficient Cd", min_value=0.0, max_value=3.0, value=1.2, step=0.1)
    rho = st.number_input("Air density rho (kg/m³)", min_value=0.5, max_value=1.5, value=1.180, step=0.005, format="%.3f")

    st.header("Numerics")
    ds_m = st.number_input("Arc-length / point spacing ds (m)", min_value=0.05, max_value=20.0, value=2.0, step=0.5)
    max_s_m = st.number_input("Max cable length for ground-contact integration (m)", min_value=10.0, max_value=50000.0, value=10000.0, step=500.0)


# ============================================================
# Protected footprint point editor
# ============================================================

st.subheader("Protected asset footprint points")

if "points_df" not in st.session_state:
    st.session_state.points_df = pd.DataFrame({
        "x_m": [600.0, 850.0, 820.0, 600.0, 740.0],
        "y_m": [-80.0, -70.0, 90.0, 75.0, 0.0],
    })

# Change the editor key whenever buttons modify the dataframe.
# This forces Streamlit's data editor to refresh immediately after add/delete/reset actions.
if "points_editor_version" not in st.session_state:
    st.session_state.points_editor_version = 0

with st.expander("Quick add / delete / reset footprint points", expanded=True):
    st.markdown("Add points in any order. The protected footprint is automatically formed from the convex hull.")

    c1, c2, c3 = st.columns([1, 1, 1])
    new_x = c1.number_input("New point X (m)", value=760.0, step=10.0, key="new_point_x")
    new_y = c2.number_input("New point Y (m)", value=110.0, step=10.0, key="new_point_y")

    if c3.button("Add point", use_container_width=True):
        st.session_state.points_df = pd.concat(
            [
                st.session_state.points_df,
                pd.DataFrame([{"x_m": float(new_x), "y_m": float(new_y)}]),
            ],
            ignore_index=True,
        )
        st.session_state.points_editor_version += 1
        st.rerun()

    d1, d2, d3, d4 = st.columns([1.2, 1, 1, 1.4])

    point_options = [
        f"{i}: X={row.x_m:.2f}, Y={row.y_m:.2f}"
        for i, row in st.session_state.points_df.reset_index(drop=True).iterrows()
    ]

    delete_choice = d1.selectbox(
        "Point to delete",
        options=list(range(len(point_options))),
        format_func=lambda i: point_options[i] if point_options else "No points",
        disabled=len(point_options) == 0,
        key="delete_point_index",
    )

    if d2.button("Delete selected", use_container_width=True, disabled=len(point_options) == 0):
        st.session_state.points_df = (
            st.session_state.points_df
            .reset_index(drop=True)
            .drop(index=int(delete_choice))
            .reset_index(drop=True)
        )
        st.session_state.points_editor_version += 1
        st.rerun()

    if d3.button("Delete last", use_container_width=True, disabled=len(st.session_state.points_df) == 0):
        st.session_state.points_df = st.session_state.points_df.iloc[:-1].reset_index(drop=True)
        st.session_state.points_editor_version += 1
        st.rerun()

    if d4.button("Reset default rectangle", use_container_width=True):
        st.session_state.points_df = pd.DataFrame({
            "x_m": [600.0, 850.0, 820.0, 600.0, 740.0],
            "y_m": [-80.0, -70.0, 90.0, 75.0, 0.0],
        })
        st.session_state.points_editor_version += 1
        st.rerun()

edited_df = st.data_editor(
    st.session_state.points_df,
    num_rows="dynamic",
    use_container_width=True,
    key=f"points_editor_{st.session_state.points_editor_version}",
)
st.session_state.points_df = edited_df

raw_points = clean_points(edited_df)
hull = convex_hull(raw_points)

if len(raw_points) < 3 or len(hull) < 3:
    st.error("Enter at least three non-collinear footprint points to form a convex hull.")
    st.stop()

st.caption(
    f"Using {len(raw_points)} raw point(s). Convex hull has {len(hull)} vertex/vertices. "
    "The protected volume calculations use the convex hull, not the raw point order."
)


# ============================================================
# Run selected cable model
# ============================================================

liftoff_xyz = [liftoff_x, liftoff_y, liftoff_z]
drone_x, drone_y = local_to_world_xy(drone_ground_range_m, 0.0, [liftoff_x, liftoff_y], drone_heading_deg)
drone_z = liftoff_z + drone_height_above_liftoff_m
drone_xyz = [drone_x, drone_y, drone_z]

mass_kg_per_m = cable_mass_kg_per_km / 1000.0
diameter_m = cable_diameter_mm / 1000.0

model_used = None
H_used = H0_N
Ty_used = initial_lateral_tension_N
df_local = None
messages = []

# Suspended trial is used in Auto and Fully suspended.
suspended_local = simulate_suspended_two_point_approx(
    H0_N,
    drone_ground_range_m,
    drone_height_above_liftoff_m,
    mass_kg_per_m,
    diameter_m,
    cd,
    rho,
    drone_speed_mps,
    headwind_mps,
    crosswind_mps,
    ds_m,
)
suspended_world = attach_world_coordinates(suspended_local, liftoff_xyz, drone_heading_deg)
suspended_min_z = float(suspended_world["z_world_m"].min())

if cable_regime == "Fully suspended source-to-drone":
    df_local = suspended_local
    model_used = "fully suspended source-to-drone"
    if tension_mode == "Solve H0 and Ty0 to match drone X/Y endpoint":
        messages.append(
            "Solve-H0/Ty0 is a ground-contact/lift-off option. In fully suspended mode, the app uses the specified H0/source tension."
        )
    if suspended_min_z < ground_z - 1e-6:
        messages.append(
            f"Warning: the fully suspended curve dips below the ground plane. Minimum Z={suspended_min_z:.2f} m, ground Z={ground_z:.2f} m. "
            "This suggests ground contact should be considered."
        )

elif cable_regime == "Auto: suspended first, ground-contact if needed":
    if suspended_min_z >= ground_z - 1e-6:
        df_local = suspended_local
        model_used = "auto → fully suspended"
        if tension_mode == "Solve H0 and Ty0 to match drone X/Y endpoint":
            messages.append(
                "Auto selected the fully suspended model because the source-to-drone cable stays above ground. "
                "The ground-contact H0/Ty0 solve was therefore not used."
            )
    else:
        messages.append(
            f"Auto: suspended curve dips below ground (min Z={suspended_min_z:.2f} m), so switching to ground-contact/lift-off model."
        )
        if tension_mode == "Solve H0 and Ty0 to match drone X/Y endpoint":
            H_used, Ty_used, solved_x, solved_y, df_local, error = solve_H0_Ty0_for_ground_endpoint(
                drone_ground_range_m,
                0.0,  # local final drone y
                drone_height_above_liftoff_m,
                mass_kg_per_m,
                diameter_m,
                cd,
                rho,
                drone_speed_mps,
                headwind_mps,
                crosswind_mps,
                climb_rate_mps,
                ds_m,
                max_s_m,
                include_drag,
                H_min,
                H_max,
                Ty_min,
                Ty_max,
            )
            if error:
                st.error(error)
                st.stop()
        else:
            df_local, reached = simulate_ground_liftoff_local(
                H0_N,
                drone_height_above_liftoff_m,
                mass_kg_per_m,
                diameter_m,
                cd,
                rho,
                drone_speed_mps,
                headwind_mps,
                crosswind_mps,
                climb_rate_mps,
                initial_lateral_tension_N,
                ds_m,
                max_s_m,
                include_drag,
            )
            if not reached:
                st.warning("Ground-contact model did not reach selected drone height within max cable length.")
        model_used = "auto → ground-contact / lift-off"

else:  # Ground-contact / lift-off
    model_used = "ground-contact / lift-off"
    if tension_mode == "Solve H0 and Ty0 to match drone X/Y endpoint":
        H_used, Ty_used, solved_x, solved_y, df_local, error = solve_H0_Ty0_for_ground_endpoint(
            drone_ground_range_m,
            0.0,  # local final drone y
            drone_height_above_liftoff_m,
            mass_kg_per_m,
            diameter_m,
            cd,
            rho,
            drone_speed_mps,
            headwind_mps,
            crosswind_mps,
            climb_rate_mps,
            ds_m,
            max_s_m,
            include_drag,
            H_min,
            H_max,
            Ty_min,
            Ty_max,
        )
        if error:
            st.error(error)
            st.stop()
    else:
        df_local, reached = simulate_ground_liftoff_local(
            H0_N,
            drone_height_above_liftoff_m,
            mass_kg_per_m,
            diameter_m,
            cd,
            rho,
            drone_speed_mps,
            headwind_mps,
            crosswind_mps,
            climb_rate_mps,
            initial_lateral_tension_N,
            ds_m,
            max_s_m,
            include_drag,
        )
        if not reached:
            st.warning("Ground-contact model did not reach selected drone height within max cable length.")

    # Detect common invalidity: ground-contact curve endpoint does not match actual drone x/y.
    if df_local is not None and len(df_local) > 0:
        x_at_height = float(df_local["x_local_m"].iloc[-1])
        y_at_height = float(df_local["y_local_m"].iloc[-1])

        if x_at_height > drone_ground_range_m * 1.02 or x_at_height < drone_ground_range_m * 0.98:
            messages.append(
                f"Warning: ground-contact model reaches drone height at x≈{x_at_height:.1f} m, "
                f"but the drone range is {drone_ground_range_m:.1f} m. "
                "This suggests the specified H0/Ty0 assumption is inconsistent; use endpoint solve or Auto/Fully suspended mode."
            )

        if abs(y_at_height) > max(1.0, 0.02 * drone_ground_range_m):
            messages.append(
                f"Warning: ground-contact model reaches drone height at y≈{y_at_height:.1f} m, "
                "but the drone final local y is 0. "
                "This suggests specified Ty0 is inconsistent; use endpoint solve for crosswind cases."
            )

if df_local is None or len(df_local) < 2:
    st.error("Cable simulation did not produce enough points. Check inputs.")
    st.stop()

df = attach_world_coordinates(df_local, liftoff_xyz, drone_heading_deg)

endpoint_x_local = float(df_local["x_local_m"].iloc[-1])
endpoint_y_local = float(df_local["y_local_m"].iloc[-1])
endpoint_z_local = float(df_local["z_local_m"].iloc[-1])
endpoint_error_x = endpoint_x_local - drone_ground_range_m
endpoint_error_y = endpoint_y_local - 0.0
endpoint_error_z = endpoint_z_local - drone_height_above_liftoff_m


# ============================================================
# Protected volume entry and cutter deployment
# ============================================================

entry_row = find_first_volume_entry(df, hull, protected_z_min, protected_z_max)
cutter = None
result_status = ""
result_message = ""

if entry_row is None:
    result_status = "no_volume_intersection"
    result_message = (
        "The simulated cable path does not enter the protected 3D convex-hull volume. "
        "For this scenario, a cutter is not required to protect this volume."
    )
else:
    entry_s = float(entry_row["s_m"])
    crossings = find_height_crossings_before_s(df, cutter_height_m, entry_s)

    if len(crossings) == 0:
        result_status = "cannot_intercept"
        pre_entry = df[df["s_m"] <= entry_s]
        pre_entry_min_z = float(pre_entry["z_world_m"].min())
        pre_entry_max_z = float(pre_entry["z_world_m"].max())
        result_message = (
            f"Cutter cannot intercept the cable before it enters the protected volume at cutter height Z={cutter_height_m:.2f} m. "
            f"Before volume entry, the cable Z range is approximately {pre_entry_min_z:.2f} m to {pre_entry_max_z:.2f} m."
        )
    else:
        # Choose the last crossing before protected volume entry.
        cutter_row = max(crossings, key=lambda r: float(r["s_m"]))
        path_heading = local_path_direction_at_s(df, float(cutter_row["s_m"]), drone_heading_deg)
        cutter_orientation = (path_heading + 90.0) % 360.0

        cx = float(cutter_row["x_world_m"])
        cy = float(cutter_row["y_world_m"])
        cz = float(cutter_row["z_world_m"])
        p1, p2 = cutter_segment_from_center(cx, cy, cz, cutter_orientation, cutter_length_m)

        cutter = {
            "center": (cx, cy, cz),
            "p1": p1,
            "p2": p2,
            "orientation_deg": cutter_orientation,
            "path_heading_deg": path_heading,
            "row": cutter_row,
            "entry_s": entry_s,
            "standoff_arc_m": entry_s - float(cutter_row["s_m"]),
            "horizontal_distance_to_entry_m": math.hypot(cx - float(entry_row["x_world_m"]), cy - float(entry_row["y_world_m"])),
        }
        result_status = "intercept_possible"
        result_message = (
            f"Intercept possible. Place cutter centre at X={cx:.1f} m, Y={cy:.1f} m, Z={cz:.2f} m. "
            f"Orient cutter at {cutter_orientation:.1f}° where 0°=+X and 90°=+Y."
        )


# ============================================================
# Summary
# ============================================================

st.subheader("Cable model result")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Model used", model_used)
col2.metric("H used", f"{H_used:.3g} N")
col3.metric("Ty0 used", f"{Ty_used:.3g} N")
col4.metric("Min cable Z", f"{float(df['z_world_m'].min()):.2f} m")
col5.metric("End local y", f"{endpoint_y_local:.2f} m")
col6.metric("Cable end Z", f"{float(df['z_world_m'].iloc[-1]):.1f} m")

endpoint_table = pd.DataFrame([
    {"Metric": "Target local X", "Value": f"{drone_ground_range_m:.3f} m"},
    {"Metric": "Cable endpoint local X", "Value": f"{endpoint_x_local:.3f} m"},
    {"Metric": "Endpoint X error", "Value": f"{endpoint_error_x:.3f} m"},
    {"Metric": "Target local Y", "Value": "0.000 m"},
    {"Metric": "Cable endpoint local Y", "Value": f"{endpoint_y_local:.3f} m"},
    {"Metric": "Endpoint Y error", "Value": f"{endpoint_error_y:.3f} m"},
    {"Metric": "Target local Z", "Value": f"{drone_height_above_liftoff_m:.3f} m"},
    {"Metric": "Cable endpoint local Z", "Value": f"{endpoint_z_local:.3f} m"},
    {"Metric": "Endpoint Z error", "Value": f"{endpoint_error_z:.3f} m"},
])
st.dataframe(endpoint_table, hide_index=True, use_container_width=True)

for msg in messages:
    st.warning(msg)

st.subheader("Cutter siting result")
if result_status == "intercept_possible":
    st.success(result_message)
elif result_status == "no_volume_intersection":
    st.info(result_message)
else:
    st.error(result_message)

if entry_row is not None:
    entry_table = pd.DataFrame([
        {"Metric": "Entry X", "Value": f"{float(entry_row['x_world_m']):.2f} m"},
        {"Metric": "Entry Y", "Value": f"{float(entry_row['y_world_m']):.2f} m"},
        {"Metric": "Entry Z", "Value": f"{float(entry_row['z_world_m']):.2f} m"},
        {"Metric": "Entry arc length s", "Value": f"{float(entry_row['s_m']):.2f} m"},
    ])
    st.dataframe(entry_table, hide_index=True, use_container_width=True)

if cutter is not None:
    c = cutter["center"]
    p1 = cutter["p1"]
    p2 = cutter["p2"]
    cutter_table = pd.DataFrame([
        {"Metric": "Cutter centre X", "Value": f"{c[0]:.2f} m"},
        {"Metric": "Cutter centre Y", "Value": f"{c[1]:.2f} m"},
        {"Metric": "Cutter active height Z", "Value": f"{c[2]:.2f} m"},
        {"Metric": "Cutter length", "Value": f"{cutter_length_m:.2f} m"},
        {"Metric": "Cutter orientation", "Value": f"{cutter['orientation_deg']:.1f}°"},
        {"Metric": "Cable local path heading at intercept", "Value": f"{cutter['path_heading_deg']:.1f}°"},
        {"Metric": "Endpoint 1", "Value": f"X={p1[0]:.2f}, Y={p1[1]:.2f}, Z={p1[2]:.2f}"},
        {"Metric": "Endpoint 2", "Value": f"X={p2[0]:.2f}, Y={p2[1]:.2f}, Z={p2[2]:.2f}"},
        {"Metric": "Horizontal distance from cutter to volume entry", "Value": f"{cutter['horizontal_distance_to_entry_m']:.2f} m"},
        {"Metric": "Arc-length standoff before volume entry", "Value": f"{cutter['standoff_arc_m']:.2f} m"},
        {"Metric": "Cross-path tolerance if centred", "Value": f"±{0.5 * cutter_length_m:.2f} m"},
    ])
    st.dataframe(cutter_table, hide_index=True, use_container_width=True)


# ============================================================
# Visualisations
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(["3D visualiser", "Plan view", "Side view", "Height sweep", "Raw data"])

with tab1:
    fig3d = make_3d_plot(df, hull, raw_points, protected_z_min, protected_z_max, liftoff_xyz, drone_xyz, entry_row, cutter, model_used)
    st.plotly_chart(fig3d, use_container_width=True)

with tab2:
    fig_plan = make_plan_plot(df, hull, raw_points, liftoff_xyz, drone_xyz, entry_row, cutter)
    st.plotly_chart(fig_plan, use_container_width=True)

with tab3:
    fig_side = make_side_plot(df, drone_heading_deg, liftoff_xyz, entry_row, cutter, protected_z_min, protected_z_max)
    st.plotly_chart(fig_side, use_container_width=True)

with tab4:
    st.markdown("This sweep checks cutter heights from 0 m to 2 m. For each height, the app finds the last cable crossing before protected volume entry.")

    if entry_row is None:
        st.info("Cable does not enter the protected volume, so there is no required intercept point.")
    else:
        entry_s = float(entry_row["s_m"])
        sweep_rows = []
        for h in np.linspace(0.0, 2.0, 41):
            candidates = find_height_crossings_before_s(df, float(h), entry_s)
            if len(candidates) == 0:
                sweep_rows.append({
                    "cutter_height_m": float(h),
                    "can_intercept_before_volume": False,
                    "recommended_x_m": np.nan,
                    "recommended_y_m": np.nan,
                    "orientation_deg": np.nan,
                    "arc_standoff_before_volume_m": np.nan,
                    "horizontal_distance_to_volume_entry_m": np.nan,
                })
            else:
                row = max(candidates, key=lambda r: float(r["s_m"]))
                ph = local_path_direction_at_s(df, float(row["s_m"]), drone_heading_deg)
                orient = (ph + 90.0) % 360.0
                d_entry = math.hypot(
                    float(row["x_world_m"]) - float(entry_row["x_world_m"]),
                    float(row["y_world_m"]) - float(entry_row["y_world_m"]),
                )
                sweep_rows.append({
                    "cutter_height_m": float(h),
                    "can_intercept_before_volume": True,
                    "recommended_x_m": float(row["x_world_m"]),
                    "recommended_y_m": float(row["y_world_m"]),
                    "orientation_deg": orient,
                    "arc_standoff_before_volume_m": entry_s - float(row["s_m"]),
                    "horizontal_distance_to_volume_entry_m": d_entry,
                })

        sweep_df = pd.DataFrame(sweep_rows)
        st.dataframe(sweep_df, use_container_width=True)

        plot_df = sweep_df[sweep_df["can_intercept_before_volume"]].copy()
        if len(plot_df) > 0:
            fig_sweep = go.Figure()
            fig_sweep.add_trace(go.Scatter(
                x=plot_df["cutter_height_m"],
                y=plot_df["arc_standoff_before_volume_m"],
                mode="lines+markers",
                name="Arc standoff before volume",
            ))
            fig_sweep.add_trace(go.Scatter(
                x=plot_df["cutter_height_m"],
                y=plot_df["horizontal_distance_to_volume_entry_m"],
                mode="lines+markers",
                name="Horizontal distance to volume entry",
            ))
            fig_sweep.update_layout(
                title="Cutter height versus available standoff",
                xaxis_title="Cutter active height Z (m)",
                yaxis_title="Distance (m)",
                height=480,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_sweep, use_container_width=True)
        else:
            st.warning("No cutter height in the 0–2 m sweep intercepts the cable before protected volume entry.")

with tab5:
    st.markdown("### Raw footprint points")
    st.dataframe(pd.DataFrame(raw_points, columns=["x_m", "y_m"]), use_container_width=True)

    st.markdown("### Convex hull vertices used for protected footprint")
    st.dataframe(pd.DataFrame(hull, columns=["x_m", "y_m"]), use_container_width=True)

    st.markdown("### Cable points")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cable points CSV",
        data=csv,
        file_name="cable_points_model_selector_ty0_solve.csv",
        mime="text/csv",
    )


with st.expander("Theory and model selector notes"):
    st.markdown(
        """
### Why a model selector is needed

The old ground-contact model assumed the cable starts from a ground lift-off point with zero vertical tension:
"""
    )

    st.latex(r"""
    \mathbf{r}(0) = [0,0,0],
    \quad
    \mathbf{T}(0) = [H_0,T_{y0},0]
    """)

    st.markdown(
        """
That is only valid if the cable actually touches the ground and then lifts off. For close/high/tensioned drones, the cable may be fully suspended from source to drone. In that regime, forcing a ground-lift-off model can make the cable extend beyond the drone before it reaches the drone height.

### Auto mode

Auto mode first builds a source-to-drone suspended curve. If the suspended curve stays above the ground plane, it uses that. If the suspended curve dips below the ground plane, the app switches to the ground-contact / lift-off model.

### Fully suspended source-to-drone mode

This mode forces the cable to terminate at the drone position. It uses a parabolic approximation:
"""
    )

    st.latex(r"""
    z(u) = h u - \delta_{mid}\,4u(1-u)
    """)

    st.latex(r"""
    \delta_{mid} \approx \frac{wD^2}{8H}
    """)

    st.markdown(
        """
where `D` is drone ground range, `h` is drone height above source, `w` is cable weight per metre, and `H` is the specified horizontal/source tension.

Crosswind creates a lateral bow in the middle of the cable, but the endpoints remain fixed at the source and drone:
"""
    )

    st.latex(r"""
    y(u) = y_{mid}\,4u(1-u)
    """)

    st.markdown(
        """
This is a stable approximation, not a full boundary-value cable solver. A full model would require either paid-out cable length or a more complete tension boundary condition.

### Ground-contact / lift-off mode

This mode uses the original numerical cable equation with gravity and aerodynamic drag:
"""
    )

    st.latex(r"""
    \frac{d\mathbf{r}}{ds}
    =
    \frac{\mathbf{T}}{|\mathbf{T}|}
    """)

    st.latex(r"""
    \frac{d\mathbf{T}}{ds} + \mathbf{f} = 0
    """)

    st.markdown(
        """
with distributed force:
"""
    )

    st.latex(r"""
    \mathbf{f}
    =
    [0,0,-mg]
    -
    \frac{1}{2}\rho C_D d |\mathbf{v}_{\perp}|\mathbf{v}_{\perp}
    """)

    st.markdown(
        """
Use this only when there is likely to be a real ground-contact / lift-off point.

### Ground-contact endpoint solve with crosswind

In crosswind, the cable can bow sideways but should still terminate at the drone. The endpoint solve finds both `H0` and `Ty0` such that:
"""
    )

    st.latex(r"""
    x_{end} = x_{drone},
    \quad
    y_{end} = y_{drone}
    """)

    st.markdown(
        """
The app does this with a nested solve:

1. For a trial `Ty0`, solve `H0` so the cable reaches the target drone range.
2. Adjust `Ty0` until the final lateral endpoint matches the drone local `y=0`.

If the solver cannot bracket the final lateral position, widen the `Ty0` lower/upper bracket.

### Cutter siting logic

The app checks whether the simulated cable enters the protected convex-hull volume. If it does, it finds the last crossing of the selected cutter height before volume entry. The cutter is centred on that crossing and oriented perpendicular to the local horizontal cable path.

### Limitations

This remains a quasi-static engineering visualiser. It does not include cable whipping, gust response, reel friction, ground sliding friction, propwash, impacts, or cutter structural dynamics.
"""
    )
