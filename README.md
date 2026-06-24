# Drone fibre-optic cable sag model with convex-hull protected volume

This Streamlit app models a drone fibre-optic cable and recommends a cutter deployment location and orientation.

## Features

- Add raw ground points for the protected asset.
- Automatically form the protected footprint using the convex hull of the raw points.
- Show raw points and the convex-hull footprint in plan and 3D views.
- Extrude the convex hull into a transparent protected 3D volume.
- Define drone lift-off point, range, height, heading, speed and wind.
- Define cable linear mass, diameter, drag coefficient and tension model.
- Define cutter active height and active length.
- Compute whether the cable enters the protected volume.
- If it does, recommend cutter centre, endpoint coordinates and ground orientation.
- If it cannot intercept, explicitly state why.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Coordinate convention

- X, Y, Z are metres.
- Z is height.
- Heading 0° = +X.
- Heading 90° = +Y.

## Convex hull workflow

Use the **Quick add point** panel or edit the table directly. You do not need to enter points in order. The app wraps the smallest convex polygon around all valid points and uses that hull as the protected footprint.

Interior points are still plotted as raw points, but they do not change the convex hull.

## Siting logic

If the cable enters the protected 3D volume, the app finds the last crossing of the selected cutter height before volume entry. It places the cutter centre at that crossing and orients the cutter perpendicular to the local horizontal cable direction.

If the cable never enters the protected volume, the app explicitly states that no cutter is required for that scenario.

If the cable enters the protected volume but never crosses the selected cutter height before entry, the app explicitly states that the cutter cannot intercept at that height.
