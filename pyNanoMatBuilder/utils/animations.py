import os
from pathlib import Path

from .core import (pyNMB_location, get_resource_path, timer, RAB, Rbetween2Points,
                   vector, vectorBetween2Points, coord2xyz, vertex, vertexScaled, RadiusSphereAfterV,
                   centerOfGravity, center2cog, normOfV, normV, centerToVertices, Rx, Ry, Rz,
                   EulerRotationMatrix, plotPalette, rgb2hex, clone, deleteElementsOfAList,
                   planeFittingLSF, AngleBetweenVV, signedAngleBetweenVV
                   )
from .core import centertxt, centerTitle, fg, bg, hl, color

from .io import read
from .external_pgm import saveCoords_DrawJmol

def render_frames_jmol(prefix=None, n_frames=None, view_script="", output_dir=None,
                       xyz_files=None, input_dir=None, start=0, noOutput=True,
                       **jmol_kwargs):
    """
    Render a series of .xyz frames into .png images via saveCoords_DrawJmol.

    Two mutually exclusive modes:

    1. Prefix mode: provide `prefix` and `n_frames`. Reads frames named
       '<prefix><i>.xyz' (i zero-padded to 2 digits) from input_dir.
    2. File-list mode: provide `xyz_files`, an explicit ordered list of .xyz
       paths. The PNG stem is taken from each file's own stem, so naming need
       not follow the zero-padded convention.

    Args:
        prefix (str, optional): Common frame name prefix (prefix mode).
        n_frames (int, optional): Number of frames to render (prefix mode).
        view_script (str): Jmol script string controlling the camera/view.
        output_dir (str): Directory where the .png files are written.
        xyz_files (list[str|Path], optional): Explicit ordered list of .xyz
            files to render (file-list mode). Takes precedence over prefix mode.
        input_dir (str, optional): Directory holding the source .xyz frames
            (prefix mode only). Defaults to output_dir.
        start (int, optional): Index of the first frame (prefix mode, default 0).
        noOutput (bool, optional): If True, suppresses output (default True).
        **jmol_kwargs: Extra arguments forwarded to saveCoords_DrawJmol.

    Returns:
        list[str]: Paths of the rendered .png files, in frame order. Frames
            whose .xyz source is missing are skipped (with a warning).

    Note:
        - saveXYZ is forced to False (we only want the .png output here).
        - In prefix mode, frame naming must match what was produced by `write`.
    """
    if output_dir is None:
        raise ValueError("output_dir is required.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build the ordered list of (frame_prefix, xyz_path) to render
    frames = []
    if xyz_files is not None:
        base = Path(input_dir) if input_dir is not None else Path(".")
        for f in xyz_files:
            p = Path(f)
            if not p.is_absolute() and p.parent == Path("."):
                p = base / p          # resolve bare names against input_dir
            frames.append((p.stem, p))
    elif prefix is not None and n_frames is not None:
        in_dir = Path(input_dir) if input_dir is not None else Path(output_dir)
        for i in range(start, start + n_frames):
            fp = f"{prefix}{i:02d}"
            frames.append((fp, in_dir / f"{fp}.xyz"))
    else:
        raise ValueError("Provide either `xyz_files`, or both `prefix` and "
                         "`n_frames`.")

    png_paths = []
    for frame_prefix, xyz_path in frames:
        if not xyz_path.exists():
            print(f"{fg.RED}Warning: frame '{xyz_path}' not found — "
                  f"skipped.{fg.OFF}")
            continue

        if not noOutput:
            print(f"frame {frame_prefix}")

        atoms = read(xyz_path)
        saveCoords_DrawJmol(atoms,
                            prefix=frame_prefix,
                            scriptJ=view_script,
                            noOutput=noOutput,
                            user_output_dir=str(output_dir),
                            saveXYZ=True,
                            **jmol_kwargs)

        png_path = Path(output_dir) / f"{frame_prefix}.png"
        if png_path.exists():
            png_paths.append(str(png_path))
        else:
            print(f"{fg.RED}Warning: expected PNG '{png_path}' was not "
                  f"produced by saveCoords_DrawJmol — skipped.{fg.OFF}")

    if not png_paths:
        print(f"{fg.RED}Warning: no PNG frames were rendered.{fg.OFF}")

    return png_paths

def frames_to_movie(png_paths, output_file, fps=5, pingpong=False,
                    pingpong_hold_ends=False):
    """
    Assemble a list of PNG images into an animated movie (.mp4 or .gif).

    The output format is chosen from the output_file extension.

    Args:
        png_paths (list[str]): Ordered paths of the PNG frames to assemble.
        output_file (str): Output path; '.mp4' or '.gif' extension selects the
            format.
        fps (int, optional): Frames per second (default 5).
        pingpong (bool, optional): If True, plays the sequence forward then
            backward (excluding the duplicated endpoints) for a seamless loop
            (default False).
        pingpong_hold_ends (bool, optional): Only relevant when pingpong is
            True. Controls how the endpoints are treated on the return pass:
              - False (default): clean bounce — endpoints are not repeated.
                For 5 frames the cycle is 0,1,2,3,4,3,2,1, so the first and
                last frames are shown once per cycle (no pause at the extremes).
              - True: the endpoints are duplicated on the turnaround
                (0,1,2,3,4,4,3,2,1,0), producing a short hold at each extreme.

    Returns:
        str or None: The output path on success, or None if nothing was written.

    Note:
        - Requires the 'imageio' package. If absent, a warning is printed and
          None is returned.
        - Missing PNG files in png_paths are skipped with a warning.
    """
    try:
        import imageio.v2 as iio
    except ImportError:
        print(f"{fg.RED}Warning: 'imageio' is not installed — "
              f"cannot write '{output_file}'. "
              f"Install it with: pip install imageio{fg.OFF}")
        return None

    if not png_paths:
        print(f"{fg.RED}Warning: empty frame list — "
              f"'{output_file}' not written.{fg.OFF}")
        return None

    extension = Path(output_file).suffix.lower()
    if extension not in (".mp4", ".gif"):
        print(f"{fg.RED}Warning: unsupported output extension '{extension}' — "
              f"use '.mp4' or '.gif'. '{output_file}' not written.{fg.OFF}")
        return None

    # Build the playback order
    indices = list(range(len(png_paths)))
    if pingpong and len(png_paths) > 2:
        if pingpong_hold_ends:
            # Hold BOTH endpoints when looping.
            # Cycle = 0,1,2,3,4,4,3,2,1,0 ; looped -> ...,1,0,0,1,...
            # so 4,4 holds mid-cycle and 0,0 holds across the loop seam.
            n = len(png_paths)
            indices = list(range(n)) + list(range(n - 1, -1, -1))
        else:
            # Clean bounce, endpoints not repeated: 0,1,2,3,4,3,2,1
            # -> uniform pacing, no pause at the extremes.
            indices += list(range(len(png_paths) - 2, 0, -1))

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # GIF needs an explicit loop=0 to repeat indefinitely; mp4 doesn't take it
    writer_kwargs = {"fps": fps}
    if extension == ".gif":
        writer_kwargs["loop"] = 0

    writer = iio.get_writer(output_file, **writer_kwargs)
    
    n_written = 0
    try:
        for idx in indices:
            png = png_paths[idx]
            if not os.path.exists(png):
                print(f"{fg.RED}Warning: frame '{png}' not found — "
                      f"skipped.{fg.OFF}")
                continue
            writer.append_data(iio.imread(png))
            n_written += 1
    finally:
        writer.close()

    if n_written == 0:
        print(f"{fg.RED}Warning: no frames written to "
              f"'{output_file}'.{fg.OFF}")
        return None

    print(f"{fg.GREEN}{output_file} created "
          f"({n_written} frames @ {fps} fps).{fg.OFF}")
    return output_file
