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

def render_frames_jmol(prefix, n_frames, view_script, output_dir,
                       input_dir=None, start=0, noOutput=True, **jmol_kwargs):
    """
    Render a series of .xyz frames into .png images via saveCoords_DrawJmol.

    Reads frames named '<prefix><i>.xyz' (with i zero-padded to 2 digits) from
    input_dir, and renders each one to '<output_dir>/<prefix><i>.png' using the
    given Jmol view script.

    Args:
        prefix (str): Common frame name prefix, e.g. 'Aufcc_c_Oh_'. The frame
            index is appended zero-padded to 2 digits ('Aufcc_c_Oh_00', ...).
        n_frames (int): Number of frames to render (indices start..start+n_frames-1).
        view_script (str): Jmol script string controlling the camera/view,
            passed to saveCoords_DrawJmol as scriptJ.
        output_dir (str): Directory where the .png files are written
            (passed to saveCoords_DrawJmol as user_output_dir).
        input_dir (str, optional): Directory holding the source .xyz frames.
            Defaults to output_dir.
        start (int, optional): Index of the first frame (default 0).
        noOutput (bool, optional): If True, suppresses saveCoords_DrawJmol
            output (default True).
        **jmol_kwargs: Extra arguments forwarded to saveCoords_DrawJmol
            (e.g. cpk=1.7, boundaries=False).

    Returns:
        list[str]: Paths of the rendered .png files, in frame order. Frames
            whose .xyz source is missing are skipped (with a warning) and
            absent from the returned list.

    Note:
        - saveXYZ is forced to False (we only want the .png output here).
        - Frame naming must match what was produced by `write`, i.e. the same
          zero-padded 2-digit convention.
    """
    if input_dir is None:
        input_dir = output_dir

    input_dir = Path(input_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    png_paths = []
    for i in range(start, start + n_frames):
        frame_prefix = f"{prefix}{i:02d}"
        xyz_path = input_dir / f"{frame_prefix}.xyz"

        if not xyz_path.exists():
            print(f"{fg.RED}Warning: frame '{xyz_path}' not found — "
                  f"skipped.{fg.OFF}")
            continue

        if not noOutput:
            print(f"frame {i:02d}")

        atoms = read(xyz_path)
        saveCoords_DrawJmol(atoms,
                            prefix=frame_prefix,
                            scriptJ=view_script,
                            noOutput=noOutput,
                            user_output_dir=str(output_dir),
                            saveXYZ=False,
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


def frames_to_movie(png_paths, output_file, fps=5, pingpong=False):
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

    # extension = Path(output_file).suffix.lower()
    # if extension not in (".mp4", ".gif"):
    #     print(f"{fg.RED}Warning: unsupported output extension '{extension}' — "
    #           f"use '.mp4' or '.gif'. '{output_file}' not written.{fg.OFF}")
    #     return None

    # # Build the playback order
    # indices = list(range(len(png_paths)))
    # if pingpong and len(png_paths) > 2:
    #     indices += list(range(len(png_paths) - 2, 0, -1))

    # Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # writer = iio.get_writer(output_file, fps=fps)
    extension = Path(output_file).suffix.lower()
    if extension not in (".mp4", ".gif"):
        print(f"{fg.RED}Warning: unsupported output extension '{extension}' — "
              f"use '.mp4' or '.gif'. '{output_file}' not written.{fg.OFF}")
        return None

    # Build the playback order
    indices = list(range(len(png_paths)))
    if pingpong and len(png_paths) > 2:
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