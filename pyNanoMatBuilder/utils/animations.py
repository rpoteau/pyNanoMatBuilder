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

def frames_to_movie(png_paths, output_file, fps=25, seconds_per_frame=1.0,
                    pingpong=False, pingpong_hold_ends=False):
    """
    Assemble a list of PNG images into an animated movie (.mp4 or .gif).

    Each source image is held on screen for `seconds_per_frame` seconds by
    repeating it (fps * seconds_per_frame) times in the output stream. Encoding
    at a standard frame rate (e.g. 25 fps) with repeated frames, rather than at
    a very low frame rate such as 1 fps, produces a video that plays reliably in
    browser-based viewers such as PowerPoint for the web, which handle very low
    frame rates poorly and may drop frames in slideshow mode.

    The output format is chosen from the output_file extension.

    Args:
        png_paths (list[str]): Ordered paths of the PNG frames to assemble.
        output_file (str): Output path; '.mp4' or '.gif' extension selects the
            format.
        fps (int, optional): Frames per second of the output stream (default 25).
            Kept standard for browser compatibility; the perceived pace is set
            by seconds_per_frame, not by fps.
        seconds_per_frame (float, optional): On-screen duration of each source
            image, in seconds (default 1.0). Each image is repeated
            round(fps * seconds_per_frame) times in the stream.
        pingpong (bool, optional): If True, plays the sequence forward then
            backward for a seamless loop (default False).
        pingpong_hold_ends (bool, optional): Only relevant when pingpong is
            True. If True, the endpoints are duplicated on the turnaround
            (0,1,2,3,4,4,3,2,1,0), producing a short hold at each extreme; if
            False, a clean bounce is used (0,1,2,3,4,3,2,1).

    Returns:
        str or None: The output path on success, or None if nothing was written.

    Note:
        - Requires the 'imageio' package (and 'imageio-ffmpeg' for .mp4).
        - Missing PNG files in png_paths are skipped with a warning.
        - The MP4 branch forces a PowerPoint-friendly encoding: H.264 (libx264),
          yuv420p pixel format, even dimensions, and every frame as a keyframe.
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

    # Build the playback order of the SOURCE images
    indices = list(range(len(png_paths)))
    if pingpong and len(png_paths) > 2:
        if pingpong_hold_ends:
            n = len(png_paths)
            indices = list(range(n)) + list(range(n - 1, -1, -1))
        else:
            indices += list(range(len(png_paths) - 2, 0, -1))

    # Number of times each source image is repeated so it stays on screen for
    # seconds_per_frame at the chosen (standard) fps.
    repeat = max(1, int(round(fps * seconds_per_frame)))

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    writer_kwargs = {"fps": fps}
    if extension == ".gif":
        writer_kwargs["loop"] = 0
    else:
        # MP4: PowerPoint-friendly encoding. yuv420p previews and plays
        # everywhere; libx264 is the standard codec; the scale filter forces
        # even dimensions (required by yuv420p); -g 1 makes every frame a
        # keyframe, so no frame can be dropped for lack of a reference; and
        # +faststart lets players begin playback immediately.
        writer_kwargs["codec"] = "libx264"
        writer_kwargs["pixelformat"] = "yuv420p"
        writer_kwargs["output_params"] = [
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-g", "1",
            "-movflags", "+faststart",
        ]

    writer = iio.get_writer(output_file, **writer_kwargs)

    n_written = 0
    try:
        for idx in indices:
            png = png_paths[idx]
            if not os.path.exists(png):
                print(f"{fg.RED}Warning: frame '{png}' not found — "
                      f"skipped.{fg.OFF}")
                continue
            img = iio.imread(png)
            for _ in range(repeat):          # hold this image on screen
                writer.append_data(img)
                n_written += 1
    finally:
        writer.close()

    if n_written == 0:
        print(f"{fg.RED}Warning: no frames written to "
              f"'{output_file}'.{fg.OFF}")
        return None

    duration = n_written / float(fps)
    print(f"{fg.GREEN}{output_file} created "
          f"({n_written} frames @ {fps} fps, {duration:.1f} s).{fg.OFF}")
    return output_file