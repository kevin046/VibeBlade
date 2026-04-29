"""Build VibeBlade native C++ backend — cross-platform (Windows + Linux + macOS).

Usage:
    python build_cpp.py [Release|Debug]

Requires: cmake, C++17 compiler, pybind11, numpy
"""
import os
import subprocess
import sys
import glob
import shutil


def main():
    build_type = sys.argv[1] if len(sys.argv) > 1 else "Release"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(script_dir, "build")
    project_dir = os.path.dirname(script_dir)
    pkg_dir = os.path.join(project_dir, "vibeblade")

    python_exe = sys.executable

    print()
    print("  VibeBlade Native Backend Build")
    print(f"  Build type:  {build_type}")
    print(f"  Python:      {python_exe}")
    print(f"  Project:     {project_dir}")
    print()

    os.makedirs(build_dir, exist_ok=True)

    # Get pybind11 cmake dir
    try:
        pybind_dir = subprocess.check_output(
            [python_exe, "-c",
             "import pybind11, os; "
             "print(os.path.join(os.path.dirname(pybind11.__file__), "
             "'share', 'cmake', 'pybind11'))"],
            text=True
        ).strip()
    except Exception as e:
        print(f"  ERROR: pybind11 not found: {e}")
        print("  Install: pip install pybind11")
        sys.exit(1)

    # Configure
    print("  Configuring...")
    cmake_args = [
        "cmake", "..",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DPython3_EXECUTABLE={python_exe}",
        "-DPYBIND11_FINDPYTHON=ON",
        f"-Dpybind11_DIR={pybind_dir}",
    ]

    # Windows: use default generator (Visual Studio or Ninja)
    # Linux/macOS: use Unix Makefiles
    if sys.platform != "win32":
        cmake_args.append("-G")
        cmake_args.append("Unix Makefiles")

    result = subprocess.run(cmake_args, cwd=build_dir)
    if result.returncode != 0:
        print("  ERROR: cmake configure failed")
        sys.exit(1)

    # Build
    print()
    print("  Building...")
    nproc = os.cpu_count() or 4
    build_cmd = ["cmake", "--build", ".", "--config", build_type, "-j", str(nproc)]
    result = subprocess.run(build_cmd, cwd=build_dir)
    if result.returncode != 0:
        print("  ERROR: build failed")
        sys.exit(1)

    # Find the compiled module
    patterns = [
        os.path.join(build_dir, "_vibeblade_native*.so"),
        os.path.join(build_dir, "_vibeblade_native*.pyd"),
        os.path.join(build_dir, "**", "_vibeblade_native*.so"),
        os.path.join(build_dir, "**", "_vibeblade_native*.pyd"),
    ]

    so_file = None
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            so_file = matches[0]
            break

    if not so_file:
        print("  ERROR: no .so/.pyd found after build")
        sys.exit(1)

    # Copy to vibeblade package
    ext = os.path.splitext(so_file)[1]
    dest = os.path.join(pkg_dir, f"_vibeblade_native{ext}")
    shutil.copy2(so_file, dest)
    print(f"  Copied {so_file} -> {dest}")

    # Verify import
    print()
    print("  Verifying import...")
    test_code = (
        "import sys; sys.path.insert(0, r'{project_dir}'); "
        "import vibeblade._vibeblade_native as nat; "
        "print(f'  SIMD: {nat.SIMD_BACKEND}'); "
        "print(f'  OK')"
    ).format(project_dir=project_dir)
    result = subprocess.run([python_exe, "-c", test_code])
    if result.returncode != 0:
        print("  ERROR: import verification failed")
        sys.exit(1)

    print()
    print("  Native backend ready!")


if __name__ == "__main__":
    main()
