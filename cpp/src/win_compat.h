#pragma once
// Windows (MSVC) compatibility for VibeBlade C++ backend
// Include this FIRST in every .cpp file that needs it.

#ifdef _WIN32
    // aligned_alloc is C11, not available in MSVC C++17
    #include <malloc.h>
    #define aligned_alloc(alignment, size) _aligned_malloc((size), (alignment))
    #define aligned_free(ptr) _aligned_free(ptr)

    // POSIX file I/O → use io.h / Windows API
    #include <io.h>
    #include <fcntl.h>
    #include <sys/stat.h>
    #define O_RDONLY _O_RDONLY
    #define close _close
    #define open _open
    #define read _read
    #define fstat _fstat64
    #define ssize_t int64_t
    // EINTR already defined in errno.h on Windows

    struct win_stat {
        int64_t st_size;
    };
    // Override struct stat for Windows
    #define stat win_stat
#else
    #include <unistd.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #define aligned_free(ptr) free(ptr)
#endif
