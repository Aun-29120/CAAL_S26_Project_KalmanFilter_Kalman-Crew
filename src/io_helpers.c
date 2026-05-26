/* ================================================================
 * io_helpers.c — CSV I/O helpers called from RISC-V assembly
 *
 * Provides:  read_csv, open_output, write_row, close_output
 * These are the ONLY C functions used. All math is in assembly.
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 16384

/* ----------------------------------------------------------------
 * double* read_csv(const char* path, int* out_rows, int* out_cols)
 *
 * Reads a CSV of doubles (with header). Returns flat row-major
 * array of size rows*cols.  Stores row/col counts via pointers.
 * ---------------------------------------------------------------- */
double* read_csv(const char* path, int* out_rows, int* out_cols) {
    FILE* fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    char line[MAX_LINE];

    /* --- Count columns from header --- */
    fgets(line, MAX_LINE, fp);
    int cols = 1;
    for (char* p = line; *p; p++)
        if (*p == ',') cols++;

    /* --- Count data rows --- */
    int rows = 0;
    while (fgets(line, MAX_LINE, fp)) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        rows++;
    }
    rewind(fp);
    fgets(line, MAX_LINE, fp);  /* skip header again */

    /* --- Allocate and read --- */
    double* data = (double*)malloc(rows * cols * sizeof(double));
    for (int r = 0; r < rows; r++) {
        fgets(line, MAX_LINE, fp);
        char* p = line;
        for (int c = 0; c < cols; c++) {
            data[r * cols + c] = strtod(p, &p);
            if (*p == ',') p++;
        }
    }
    fclose(fp);

    *out_rows = rows;
    *out_cols = cols;
    return data;
}

/* ----------------------------------------------------------------
 * FILE* open_output(const char* path)
 * ---------------------------------------------------------------- */
FILE* open_output(const char* path) {
    FILE* fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "Cannot write %s\n", path); exit(1); }

    /* Write header: frame, then per-joint state names */
    static const char* joints[] = {
        "pelvis","L5","L3","T12","T8","neck","head",
        "shoulderRight","upperArmRight","forearmRight","handRight",
        "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
        "upperLegRight","lowerLegRight","footRight","toeRight",
        "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
    };
    fprintf(fp, "frame");
    for (int j = 0; j < 23; j++) {
        const char* n = joints[j];
        fprintf(fp, ",%s_px,%s_vx,%s_ax,%s_jx", n, n, n, n);
        fprintf(fp, ",%s_py,%s_vy,%s_ay,%s_jy", n, n, n, n);
        fprintf(fp, ",%s_pz,%s_vz,%s_az,%s_jz", n, n, n, n);
    }
    fprintf(fp, "\n");
    return fp;
}

/* ----------------------------------------------------------------
 * void write_row(FILE* fp, double* x, int n)
 *
 * Writes one CSV row: frame_counter, x[0], x[1], ..., x[n-1]
 * Frame counter auto-increments.
 * ---------------------------------------------------------------- */
static int g_frame = 0;
void write_row(FILE* fp, double* x, int n) {
    fprintf(fp, "%d", g_frame++);
    for (int i = 0; i < n; i++)
        fprintf(fp, ",%.15g", x[i]);
    fprintf(fp, "\n");
}

/* ----------------------------------------------------------------
 * void close_output(FILE* fp)
 * ---------------------------------------------------------------- */
void close_output(FILE* fp) {
    fclose(fp);
}

/* Reset frame counter (call before second filter, e.g. EKF) */
void reset_frame_counter(void) {
    g_frame = 0;
}

/* ----------------------------------------------------------------
 * double* read_state_csv(const char* path, int* out_rows)
 *
 * Reads a state CSV (frame,val0,val1,...,val275).
 * Skips the frame column. Returns flat array of rows*276 doubles.
 * Returns NULL if file cannot be opened.
 * ---------------------------------------------------------------- */
double* read_state_csv(const char* path, int* out_rows) {
    FILE* fp = fopen(path, "r");
    if (!fp) return NULL;

    char line[65536];
    fgets(line, sizeof(line), fp);  /* skip header */

    int rows = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        rows++;
    }
    rewind(fp);
    fgets(line, sizeof(line), fp);

    double* data = (double*)malloc(rows * 276 * sizeof(double));
    for (int r = 0; r < rows; r++) {
        fgets(line, sizeof(line), fp);
        char* p = line;
        strtod(p, &p);  /* skip frame number */
        if (*p == ',') p++;
        for (int c = 0; c < 276; c++) {
            data[r * 276 + c] = strtod(p, &p);
            if (*p == ',') p++;
        }
    }
    fclose(fp);
    *out_rows = rows;
    return data;
}

/* ----------------------------------------------------------------
 * void write_all_states(FILE* fp, double* states, int T, int N)
 *
 * Writes T rows of N doubles each. states is flat T*N array.
 * ---------------------------------------------------------------- */
void write_all_states(FILE* fp, double* states, int T, int N) {
    for (int t = 0; t < T; t++) {
        fprintf(fp, "%d", t);
        for (int i = 0; i < N; i++)
            fprintf(fp, ",%.15g", states[t * N + i]);
        fprintf(fp, "\n");
    }
}
