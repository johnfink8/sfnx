// The register layout of Volta's mma.sync.m8n8k4.row.col.f32.f16.f16.f32, measured: for every lane's A slot (4 halves:
// row-major A) a 1 there and B all ones shows the slot's row and quad-pair in D; for every B slot (col-major B) the same
// with A all ones shows its column. Then each lane's 8 accumulators get their (row, column) from those.
//   nvcc -arch=sm_70 -o bin/m884_probe tools/m884_probe.cu && bin/m884_probe
#include <cuda_fp16.h>

#include <cstdio>
#include <cstring>

__global__ void k_mma(const float* a, const float* b, float* d) {
    const int l = threadIdx.x;
    __half2 a01 = __floats2half2_rn(a[l * 4], a[l * 4 + 1]), a23 = __floats2half2_rn(a[l * 4 + 2], a[l * 4 + 3]);
    __half2 b01 = __floats2half2_rn(b[l * 4], b[l * 4 + 1]), b23 = __floats2half2_rn(b[l * 4 + 2], b[l * 4 + 3]);
    float c[8] = {};
    const unsigned A0 = *reinterpret_cast<unsigned*>(&a01), A1 = *reinterpret_cast<unsigned*>(&a23);
    const unsigned B0 = *reinterpret_cast<unsigned*>(&b01), B1 = *reinterpret_cast<unsigned*>(&b23);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(A0), "r"(A1), "r"(B0), "r"(B1));
    for (int j = 0; j < 8; ++j) d[l * 8 + j] = c[j];
}

int main() {
    float *da, *db, *dd;
    cudaMalloc(&da, 128 * 4);
    cudaMalloc(&db, 128 * 4);
    cudaMalloc(&dd, 256 * 4);
    float ha[128], hb[128], hd[256];
    // which D entries (lane, j) light up for each A slot (B all ones: the whole row of the slot's row) / B slot
    int arow[32][4], bcol[32][4];   // per slot: a lit (lane, j) mask id, resolved below
    static unsigned char lit_a[128][256], lit_b[128][256];
    for (int pass = 0; pass < 2; ++pass)
        for (int s = 0; s < 128; ++s) {
            for (int i = 0; i < 128; ++i) ha[i] = pass == 0 ? (i == s) : 1.f, hb[i] = pass == 0 ? 1.f : (i == s);
            cudaMemcpy(da, ha, sizeof ha, cudaMemcpyHostToDevice);
            cudaMemcpy(db, hb, sizeof hb, cudaMemcpyHostToDevice);
            k_mma<<<1, 32>>>(da, db, dd);
            cudaMemcpy(hd, dd, sizeof hd, cudaMemcpyDeviceToHost);
            for (int i = 0; i < 256; ++i) (pass == 0 ? lit_a : lit_b)[s][i] = hd[i] != 0.f;
        }
    // an A slot lights one row of one quad-pair's 8x8 D: 8 entries. Group the slots by the set they light: rows.
    // A D entry (lane, j) is lit by A slots of one row and by B slots of one column.
    int row_of_entry[256], col_of_entry[256], qp_of_entry[256];
    for (int e = 0; e < 256; ++e) row_of_entry[e] = col_of_entry[e] = -1;
    // label rows: the first A slot to light an entry names its row group
    int nrow = 0, ncol = 0;
    int row_id_of_slot[128], col_id_of_slot[128];
    for (int s = 0; s < 128; ++s) {
        int id = -1;
        for (int t = 0; t < s && id < 0; ++t)
            if (!std::memcmp(lit_a[s], lit_a[t], 256)) id = row_id_of_slot[t];
        row_id_of_slot[s] = id < 0 ? nrow++ : id;
        id = -1;
        for (int t = 0; t < s && id < 0; ++t)
            if (!std::memcmp(lit_b[s], lit_b[t], 256)) id = col_id_of_slot[t];
        col_id_of_slot[s] = id < 0 ? ncol++ : id;
        for (int e = 0; e < 256; ++e) {
            if (lit_a[s][e]) row_of_entry[e] = row_id_of_slot[s];
            if (lit_b[s][e]) col_of_entry[e] = col_id_of_slot[s];
        }
    }
    std::printf("distinct A-slot groups (rows over the 4 quad-pairs): %d, B-slot groups (columns): %d\n", nrow, ncol);
    std::printf("A: lane -> its 4 slots' row ids (k = slot index presumably)\n");
    for (int l = 0; l < 32; ++l) {
        std::printf("  lane %2d: A rows %2d %2d %2d %2d | B cols %2d %2d %2d %2d\n", l, row_id_of_slot[l * 4], row_id_of_slot[l * 4 + 1],
                    row_id_of_slot[l * 4 + 2], row_id_of_slot[l * 4 + 3], col_id_of_slot[l * 4], col_id_of_slot[l * 4 + 1],
                    col_id_of_slot[l * 4 + 2], col_id_of_slot[l * 4 + 3]);
    }
    std::printf("C/D: lane -> (row id, col id) of c0..c7\n");
    for (int l = 0; l < 32; ++l) {
        std::printf("  lane %2d:", l);
        for (int j = 0; j < 8; ++j) std::printf(" (%2d,%2d)", row_of_entry[l * 8 + j], col_of_entry[l * 8 + j]);
        std::printf("\n");
    }
    (void) arow, (void) bcol, (void) qp_of_entry;
    return 0;
}
