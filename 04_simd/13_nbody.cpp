#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#include <string.h>
#include <algorithm>

// #define DEBUG

inline void dump(char const *msg, __m256 v) {
#ifdef DEBUG
  float a[8];
  _mm256_store_ps(a, v);
  printf("%s", msg);
  for (int i=0; i<8; ++i) {
    printf("%f, ", a[i]);
  }
  puts("");
#endif
}

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  const int FLOAT_BITS = 32;
  const int XMM_BITS = 256;
  const int STRIDE = XMM_BITS/FLOAT_BITS;

  const __m256 zero = _mm256_setzero_ps();
  const __m256 one = _mm256_set1_ps(1.0);

  for(int i=0; i<N; i++) {
    __m256 fx_acc = _mm256_setzero_ps();
    __m256 fy_acc = _mm256_setzero_ps();

    for(int j=0; j<N; j+=STRIDE) {
      __m256i load_mask;
      {
        int mask[STRIDE] = {};
        int first_invalid_idx = std::min(N - j, STRIDE);
        memset(mask, -1, first_invalid_idx * sizeof(float));
        load_mask = _mm256_load_si256((__m256i*)mask);
      }

      __m256 rx, ry;
      {
        __m256 xi = _mm256_set1_ps(x[i]);
        __m256 yi = _mm256_set1_ps(y[i]);
        __m256 xj = _mm256_maskload_ps(x + j, load_mask);
        dump("        xj : ", xj);
        __m256 yj = _mm256_maskload_ps(y + j, load_mask);
        dump("        yj : ", yj);
        rx = _mm256_sub_ps(xi, xj);
        ry = _mm256_sub_ps(yi, yj);
      }

      __m256 rm3_masked;
      {
        __m256 rx2 = _mm256_mul_ps(rx, rx);
        __m256 ry2 = _mm256_mul_ps(ry, ry);
        __m256 r2 = _mm256_add_ps(rx2, ry2);
        __m256 rm1 = _mm256_rsqrt_ps(r2);
        __m256 rm3 = _mm256_div_ps(rm1, r2);

        dump("       rm3 : ", rm3);

        #define BLEND(M) _mm256_blend_ps(rm3, zero, 1 << (M));
        switch (i - j) {
          case 0: rm3_masked = BLEND(0); break;
          case 1: rm3_masked = BLEND(1); break;
          case 2: rm3_masked = BLEND(2); break;
          case 3: rm3_masked = BLEND(3); break;
          case 4: rm3_masked = BLEND(4); break;
          case 5: rm3_masked = BLEND(5); break;
          case 6: rm3_masked = BLEND(6); break;
          case 7: rm3_masked = BLEND(7); break;
          default: rm3_masked = rm3; break;
        }
        #undef BLEND

        dump("rm3_masked : ", rm3_masked);
      }

      __m256 rx_mj_rm3_masked, ry_mj_rm3_masked;
      {
        __m256 mj = _mm256_maskload_ps(m + j, load_mask);
        dump("        mj : ", mj);

        __m256 rx_mj = _mm256_mul_ps(rx, mj);
        __m256 ry_mj = _mm256_mul_ps(ry, mj);

        rx_mj_rm3_masked = _mm256_mul_ps(rx_mj, rm3_masked);
        ry_mj_rm3_masked = _mm256_mul_ps(ry_mj, rm3_masked);
      }

      fx_acc = _mm256_sub_ps(fx_acc, rx_mj_rm3_masked);
      fy_acc = _mm256_sub_ps(fy_acc, ry_mj_rm3_masked);
    }

    float a[8];
    _mm256_store_ps(a, _mm256_dp_ps(fx_acc, one, 0b11111111));
    fx[i] = a[0] + a[4];
    _mm256_store_ps(a, _mm256_dp_ps(fy_acc, one, 0b11111111));
    fy[i] = a[0] + a[4];

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
