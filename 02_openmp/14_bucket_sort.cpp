#include <cstdio>
#include <cstdlib>
#include <vector>

inline static void scan(std::vector<int> &v, int const len) {
  std::vector<int> cpy(v);
#pragma omp parallel
  for (int j=1; j<len; j<<=1) {
#pragma omp for
    for(int i=0; i<len; i++)
      cpy[i] = v[i];
#pragma omp for
    for (int i=j; i<len; i++)
      v[i] += cpy[i-j];
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0);
#pragma omp parallel for
  for (int i=0; i<n; i++)
#pragma omp atomic update
    bucket[key[i]]++;

  std::vector<int> offset(range,0);
#pragma opm parallel for
  for (int i=1; i<range; i++)
    offset[i] = bucket[i-1];

  scan(offset, range);

#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
