#include <stdint.h>
#include <stddef.h>

// Функция сложения массивов, которая может быть автоматически векторизована
void autovec_add(int8_t* a, int8_t* b, int8_t* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    return 0;
}
