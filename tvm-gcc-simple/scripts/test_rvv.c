// Простой тест на поддержку RVV 0.7.1
#include <stdint.h>
#include <stddef.h>

void vector_add_test() {
    // Минимальный тест, используя точно те же инструкции, что и в автовекторизированном коде
    asm volatile(
        "csrr t0, vlenb\n\t"          // Получить размер вектора в байтах
        "vsetvli a4, zero, e8, m1\n\t" // Установка vector length
        : : : "t0", "a4"
    );
}

int main() {
    vector_add_test();
    return 0;
}
