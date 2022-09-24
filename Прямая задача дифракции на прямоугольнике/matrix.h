#pragma once

#ifndef MATRIX_H
#define MATRIX_H
#include <Windows.h>
#include <iostream>
#include <malloc.h>
#include <fstream>
#include <math.h>


using namespace std;

namespace Matrix_lib {
    template <class t>
    t** createm(size_t M = 3, size_t N = 4) {
        t** var = (t**)malloc(M * sizeof(t*));
        for (int i = 0; i < M; i++)
            var[i] = (t*)malloc(N * sizeof(t));
        return var;
    }
    template <class t>
    t* createv(size_t N = 3) {
        double* var = (t*)malloc(N * sizeof(t));
        return var;
    }

    inline void print(double** var, string c = "") {

        size_t N = _msize(var[0]) / sizeof(var[0][0]);
        size_t M = N - 1;

        HANDLE hConsoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
        for (size_t i = 0; i <= M; i++)
        {
            for (size_t j = 0; j <= N; j++)
            {
                if (i == j) {
                    if (c == "g" || c == "G")SetConsoleTextAttribute(hConsoleHandle, FOREGROUND_GREEN);
                    if (c == "b" || c == "B")SetConsoleTextAttribute(hConsoleHandle, FOREGROUND_BLUE);
                    if (c == "r" || c == "R")SetConsoleTextAttribute(hConsoleHandle, FOREGROUND_RED);
                    if (c == "i" || c == "I")SetConsoleTextAttribute(hConsoleHandle, FOREGROUND_INTENSITY);

                    printf("%4.*f ", 3, var[i][j]);
                    fflush(stdout);
                    SetConsoleTextAttribute(hConsoleHandle, 15);
                }
                else {
                    printf("%4.*f ", 3, var[i][j]);
                    fflush(stdout);
                }

            }
            printf("\n");
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    }

    inline void print(double* var, size_t N = -1) {
        if (N == -1) {
            N = _msize(var) / sizeof(var[0]);
        }
        for (size_t i = 0; i < N; i++)
        {
            printf("%f\n", var[i]);
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    }

    bool print_file(const char Path[], double** var) {
        ofstream file(Path);
        size_t N = _msize(var[0]) / sizeof(var[0][0]);

        if (!file.is_open()) {
            return 0;
        }
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j <= N; j++) {
                file << var[i][j] << " ";
            }
            file << "\n";
        }
        return 1;
    }

    bool read_file(const char Path[], double**& var) {
        ifstream file(Path);
        size_t N = _msize(var[0]) / sizeof(var[0][0]);

        if (!file.is_open()) {
            return 0;
        }
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j <= N; j++) {
                file >> var[i][j];
            }
        }
        return 1;
    }

    void LU(double** var, double**& L, double**& U) {
        size_t N = _msize(var[0]) / sizeof(var[0][0]);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) L[i][i] = 1;
                else L[i][j] = 0;
                U[i][j] = 0;
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double sumU = 0, sumL = 0;
                if (i <= j) {
                    for (int z = 0; z <= i - 1; z++) {
                        sumU += L[i][z] * U[z][j];
                    }
                    U[i][j] = var[i][j] - sumU;
                }

                if (i > j) {
                    for (int z = 0; z <= j - 1; z++) {
                        sumL += L[i][z] * U[z][j];
                    }
                    L[i][j] = (var[i][j] - sumL) / U[j][j];
                }
            }
        }
    }

    /*inline double** mult(double** var1, double** var2) {
        size_t N = _msize(var1[0]) / sizeof(var1[0][0]);
        double** res = createm(N, N);
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                double sum = 0;
                for (size_t k = 0; k < N; k++)
                {
                    sum += var1[i][k] * var2[k][j];
                }
                res[i][j] = sum;
            }
        }
        return res;
    }*/

    template<typename t>
    inline t** mult(t var1, t** var2) {
        size_t N = _msize(var2[0]) / sizeof(var2[0][0]);
        t** res = createm<t>(N, N);
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                res[i][j] = var1 * var2[i][j];
            }
        }
        return res;
    }

    inline double norma(double** var)
    {
        size_t N = _msize(var[0]) / sizeof(var[0][0]);
        double sum = 0;
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                sum += pow(abs(var[i][j]), 2);
            }
        }
        return sqrt(sum);
    }

    double det(double** var) {
        size_t N = _msize(var[0]) / sizeof(var[0][0]);
        double var_Det = 1;
        for (size_t k = 0; k < N; k++) {
            double ed = 1;
            if (var[k][k] != ed) {
                double T = var[k][k];
                var_Det = var_Det * T;
                for (size_t j = k; j < N; j++) {
                    var[k][j] = var[k][j] / T;
                }
            }
            for (size_t i = k; i < N; i++) {
                if ((var[i][k] != ed) && (i != k)) {
                    double T = var[i][k];
                    var[i][k] = 0;
                    for (size_t j = k + 1; j < N; j++) {
                        var[i][j] -= var[k][j] * T;
                    }
                }
            }
        }
        return var_Det;
    }

   /* double** diag(double** var) {
        size_t N = _msize(var[0]) / sizeof(var[0][0]);
        double** var_D = createm(N, N), ** var_L = createm(N, N), ** var_U = createm(N, N);

        LU(var, var_L, var_U);
        var_D = mult(var_U, var_L);

        for (size_t k = 1; k <= 10; k++) {
            LU(var_D, var_L, var_U);
            var_D = mult(var_U, var_L);
        }
        free(var_U); free(var_L);

        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                if (i != j) var_D[i][j] = 0;
            }
        }
        return var_D;
    }*/

    /*double* eigenvalues(double** var) {
        size_t N = _msize(var[0]) / sizeof(var[0][0]);
        double** var_D = diag(var), * res = createv(N);
        for (size_t i = 0; i < N; i++)
            res[i] = var_D[i][i];

        return res;
    }*/

    /*double cond(double** var) {
        size_t N = _msize(var[0]) / sizeof(var[0][0]);
        double* var1 = eigenvalues(var);

        double max = 0, min = 0;
        for (size_t i = 0; i < N; i++)
        {
            if (abs(var1[i]) >= max) max = abs(var1[i]);
            if (abs(var1[i]) <= max) min = abs(var1[i]);
        }
        cout << "Max = " << max << "   Min = " << min << endl;
        return max / min;
    }*/

    template<typename t>
    t** Minus(t** var1, t** var2) {
        size_t N = _msize(var1[0]) / sizeof(var1[0][0]);
        t** res = createm<t>(N, N);

        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                res[i][j] = var1[i][j] - var2[i][j];
            }
        }
        return res;
    }

    template <class t>
    t** Plus(t** var1, t** var2) {
        size_t N = _msize(var1[0]) / sizeof(var1[0][0]);
        t** res = createm<t>(N, N);

        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                res[i][j] = var1[i][j] + var2[i][j];
            }
        }
        return res;
    }
    void gm(double**& var) {
        size_t N = _msize(var[0]) / sizeof(var[0][0]) - 1;

        cout << "N = " << N << endl;

        for (size_t k = 0; k < N; k++) {
            if (k == 0)printf("\nk=%i\n", k);
            if (k != 0)printf("k=%i\n", k);
            fflush(stdout);
            double ed = 1;
            if (var[k][k] != ed) {
                double T = var[k][k];
                for (size_t j = k; j < N + 1; j++) {
                    var[k][j] = var[k][j] / T;
                }
            }
            for (size_t i = k; i < N; i++) {
                if ((var[i][k] != ed) && (i != k)) {
                    double T = var[i][k];
                    var[i][k] = 0;
                    for (size_t j = k + 1; j < N + 1; j++) {
                        var[i][j] -= var[k][j] * T;
                    }
                }
            }
        }
        for (int i = N - 2; i >= 0; i--) {
            double Sum = var[i][N];
            for (size_t j = i + 1; j < N; j++) {
                Sum -= var[i][j] * var[j][N];
            }
            var[i][N] = Sum;
        }
    }
    template <class t>
    t** precond(t** var, int k) {
        size_t N = _msize(var[0]) / sizeof(var[0][0]);
        t** E = createm<t>(N, N);
        t** B = createm<t>(N, N), ** res = createm<t>(N, N + 1);
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                if (i == j) E[i][j] = 1;
                if (i != j) E[i][j] = 0;
            }
        }

        B = Minus<t>(var, E);
        double alpha = 2.0 * k * pow(10, -6);
        res = Plus<complex<double>>(mult<t>(alpha, E), B);

        return res;
    }
    /*void free(double** var) {
        size_t N = _msize(var[0]) / sizeof(var[0][0]);
        for (size_t i = 0; i < N; i++) {
            free(var[i]);
        }
    }*/
}
#endif MATRIX_H