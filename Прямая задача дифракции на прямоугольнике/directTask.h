#pragma once
#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <mpi.h>
#include "matrix.h"

using namespace std;

class DirectTaskDifraction
{
public:
    DirectTaskDifraction(int N, double Hz, complex<double>(* KerFunc)(double, double, double, double, double), complex<double>(* fallWave)(double, double, double), double(* K_f)(double, double));
	~DirectTaskDifraction();

    template<typename Type>
    void CreateMatrixMemory(int I, int J, Type**& A);

    template<typename Type>
    void DeleteMatrixMemory(int I, Type** A);

    int get_height_by_rank(int rank, int size, int N);

    int get_rank_by_index(int _index, int _size, int _N);

    void Gauss(complex <double>** _Matrix, complex <double>* _Vec, int _Nm, int _rank, int _size, int _height, int _stride);

    complex <double> Integr(double x_beg, double x_end, double y_beg, double y_end, double koll_x, double koll_y, double K);

    complex <double> Integr_for_reverse(double x_beg, double x_end, double y_beg, double y_end, double koll_x, double koll_y, double K);

    complex<double> getModByX_Y(double _x, double _y);

    void set_border(double A, double B, double C, double D);
    void fill_massive();

    complex<double>* alpha_beta_vec = nullptr;
    double h_x = 0, h_y = 0;

private:
	double v_l = 3. * pow(10, 10);
    double K0 = 0, A = 0, B = 0, C = 0, D = 0;
    double Pi = acos(-1);
    int n = 0;
    double (*K_f)(double, double);
    complex<double>(*Kernel)(double, double, double, double, double);
    complex<double>(*fallWave)(double, double, double);
};

DirectTaskDifraction::DirectTaskDifraction(int N, double Hz,
    complex<double>(*KerFunc)(double, double, double, double, double),
    complex<double>(*fallWave)(double, double, double),
    double (*K_f)(double, double))
{
    this->n = N;
    this->K0 = 2.0 * this->Pi / this->v_l * Hz;
    this->Kernel = KerFunc;
    this->K_f = K_f;
    this->fallWave = fallWave;
}

DirectTaskDifraction::~DirectTaskDifraction()
{
}

template<typename Type>
void DirectTaskDifraction::CreateMatrixMemory(int I, int J, Type**& A)
{
    int i1, i2;
    A = new Type * [I];
    for (i1 = 0; i1 < I; i1++) {
        A[i1] = new Type[J];
        for (i2 = 0; i2 < J; i2++) {
            A[i1][i2] = 0.0;
        }
    }
}

//Освобождает память (количество строк)
template<typename Type>
void DirectTaskDifraction::DeleteMatrixMemory(int I, Type** A)
{
    int i1;
    for (i1 = 0; i1 < I; i1++) {
        delete A[i1];
    }
    delete[]A;
}

int DirectTaskDifraction::get_height_by_rank(int rank, int size, int N) {
    int height = N / size;

    if (rank >= size - N % size) height++;
    return height;
}

int DirectTaskDifraction::get_rank_by_index(int _index, int _size, int _N) {
    int check_index_1 = 0;
    int check_index_2 = 0;
    for (int i = 0; i < _size; i++)
    {
        check_index_2 += this->get_height_by_rank(i, _size, _N);
        if (check_index_1 <= _index && _index < check_index_2) return i;
        check_index_1 = check_index_2;
    }
    return -1;
}

void DirectTaskDifraction::Gauss(complex <double>** _Matrix, complex <double>* _Vec, int _Nm, int _rank, int _size, int _height, int _stride) {
    complex <double> ed(1.0, 0.0);
    complex <double> nul(0.0, 0.0);

    complex<double>* tmp_row = new complex<double>[_Nm];
    complex<double> tmp_elem;

    MPI_Status stat;

    for (int k = 0; k < _Nm; k++) {
        if (_rank == 0) {
            cout << "k = " << k << endl;
            fflush(stdout);
        }

        int local_k = k - _stride;                                   //приводим к нужной индексации
        if (_stride <= k && k < _stride + _height) {       //если строка k принадлежит процессу 

            if (_Matrix[local_k][k] != ed) {
                complex <double> T = _Matrix[local_k][k];
                for (int j = k; j < _Nm; j++) {          //нормирование строки
                    _Matrix[local_k][j] = _Matrix[local_k][j] / T;
                }
                _Vec[local_k] = _Vec[local_k] / T;
            }
            for (int r = _rank + 1; r < _size; r++) {            //рассылаем строку

                //cout << "Begin_send to rank = " << r << endl;
                //fflush(stdout);
                MPI_Ssend(_Matrix[local_k], _Nm, MPI_DOUBLE_COMPLEX, r, 1, MPI_COMM_WORLD);

                MPI_Ssend(&_Vec[local_k], 1, MPI_DOUBLE_COMPLEX, r, 2, MPI_COMM_WORLD);

            }

            for (int i = local_k + 1; i < _height; i++) { //проходим по столбцу
                if (_Matrix[i][k] != ed) {
                    complex <double> T = _Matrix[i][k];
                    _Matrix[i][k] = 0;
                    for (int j = k + 1; j < _Nm; j++) { //проходим по двум строкам и вычитаем их
                        _Matrix[i][j] -= _Matrix[local_k][j] * T;
                    }
                    _Vec[i] -= _Vec[local_k] * T;
                }
            }
        }
        else {
            if (_stride >= k) {     //проверить

                MPI_Recv(tmp_row, _Nm, MPI_DOUBLE_COMPLEX, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &stat);

                MPI_Recv(&tmp_elem, 1, MPI_DOUBLE_COMPLEX, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &stat);
                fflush(stdout);
                for (int i = 0; i < _height; i++) { //проходим по столбцу
                    if (_Matrix[i][k] != ed) {
                        complex <double> T = _Matrix[i][k];
                        _Matrix[i][k] = 0;
                        for (int j = k + 1; j < _Nm; j++) { //проходим по двум строкам и вычитаем их
                            _Matrix[i][j] -= tmp_row[j] * T;
                        }
                        _Vec[i] -= tmp_elem * T;
                    }
                }
            }
        }
    }

    int variables = _Nm - _stride - _height;    //столько переменных недоступно в каждом матрице (в последней = 0)

    //cout << "rank = " << _rank << " var = " << variables << endl;

    for (int ind_x = 0; ind_x < variables; ind_x++) {   //тут принимаем все недостающие переменные для того, чтобы выполнить обратный ход в этой части матрицы
        complex <double> recv_x;
        MPI_Recv(&recv_x, 1, MPI_DOUBLE_COMPLEX, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &stat);
        for (int i = 0; i < _height; i++) {     //отнимаем принятое значение, умноженное на данный столбец, от вектора правой части
            int j = _Nm - ind_x - 1;
            _Vec[i] -= _Matrix[i][j] * recv_x;
        }
    }

    for (int i = _height - 1; i >= 0; i--) {
        int ind_d = _stride + i;                    //столбец диагонального элемента
        for (int j = _Nm - variables - 1; j > ind_d; j--) {     //если убрать variables то будет непостоянная ошибка (любопытно)
            //работаем со столбцами, от диагонального до блока с неизвестными коэффициентами (потому что при принятии уже сделали все необходимые операции)
            _Vec[i] -= _Vec[j - _stride] * _Matrix[i][j];
        }
        for (int r = 0; r < _rank; r++) {           //рассылаем всем кто выше по матрице
            MPI_Ssend(&_Vec[i], 1, MPI_DOUBLE_COMPLEX, r, 3, MPI_COMM_WORLD);
        }
    }

    delete[]tmp_row;
}

complex <double> DirectTaskDifraction::Integr(double x_beg, double x_end, double y_beg, double y_end, double koll_x, double koll_y, double K) {
    int N_int = 4;
    double h_x = (x_end - x_beg) / double(N_int);
    double h_y = (y_end - y_beg) / double(N_int);

    complex <double> Sum = 0;
    for (size_t i = 0; i < N_int; i++)
    {
        //printf("I = %d \n", i);
        for (size_t j = 0; j < N_int; j++)
        {
            double s_x = x_beg + i * h_x + h_x / 2.0;
            double s_y = y_beg + j * h_y + h_y / 2.0;
            complex <double> tmp = (this->K0 * this->K0 - pow(this->K_f(s_x, s_y), 2)) * this->Kernel(s_x, s_y, koll_x, koll_y, K);
            Sum += tmp;
        }
    }
    return Sum * h_x * h_y;
}

complex <double> DirectTaskDifraction::Integr_for_reverse(double x_beg, double x_end, double y_beg, double y_end, double koll_x, double koll_y, double K) {
    int N_int = 4;
    double h_x = (x_end - x_beg) / double(N_int);
    double h_y = (y_end - y_beg) / double(N_int);

    complex <double> Sum = 0;
    for (size_t i = 0; i < N_int; i++)
    {
        //printf("I = %d \n", i);
        for (size_t j = 0; j < N_int; j++)
        {
            double s_x = x_beg + i * h_x + h_x / 2.0;
            double s_y = y_beg + j * h_y + h_y / 2.0;
            complex <double> tmp = this->Kernel(s_x, s_y, koll_x, koll_y, K);
            Sum += tmp;
        }
    }
    return Sum * h_x * h_y;
}

complex <double> DirectTaskDifraction::getModByX_Y(double _x, double _y) {
    complex< double> Int = 0.;
    for (size_t k = 0; k < this->n * this->n; k++)
    {
        int koord_i = k / this->n;
        int koord_j = k % this->n;
        double x_beg = this->A + koord_i * this->h_x;
        double x_end = x_beg + this->h_x;
        double y_beg = this->C + koord_j * this->h_y;
        double y_end = y_beg + this->h_y;
        Int += this->Integr(x_beg, x_end, y_beg, y_end, _x, _y, this->K0) * this->alpha_beta_vec[k];
    }
    Int += this->fallWave(this->K0, _x, _y);
    return Int;
}

inline void DirectTaskDifraction::set_border(double A, double B, double C, double D)
{
    this->A = A;
    this->B = B;
    this->C = C;
    this->D = D;
}

void DirectTaskDifraction::fill_massive()
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = this->n * this->n;
    this->h_x = (this->B - this->A) / double(this->n);
    this->h_y = (this->D - this->C) / double(this->n);

    int height, stride = 0;

    height = this->get_height_by_rank(rank, size, N);

    for (int i = 0; i < rank; i++) {
        stride += this->get_height_by_rank(i, size, N);
    }

    cout << "Proc " << rank << " create " << height << " X " << N << " Stride " << stride << endl;


    complex <double>** Am, * Vec = new complex <double>[height];


    this->CreateMatrixMemory(height, N, Am);

    for (int I = 0; I < height; I++)  //точки коллокации
    {
        int global_I = I + stride;

        int koll_i = global_I / n;
        int koll_j = global_I % n;

        double x_koll = this->A + koll_i * this->h_x + this->h_x / 2.0;
        double y_koll = this->C + koll_j * this->h_y + this->h_y / 2.0;
        for (int J = 0; J < N; J++)  //координаты
        {
            int koord_i = J / this->n;
            int koord_j = J % this->n;
            double x_beg = this->A + koord_i * this->h_x;
            double x_end = x_beg + this->h_x;
            double y_beg = this->C + koord_j * this->h_y;
            double y_end = y_beg + this->h_y;

            if (global_I == J)
                Am[I][J] = 1.0;
            else
                Am[I][J] = 0.0;
            Am[I][J] -= this->Integr(x_beg, x_end, y_beg, y_end, x_koll, y_koll, this->K0);

            //cout << "     J = " << J << endl;
        }
        Vec[I] = this->fallWave(this->K0, x_koll, y_koll);

        if (rank == 0)
            cout << "I = " << I << " " << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    this->Gauss(Am, Vec, N, rank, size, height, stride);
    MPI_Barrier(MPI_COMM_WORLD);
    this->DeleteMatrixMemory(height, Am);

    this->alpha_beta_vec = new complex <double>[N];

    int* array_stride = nullptr;
    int* array_height = nullptr;


    //объединение полученных данных
    if (rank == 0) {
        array_stride = new int[size];
        array_height = new int[size];
        for (int r = 0; r < size; r++) {
            array_height[r] = this->get_height_by_rank(r, size, N);
            array_stride[r] = 0;
            for (int i = 0; i < r; i++) {
                array_stride[r] += this->get_height_by_rank(i, size, N);
            }
            //cout << "rank = " << r << "   stride = " << array_stride[r] << "   height = " << array_height[r] << endl;
        }
    }
    MPI_Gatherv(Vec, height, MPI_DOUBLE_COMPLEX, this->alpha_beta_vec, array_height, array_stride, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    MPI_Bcast(this->alpha_beta_vec, N, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    delete[] Vec;
}
