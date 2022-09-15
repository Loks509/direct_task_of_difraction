#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <mpi.h>

using namespace std;

double Pi = acos(-1);
double R = 1.0;
double K0 = 2.;
//double K = K0 * 1.2;
//double K0 = 1;
double K = K0 * 1.5;


complex <double> Kernel(double x1, double y1, double x2, double y2, double k) { //если rho_1 = rho_2 то true, иначе false
    complex <double> ed(0, 1.0);
    double l = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
    return exp(ed * k * l) / l;
}

complex <double> diffKernel(double k, double rho, double phi_1, double phi_2, double theta_1, double theta_2) {
    complex <double> ed(0, 1.0);
    double l = rho * sqrt(2 - 2 * sin(theta_1) * sin(theta_2) * cos(phi_1 - phi_2) - 2 * cos(theta_1) * cos(theta_2));
    complex <double> tmp = ed * k * rho * l;
    return exp(tmp) * (tmp - 1.0) / (4.0 * Pi * pow(rho, 2) * l);
}

complex <double> fallWave(double k, double x) {
    complex <double> ed(0, 1.0);
    return exp(ed * k * x);
}

complex <double> difffallWave(double k, double rho, double theta) {
    complex <double> ed(0, 1.0);
    complex <double> tmp = ed * k * cos(theta);
    return tmp * exp(tmp * rho);
}

complex <double> timeFunc(double t) {
    complex <double> ed(0, 1.0);
    double omega = K0 * 3 * pow(10, 8);
    return exp(ed * omega * t);
}

template<typename Type>
void CreateMatrixMemory(int I, int J, Type**& A)
{
    int i1, i2;
    A = new Type *[I];
    for (i1 = 0; i1 < I; i1++) {
        A[i1] = new Type[J];
        for (i2 = 0; i2 < J; i2++) {
            A[i1][i2] = 0.0;
        }
    }
}

//Освобождает память (количество строк)
template<typename Type>
void DeleteMatrixMemory(int I, Type** A)
{
    int i1;
    for (i1 = 0; i1 < I; i1++) {
        delete A[i1];
    }
    delete[]A;
}

void printMatrix(complex <double>** A, int N, int M) {
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < M; j++) {
            cout << A[i][j] << "  ";
        }
        cout << endl;
    }
}

int get_height_by_rank(int rank, int size, int N) {
    int height = N / size;

    if (rank >= size - N % size) height++;
    return height;
}

int get_rank_by_index(int _index, int _size, int _N) {
    int check_index_1 = 0;
    int check_index_2 = 0;
    for (int i = 0; i < _size; i++)
    {
        check_index_2 += get_height_by_rank(i, _size, _N);
        if (check_index_1 <= _index && _index < check_index_2) return i;
        check_index_1 = check_index_2;
    }
    return -1;
}

void print_matrix(complex<double>** _matr, int _rank, int _N, int _height, string dop_str = "") {
    string s;
    s += dop_str + "\n";
    for (int i = 0; i < _height; i++) {
        for (int j = 0; j < _N; j++)
        {
            s += "(" + to_string(_matr[i][j].real()) + "," + to_string(_matr[i][j].imag()) + ")" + "  ";
        }
        s += "\n";
    }

    cout << "rank = " << _rank << "\n" << s << endl;
    fflush(stdout);
}

void print_vec(complex<double>* _vec, int _rank, int _height, string dop_str = "") {
    string s;
    s += dop_str + "\n";
    for (int i = 0; i < _height; i++) {

        s += "(" + to_string(_vec[i].real()) + "," + to_string(_vec[i].imag()) + ")" + "  ";

        s += "\n";
    }

    cout << "rank = " << _rank << "\n" << s << endl;
    fflush(stdout);
}

void Gauss(complex <double>** _Matrix, complex <double>* _Vec, int _Nm, int _rank, int _size, int _height, int _stride) {
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
                //cout << "rank = " << _rank << " end recv from rank = " << stat.MPI_SOURCE << endl;
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


    //for (int i = _height - 1; i >= 0; i--) {
    //    for (int j = _Nm - 1; j >= _stride + i; j--) {  //неверно. последний ранг не должен вообще ничего принимать, наверное надо как-то разделить
    //        if (j == _stride + i) {     //дошли до диагонального элемента

    //        }
    //        else {                      //ждем данных

    //        }
    //    }
    //}
    int variables = _Nm - _stride - _height;    //столько переменных недоступно в каждом матрице (в последней = 0)

    //cout << "rank = " << _rank << " var = " << variables << endl;

    for (int ind_x = 0; ind_x < variables; ind_x++) {   //тут принимаем все недостающие переменные для того, чтобы выполнить обратный ход в этой части матрицы
        complex <double> recv_x;
        MPI_Recv(&recv_x, 1, MPI_DOUBLE_COMPLEX, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &stat);
        //cout << "rank = "<<_rank<<"  "<<ind_x<<"/"<<variables << " recv data = " << recv_x << "  from rank = " << stat.MPI_SOURCE << endl;
        //fflush(stdout);
        for (int i = 0; i < _height; i++) {     //отнимаем принятое значение, умноженное на данный столбец, от вектора правой части
            int j = _Nm - ind_x - 1;
            _Vec[i] -= _Matrix[i][j] * recv_x;
        }
    }

    for (int i = _height - 1; i >= 0; i--) {
        int ind_d = _stride + i;                    //столбец диагонального элемента
        for (int j = _Nm - variables - 1; j > ind_d; j--) {     //если убрать variables то будет непостоянная ошибка (любопытно)
            //работаем со столбцами, от диагонального до блока с неизвестными коэффициентами (потому что при принятии уже сделали все необходимые операции)
            //cout << "rank = " << _rank << "  j = " << j << "  " << _stride << "  razn = " << j - _stride << endl;
            //fflush(stdout);
            _Vec[i] -= _Vec[j - _stride] * _Matrix[i][j];
        }
        for (int r = 0; r < _rank; r++) {           //рассылаем всем кто выше по матрице
            MPI_Ssend(&_Vec[i], 1, MPI_DOUBLE_COMPLEX, r, 3, MPI_COMM_WORLD);
            //cout << "send data = " << _Vec[i] << "  to rank = " << r << endl;
            //fflush(stdout);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    delete[]tmp_row;
}

void GenerateVTK(vector<vector<double>> data, string name_file = "out.vtk", double time = -1.0) {

    int points = data.size();
    cout << "Points " << points << endl;

    ofstream file2(name_file);
    file2 << "# vtk DataFile Version 2.0\n" <<
        "Cube example\n" <<
        "ASCII\n" <<
        "DATASET POLYDATA\n" <<
        "POINTS " << points << " float" << endl;

    for (size_t i = 0; i < points; i++)
    {
        file2 << data[i][0] << " " << data[i][1] << " " << data[i][2] << endl;
    }
    int side = sqrt(points);
    int CountOfPolygons = pow(side - 1, 2);
    file2 << "POLYGONS " << CountOfPolygons << " " << CountOfPolygons * 5 << endl;//5 - количество координат на полигон 

    for (size_t i = 0; i < points - side - 1; i++)
    {
        int wP = i;
        if ((wP + 1) % side != 0 || wP == 0)
            file2 << 4 << " " << wP << " " << wP + 1 << " " << wP + side + 1 << " " << wP + side << endl;
    }
    file2 << "POINT_DATA " << points << endl <<
        "SCALARS Magnitude float 1\n" <<
        "LOOKUP_TABLE default" << endl;

    for (size_t i = 0; i < points; i++)
    {
        if (time == -1)
            file2 << data[i][3] << endl;
        else
            file2 << abs(complex<double>(data[i][4], data[i][5]) * timeFunc(time)) << endl;
    }

    file2 <<
        "SCALARS real float 1\n" <<
        "LOOKUP_TABLE default" << endl;

    for (size_t i = 0; i < points; i++)
    {
        if (time == -1)
            file2 << data[i][4] << endl;
        else
            file2 << (complex<double>(data[i][4], data[i][5]) * timeFunc(time)).real() << endl;
    }

    file2 <<
        "SCALARS imag float 1\n" <<
        "LOOKUP_TABLE default" << endl;

    for (size_t i = 0; i < points; i++)
    {
        if (time == -1)
            file2 << data[i][5] << endl;
        else
            file2 << (complex<double>(data[i][4], data[i][5]) * timeFunc(time)).imag() << endl;
    }
    file2.close();
}

void GenerateVTK_grid(double** data, int count_p, string name_file = "out.vtk", double time = -1.0) {

    int countOfPoints = count_p;
    int size = pow(countOfPoints, 1.0 / 3.0);

    cout << "Points " << countOfPoints << endl;

    ofstream out(name_file);

    out << "# vtk DataFile Version 3.0" << endl <<
        "Example 3D regular grid VTK file." << endl <<
        "ASCII" << endl <<
        "DATASET STRUCTURED_GRID" << endl <<
        "DIMENSIONS " << size + 1 << " " << size + 1 << " " << size + 1 << endl <<
        "POINTS " << countOfPoints << " double" << endl;

    for (int ind = 0; ind < countOfPoints; ind++) {
        out << data[0][ind] << " " << data[1][ind] << " " << data[2][ind] << endl;
    }
    out << "POINT_DATA " << countOfPoints << endl <<
        "SCALARS J double 1" << endl <<
        "LOOKUP_TABLE default" << endl;
    for (int ind = 0; ind < countOfPoints; ind++) {
        if (time == -1) out << data[3][ind] << endl;
        else            out << abs(complex<double>(data[4][ind], data[5][ind]) * timeFunc(time)) << endl;
    }

    out <<
        "SCALARS real double 1" << endl <<
        "LOOKUP_TABLE default" << endl;
    for (int ind = 0; ind < countOfPoints; ind++) {
        if (time == -1) out << data[4][ind] << endl;
        else            out << (complex<double>(data[4][ind], data[5][ind]) * timeFunc(time)).real() << endl;
    }

    out << 
        "SCALARS imag double 1" << endl <<
        "LOOKUP_TABLE default" << endl;
    for (int ind = 0; ind < countOfPoints; ind++) {
        if (time == -1) out << data[5][ind] << endl;
        else            out << (complex<double>(data[4][ind], data[5][ind]) * timeFunc(time)).imag() << endl;
    }
    out.close();
}

complex <double> Integr(double x_beg, double x_end, double y_beg, double y_end, double koll_x, double koll_y, double K) {
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
            double s_y = x_beg + j * h_y + h_y / 2.0;
            complex <double> tmp = Kernel(s_x, s_y, koll_x, koll_y, K);
            Sum += tmp;
            /*if(K==K0)
                printf("Koll_phi = %f, Koll_theta = %f, s_phi = %f, s_theta = %f, tmp = %f i %f\n", koll_phi, koll_theta, s_phi, s_theta, tmp.real(), tmp.imag());*/
        }
    }
    //exit(-1);
    //cout << "Sum = " << Sum * h_x * h_y << endl;
    return Sum * h_x * h_y;
}

complex <double> diffIntegr(double phi_beg, double phi_end, double theta_beg, double theta_end, double koll_phi, double koll_theta, double K, double rho) {
    int N_int = 4;
    double h_x = (phi_end - phi_beg) / double(N_int);
    double h_y = (theta_end - theta_beg) / double(N_int);


    complex <double> Sum = 0;

    for (size_t i = 0; i < N_int; i++)
    {
        for (size_t j = 0; j < N_int; j++)
        {
            double s_phi = phi_beg + i * h_x + h_x / 2.0;
            double s_theta = theta_beg + j * h_y + h_y / 2.0;
            Sum += diffKernel(K, rho, koll_phi, s_phi, koll_theta, s_theta) * sin(s_theta);
        }
    }
    return Sum * h_x * h_y;
}

void ReadFile(const char Patch[], double** Matr, int N, int M) {
    ifstream file(Patch);
    if (!file.is_open()) {
        cout << "Read File " << Patch << " fail\n";
        exit(-3);
    }
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            file >> Matr[i][j];
        }
    }
}

void ReadFile(const char Patch[], double* Vec, int N) {
    ifstream file(Patch);
    if (!file.is_open()) {
        cout << "Read File " << Patch << " fail\n";
        exit(-3);
    }
    for (size_t i = 0; i < N; i++) {
        file >> Vec[i];
    }
}



double xyToPhi(double x, double y) {
    if (x == 0) {
        if (y > 0) return Pi / 2.0;
        else if (y < 0) return 3.0 * Pi / 2.0;
    }
    if (y == 0) {
        if (x > 0) return 0;
        else return Pi;
    }
    if (x > 0 && y > 0)         return atan(y / x);
    else if (x < 0 && y > 0)    return atan(y / x) + Pi;
    else if (x < 0 && y < 0)    return atan(y / x) + Pi;
    else if (x > 0 && y < 0)    return atan(y / x) + 2.0 * Pi;

    printf("X = %f, Y = %f", x, y);
    exit(-2);
}

bool in_range(double begin, double var, double end) {
    return (begin < var) && (var < end);
}

void toSpherical(double* decart, double* spherical) {
    double Eps = 0.000001;
    double x = decart[0];
    double y = decart[1];
    double z = decart[2];

    if (in_range(-Eps, x, Eps)) {
        if (y > 0) spherical[0] = Pi / 2.0;
        else if (y < 0) spherical[0] = 3.0 * Pi / 2.0;
    }
    else if (in_range(-Eps, y, Eps)) {
        if (x > 0) spherical[0] = 0;
        else spherical[0] = Pi;
    }
    else if (x > 0 && y > 0)         spherical[0] = atan(y / x);
    else if (x < 0)    spherical[0] = atan(y / x) + Pi;
    else if (x > 0 && y < 0)    spherical[0] = atan(y / x) + 2.0 * Pi;
    else {
        spherical[0] = atan(y / x);
    }

    if (in_range(-Eps, z, Eps)) {
       spherical[1] = Pi / 2.0;
    }
    else if (z < 0) {
         spherical[1] = Pi + atan(sqrt(x * x + y * y) / z);
    }
    else {
        spherical[1] = atan(sqrt(x * x + y * y) / z);
    }



    spherical[2] = sqrt(x * x + y * y + z * z);
}

void toDecart(double* decart, double* spherical) {
    double phi = spherical[0];
    double theta = spherical[1];
    double rho = spherical[2];
    decart[0] = rho * sin(theta) * cos(phi);
    decart[1] = rho * sin(theta) * sin(phi);
    decart[2] = rho * cos(theta);
}



int main() {
    int rank, size;
    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double A = 0, B = 2;           //изменение x
    double C = 0, D = 2;               //изменение y
    int n = 30;
    int N = n * n;
    double h_x = (B - A) / double(n);
    double h_y = (D - C) / double(n);

    int height, stride = 0;

    height = get_height_by_rank(rank, size, N);

    for (int i = 0; i < rank; i++) {
        stride += get_height_by_rank(i, size, N);
    }

    cout << "Proc " << rank << " create " << height << " X " << N <<" Stride "<<stride<< endl;


    complex <double>** Am, * Vec = new complex <double>[height];
    

    CreateMatrixMemory(height, N, Am);
    double begin = MPI_Wtime();

    double RR = R * R;
    for (int I = 0; I < height; I++)  //точки коллокации
    {
        int global_I = I + stride;

        int koll_i = global_I / n;
        int koll_j = global_I % n;

        double x_koll = A + koll_i * h_x + h_x / 2.0;
        double y_koll = C + koll_j * h_y + h_y / 2.0;
        for (int J = 0; J < N; J++)  //координаты
        {
            int koord_i = J / n;
            int koord_j = J % n;
            double x_beg = A + koord_i * h_x;
            double x_end = x_beg + h_x;
            double y_beg = C + koord_j * h_y;
            double y_end = y_beg + h_y;

            if (global_I == J)
                Am[I][J] = 1.;
            else
                Am[I][J] = 0.;
            Am[I][J] -= K * K * Integr(x_beg, x_end, y_beg, y_end, x_koll, y_koll, K);
            
            //cout << "     J = " << J << endl;
        }
        Vec[I] = fallWave(K0, x_koll);

        if (rank == 0)
            cout << "I = " << I << " " << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_fill = MPI_Wtime();
    //print_matrix(Am, rank, N, height, "do");
    //MPI_Barrier(MPI_COMM_WORLD);
    //print_vec(Vec, rank, height, "do");
    //fflush(stdout);
    Gauss(Am, Vec, N, rank, size, height, stride);

    double end_gauss = MPI_Wtime();
    //print_matrix(Am, rank, N, height, "posle");
    //MPI_Barrier(MPI_COMM_WORLD);
    //print_vec(Vec, rank, height, "posle");
    MPI_Barrier(MPI_COMM_WORLD);
    DeleteMatrixMemory(height, Am);

    complex<double>* alpha_beta_vec = new complex <double>[N];
    //complex<double>* beta_vec = new complex <double>[nn];

    int* array_stride = nullptr;
    int* array_height = nullptr;

    
    //объединение полученных данных
    if (rank == 0) {
        array_stride = new int[size];
        array_height = new int[size];
        for (int r = 0; r < size; r++) {
            array_height[r] = get_height_by_rank(r, size, N);
            array_stride[r] = 0;
            for (int i = 0; i < r; i++) {
                array_stride[r] += get_height_by_rank(i, size, N);
            }
            //cout << "rank = " << r << "   stride = " << array_stride[r] << "   height = " << array_height[r] << endl;
        }
    }
    MPI_Gatherv(Vec,height,MPI_DOUBLE_COMPLEX,alpha_beta_vec,array_height,array_stride, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    MPI_Bcast(alpha_beta_vec, N, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    double sync = MPI_Wtime();


    
    if (rank == 0) {
        ofstream alpha("alpha.txt");

        alpha << "X Y Z F real imag" << endl;
        for (size_t I = 0; I < N; I++)
        {
            int _i = I / n;
            int _j = I % n;
            double x = A + _i * h_x + h_x / 2.0;
            double y = C + _j * h_y + h_y / 2.0;
            //alpha_beta_vec[I] = fallWave(K0, x);
            alpha << x << " " << y << " " << 0 << " " << abs(alpha_beta_vec[I]) << " " << alpha_beta_vec[I].real() << " " << alpha_beta_vec[I].imag() << endl;
            //file << x << " " << Vec[i] << " " << u(x) << endl;
            //cout << phi << " " << Vec[i] << endl;
        }
        alpha.close();
    }
    if (rank == 0) {
        ofstream mod("mod.txt");
        mod << "X Y Z F real imag" << endl;
        for (int i = -2*n; i < 2*n; i++)
        {
            double x = A + h_x * i;
            for (int j = -2*n; j < 2*n; j++)
            {
                double y = C + h_y * j;

                complex< double> Int = 0.;
                for (size_t k = 0; k < N; k++)
                {
                    int koord_i = k / n;
                    int koord_j = k % n;
                    double x_beg = A + koord_i * h_x;
                    double x_end = x_beg + h_x;
                    double y_beg = C + koord_j * h_y;
                    double y_end = y_beg + h_y;
                    Int += Integr(x_beg, x_end, y_beg, y_end, x, y, K);
                }
                Int *= K * K;
                Int += fallWave(K0, x);
                mod << x << " " << y << " " << 0 << " " << abs(Int) << " " << Int.real() << " " << Int.imag() << endl;
            }
            cout << "i_x = " << i << endl;
        }
    }


    //ofstream map_int("map.txt");
    //map_int << "X Y Z abs real imag" << endl;

    //double edge = 6.0;
    //
    //double x_1 = -edge, x_2 = edge;
    //int N_x = 40;
    //double h_x = (x_2 - x_1) / (double)(N_x - 1.0);

    //double y_1 = -edge, y_2 = edge;
    //int N_y = 40;
    //double h_y = (y_2 - y_1) / (double)(N_y - 1.0);

    //double z_1 = -edge, z_2 = edge;
    //int N_z = 40;
    //double h_z = (z_2 - z_1) / (double)(N_z - 1.0);

    //int stride_x = 0;
    //int count_x = get_height_by_rank(rank, size, N_x);

    //for (int i = 0; i < rank; i++) {
    //    stride_x += get_height_by_rank(i, size, N_x);
    //}
    ////cout << "rank = " << rank << "   stride = " << stride_x << "   height = " << count_x << endl;
    //int* array_stride_x = nullptr;
    //int* array_count_x = nullptr;

    //if (rank == 0) {
    //    array_stride_x = new int[size];
    //    array_count_x = new int[size];
    //    for (int r = 0; r < size; r++) {
    //        array_count_x[r] = get_height_by_rank(r, size, N_x) * N_y * N_z;
    //        array_stride_x[r] = 0;
    //        for (int i = 0; i < r; i++) {
    //            array_stride_x[r] += get_height_by_rank(i, size, N_x) * N_y * N_z;
    //        }
    //        cout << "rank = " << r << "   stride = " << array_stride_x[r] << "   height = " << array_count_x[r] << endl;
    //    }
    //}

    //double** out_data;
    //CreateMatrixMemory(6, count_x * N_y * N_z, out_data);

    //double* decart = new double[3], * sph = new double[3];
    //
    //int counter = 0;

    //for (int i = stride_x; i < stride_x + count_x; i++) {
    //    double x = x_1 + i * h_x;
    //    for (int j = 0; j < N_y;j++) 
    //    {
    //        double y = y_1 + j * h_y;
    //        for (int k = 0; k < N_z; k++)
    //        {
    //            double z = z_1 + k * h_z;


    //            decart[0] = x;
    //            decart[1] = y;
    //            decart[2] = z;
    //            toSpherical(decart, sph);
    //            double current_K = 0;
    //            complex<double> Intens = 0.0;
    //            int stride_vec = 0;

    //            //if (!(in_range(0.0001, x, 0.0001) && in_range(0.0001, y, 0.0001) && in_range(0.0001, z, 0.0001))) {
    //            if (sph[2] > R) {
    //                current_K = K0;
    //                Intens += fallWave(K0, sph[2], sph[1]);
    //            }
    //            else {
    //                current_K = K;
    //                stride_vec = nn;
    //            }


    //            for (int ind = 0; ind < nn; ind++) {
    //                int koord_i = ind / n;
    //                int koord_j = ind % n;
    //                double phi_beg = A + koord_i * h_x;
    //                double phi_end = phi_beg + h_x;
    //                double theta_beg = C + koord_j * h_y;
    //                double theta_end = theta_beg + h_y;
    //                Intens += Integr(phi_beg, phi_end, theta_beg, theta_end, sph[0], sph[1], current_K, sph[2], R) * RR * alpha_beta_vec[ind + stride_vec];
    //            }
    //            //}
    //            //map_int << x << " " << y << " " << z << " " << abs(Intens) << " " << Intens.real() << " " << Intens.imag() << endl;
    //            //data.push_back({ x,y,z, abs(Intens), Intens.real(), Intens.imag() });

    //            if (_Is_nan(abs(Intens))) Intens = 0.0;

    //            out_data[0][counter] = x;
    //            out_data[1][counter] = y;
    //            out_data[2][counter] = z;
    //            out_data[3][counter] = abs(Intens);
    //            out_data[4][counter] = Intens.real();
    //            out_data[5][counter] = Intens.imag();
    //            counter++;
    //        }
    //    }
    //    if (rank == 0)
    //        cout << "X = " << x << endl;
    //}

    //double** all_data;
    //CreateMatrixMemory(6, N_x * N_y * N_z, all_data);

    //MPI_Gatherv(out_data[0], count_x * N_y * N_z, MPI_DOUBLE, all_data[0], array_count_x, array_stride_x, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Gatherv(out_data[1], count_x * N_y * N_z, MPI_DOUBLE, all_data[1], array_count_x, array_stride_x, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Gatherv(out_data[2], count_x * N_y * N_z, MPI_DOUBLE, all_data[2], array_count_x, array_stride_x, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Gatherv(out_data[3], count_x * N_y * N_z, MPI_DOUBLE, all_data[3], array_count_x, array_stride_x, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Gatherv(out_data[4], count_x * N_y * N_z, MPI_DOUBLE, all_data[4], array_count_x, array_stride_x, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Gatherv(out_data[5], count_x * N_y * N_z, MPI_DOUBLE, all_data[5], array_count_x, array_stride_x, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //double create_grid = MPI_Wtime();
    //if (rank == 0) {
    //    GenerateVTK_grid(all_data, N_x * N_y * N_z, "test_file.vtk");

    //    double time_one_oscl = 2 * Pi / 3 / pow(10, 8);
    //    int i = 1;
    //    for (double time = 0; time < 1 * time_one_oscl; time += time_one_oscl / 10)
    //    {
    //        string name = "time" + to_string(i) + ".vtk";
    //        GenerateVTK_grid(all_data, N_x * N_y * N_z, name, time);
    //        i++;
    //    }

    //}

    //double end = MPI_Wtime();

    //

    //if (rank == 0) {
    //    cout << "Time fill = " << end_fill - begin << "  /  " << end_fill - begin << endl;
    //    cout << "Time gauss = " << end_gauss - end_fill << "  /  " << end_gauss - begin << endl;
    //    cout << "Time sync = " << sync - end_gauss << "  /  " << sync - begin << endl;
    //    cout << "Time create_grid = " << create_grid - sync << "  /  " << create_grid - begin << endl;
    //    cout << "Time create_file = " << end - create_grid << "  /  " << end - begin << endl;
    //}



    MPI_Finalize();

    return 0;
}
