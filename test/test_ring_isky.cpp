/*
Authors: Deevashwer Rathee, Mayank Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions
:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "BuildingBlocks/aux-protocols.h"
#include "NonLinear/argmax.h"
#include "LinearOT/linear-ot.h"
#include "utils/emp-tool.h"
#include "OT/np.h"
#include "OT/ot.h"
#include "utils/prp.h"
#include <iostream>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <smmintrin.h>
#include <ctime>
#include <string>
#include <cstring>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <pthread.h>
using namespace sci;
using namespace std;

int party, port = 8000;
string address = "127.0.0.1";
int Pd = 1, Pn = 3, Pst = 0, Pr = 1, Pth = 1, Pta = 1, Pm = 1;
// int THs = 2;
// NetIO *Iot[2];
// OTPack<NetIO> *Otpackt[2];
// AuxProtocols *Auxt[2];
// LinearOT *Prodt[2];
int THs = 32;
NetIO *Iot[32];
OTPack<NetIO> *Otpackt[32];
AuxProtocols *Auxt[32];
LinearOT *Prodt[32];
PRG128 prg;
CRH crh;
int n = 10;
int m = 2;
int domain = 10000;
//"/small-correlated.txt";"/small-uniformly-distributed.txt";"/small-anti-correlated.txt";
string filename = "/small-correlated.txt";
string dataname = "corr-";
string data_path = "./data/input=10000/size=" + to_string(n) + filename;
uint64_t **SS_p, **SS_zeroM, **Skyline;
uint64_t *SS_q;
unordered_map<uint32_t, vector<vector<uint64_t>>> SS_pB;
uint64_t SS_one, SS_zero;
double pre_time = 0, sort_time = 0, que_time = 0;
int countDom = 0;
int countCmp = 0;

int MAX = 14;
int lambda = 64;
// uint32_t mask = 4294967291;
uint64_t mask = -1;

void Ret(uint64_t *v, int size)
{
  uint64_t *v_t = new uint64_t[size];
// if (party == ALICE)
// {
//   Iot[0]->send_data(v, size * sizeof(uint64_t));
//   Iot[1]->recv_data(v_t, size * sizeof(uint64_t));
// }
// else
// {
//   Iot[1]->send_data(v, size * sizeof(uint64_t));
//   Iot[0]->recv_data(v_t, size * sizeof(uint64_t));
// }
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    {
#pragma omp task firstprivate(size)
      {
        if (party == ALICE)
        {
          Iot[0]->send_data(v, size * sizeof(uint64_t));
        }
        else
        {
          Iot[1]->send_data(v, size * sizeof(uint64_t));
        }
      }
#pragma omp task firstprivate(size)
      {
        if (party == ALICE)
        {
          Iot[1]->recv_data(v_t, size * sizeof(uint64_t));
        }
        else
        {
          Iot[0]->recv_data(v_t, size * sizeof(uint64_t));
        }
      }
#pragma omp taskwait
    }
  }
  cout << "[";
  for (int i = 0; i < size - 1; i++)
  {
    cout << ((v[i] + v_t[i]) & mask) << ",";
  }
  cout << +((v[size - 1] + v_t[size - 1]) & mask) << "]";
  // cout <<endl;
  delete[] v_t;
}

void Ret(vector<uint64_t> vt, int size)
{
  uint64_t *v = new uint64_t[size];
  uint64_t *v_t = new uint64_t[size];
  copy(vt.begin(), vt.end(), v);
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    {
#pragma omp task firstprivate(size)
      {
        if (party == ALICE)
        {
          Iot[0]->send_data(v, size * sizeof(uint64_t));
        }
        else
        {
          Iot[1]->send_data(v, size * sizeof(uint64_t));
        }
      }
#pragma omp task firstprivate(size)
      {
        if (party == ALICE)
        {
          Iot[1]->recv_data(v_t, size * sizeof(uint64_t));
        }
        else
        {
          Iot[0]->recv_data(v_t, size * sizeof(uint64_t));
        }
      }
#pragma omp taskwait
    }
  }
  cout << "[";
  for (int i = 0; i < size - 1; i++)
  {
    cout << ((v[i] + v_t[i]) & mask) << ",";
  }
  cout << +((v[size - 1] + v_t[size - 1]) & mask) << "]";
  // cout <<endl;
  delete[] v;
  delete[] v_t;
}

void Ret(uint64_t *v, uint64_t *&v_t, int size)
{
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    {
#pragma omp task firstprivate(size)
      {
        if (party == ALICE)
        {
          Iot[0]->send_data(v, size * sizeof(uint64_t));
        }
        else
        {
          Iot[1]->send_data(v, size * sizeof(uint64_t));
        }
      }
#pragma omp task firstprivate(size)
      {
        if (party == ALICE)
        {
          Iot[1]->recv_data(v_t, size * sizeof(uint64_t));
        }
        else
        {
          Iot[0]->recv_data(v_t, size * sizeof(uint64_t));
        }
      }
#pragma omp taskwait
    }
  }
  for (int i = 0; i < size; i++)
  {
    v_t[i] = (v[i] + v_t[i]) & mask;
  }
}

void Ret(vector<uint64_t> vt, vector<uint64_t> &vt_t, int size)
{
  uint64_t *v = new uint64_t[size];
  uint64_t *v_t = new uint64_t[size];
  copy(vt.begin(), vt.end(), v);
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    {
#pragma omp task firstprivate(size)
      {
        if (party == ALICE)
        {
          Iot[0]->send_data(v, size * sizeof(uint64_t));
        }
        else
        {
          Iot[1]->send_data(v, size * sizeof(uint64_t));
        }
      }
#pragma omp task firstprivate(size)
      {
        if (party == ALICE)
        {
          Iot[1]->recv_data(v_t, size * sizeof(uint64_t));
        }
        else
        {
          Iot[0]->recv_data(v_t, size * sizeof(uint64_t));
        }
      }
#pragma omp taskwait
    }
  }
  for (int i = 0; i < size; i++)
  {
    v_t[i] = (v[i] + v_t[i]) & mask;
  }
  copy(v_t, v_t + size, vt_t.begin());
  delete[] v;
  delete[] v_t;
}

void Ret(uint64_t *v, vector<uint64_t> &vt_t, int size)
{
  uint64_t *v_t = new uint64_t[size];
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    {
#pragma omp task firstprivate(size)
      {
        if (party == ALICE)
        {
          Iot[0]->send_data(v, size * sizeof(uint64_t));
        }
        else
        {
          Iot[1]->send_data(v, size * sizeof(uint64_t));
        }
      }
#pragma omp task firstprivate(size)
      {
        if (party == ALICE)
        {
          Iot[1]->recv_data(v_t, size * sizeof(uint64_t));
        }
        else
        {
          Iot[0]->recv_data(v_t, size * sizeof(uint64_t));
        }
      }
#pragma omp taskwait
    }
  }
  for (int i = 0; i < size; i++)
  {
    v_t[i] = (v[i] + v_t[i]) & mask;
  }
  copy(v_t, v_t + size, vt_t.begin());
  delete[] v_t;
}

void Ret_T(uint64_t *v, uint64_t *&v_t, int size, int itr)
{
  if (party == ALICE)
  {
    Iot[itr]->send_data(v, size * sizeof(uint64_t));
    Iot[itr]->recv_data(v_t, size * sizeof(uint64_t));
  }
  else
  {
    Iot[itr]->recv_data(v_t, size * sizeof(uint64_t));
    Iot[itr]->send_data(v, size * sizeof(uint64_t));
  }
  for (int i = 0; i < size; i++)
  {
    v_t[i] = (v[i] + v_t[i]) & mask;
  }
}

void shuffle(int len, uint32_t *&res)
{
  for (int i = 0; i < len; i++)
  {
    if (rand() % 2 == 1)
    {
      uint32_t t = rand() % len;
      uint32_t tmp = res[i];
      res[i] = res[t];
      res[t] = tmp;
    }
  }
}

void comparison(int dim, uint64_t *x, uint64_t *y, uint64_t *&out, AuxProtocols *Aux)
{
  uint64_t mask = -1;
  int lambda = 64;
  uint64_t maskt_2 = (1ULL << (lambda - 1)) - 1;
  uint8_t *cmp = new uint8_t[dim];
  uint8_t *eq = new uint8_t[dim];
  uint64_t *z = new uint64_t[dim];
  uint64_t *w = new uint64_t[dim];
  uint64_t *b = new uint64_t[dim];
  // uint8_t *flp = new uint8_t[dim];
  // cout << maskt_2 << endl;
  for (int j = 0; j < dim; j++)
  {
    z[j] = (y[j] - x[j]) & mask;
    w[j] = z[j] & maskt_2;
  }
  if (party == ALICE)
  {
    for (int j = 0; j < dim; j++)
    {
      w[j] = maskt_2 - w[j];
    }
  }
  Aux->comparison_with_eq(cmp, eq, w, dim, lambda); // Bob less and equal than Alice, result is 1. Greater  is 0.
  for (int j = 0; j < dim; j++)
  {
    if (z[j] < maskt_2)
    {
      b[j] = 1;
    }
    else
    {
      b[j] = 0;
    }
  }
  for (int j = 0; j < dim; j++)
  {
    cmp[j] = (cmp[j] ^ eq[j]) & 1;
  }
  // result = (b_a \xor b_b \xor (cmp_a + cmp_B))
  // flp = eq;
  Aux->equality(b, eq, dim, lambda);
  if (party == ALICE)
  {
    for (int j = 0; j < dim; j++)
    {
      eq[j] = 1 - eq[j];
    }
  }
  // if (party == ALICE)
  //     {
  //       uint8_t *tf =  new uint8_t[dim];
  //       Iot[0]->recv_data(tf, dim * sizeof(uint8_t));
  //       Iot[1]->send_data(cmp, dim * sizeof(uint8_t));
  //       for (int i = 0; i < dim; i++)
  //       {
  //         cout<<i<<"-th:w:"<<"w_A:"<<+cmp[i]<<",w_B:"<<+tf[i]<<endl;
  //       }
  //       delete[] tf;
  //       uint8_t *tb =  new uint8_t[dim];
  //       Iot[0]->recv_data(tb, dim * sizeof(uint8_t));
  //       Iot[1]->send_data(flp, dim * sizeof(uint8_t));
  //       for (int i = 0; i < dim; i++)
  //       {
  //         cout<<i<<"-th:b:"<<"b_A:"<<+flp[i]<<",b_B:"<<+tb[i]<<endl;
  //       }
  //       delete[] tb;
  //     }
  //     else
  //     {
  //       uint8_t *tf =  new uint8_t[dim];
  //       Iot[0]->send_data(cmp, dim * sizeof(uint8_t));
  //       Iot[1]->recv_data(tf, dim * sizeof(uint8_t));
  //       for (int i = 0; i < dim; i++)
  //       {
  //         cout<<i<<"-th:w:"<<"w_A:"<<+tf[i]<<",w_B:"<<+cmp[i]<<endl;
  //       }
  //       delete[] tf;
  //       uint8_t *tb =  new uint8_t[dim];
  //       Iot[0]->send_data(flp, dim * sizeof(uint8_t));
  //       Iot[1]->recv_data(tb, dim * sizeof(uint8_t));
  //       for (int i = 0; i < dim; i++)
  //       {
  //         cout<<i<<"-th:b:"<<"b_A:"<<+tb[i]<<",b_B:"<<+flp[i]<<endl;
  //       }
  //       delete[] tb;
  //     }
  for (int j = 0; j < dim; j++)
  {
    cmp[j] = (cmp[j] ^ eq[j]) & 1; // cmp \xor flp
  }
  // if (party == ALICE)
  //     {
  //       uint8_t *tf =  new uint8_t[dim];
  //       Iot[0]->recv_data(tf, dim * sizeof(uint8_t));
  //       Iot[1]->send_data(cmp, dim * sizeof(uint8_t));
  //       for (int i = 0; i < dim; i++)
  //       {
  //         cout<<i<<"-th:w:"<<"w_A:"<<+cmp[i]<<",w_B:"<<+tf[i]<<endl;
  //       }
  //       delete[] tf;
  //     }
  //     else
  //     {
  //       uint8_t *tf =  new uint8_t[dim];
  //       Iot[0]->send_data(cmp, dim * sizeof(uint8_t));
  //       Iot[1]->recv_data(tf, dim * sizeof(uint8_t));
  //       for (int i = 0; i < dim; i++)
  //       {
  //         cout<<i<<"-th:w:"<<"w_A:"<<+tf[i]<<",w_B:"<<+cmp[i]<<endl;
  //       }
  //       delete[] tf;
  //     }
  Aux->B2A(cmp, out, dim, lambda); // binary share to arithmetic share
  // delete[] flp;
  delete[] b;
  delete[] w;
  delete[] z;
  delete[] eq;
  delete[] cmp;
}

// x <= y
void comparison(int dim, uint64_t *x, uint64_t *y, vector<uint64_t> &out, AuxProtocols *Aux)
{
  uint64_t mask = -1;
  int lambda = 64;
  uint64_t maskt_2 = (1ULL << (lambda - 1)) - 1;
  uint8_t *cmp = new uint8_t[dim];
  uint8_t *eq = new uint8_t[dim];
  uint64_t *z = new uint64_t[dim];
  uint64_t *w = new uint64_t[dim];
  uint64_t *b = new uint64_t[dim];
  // uint8_t *flp = new uint8_t[dim];
  // cout << maskt_2 << endl;
  for (int j = 0; j < dim; j++)
  {
    z[j] = (y[j] - x[j]) & mask;
    w[j] = z[j] & maskt_2;
  }
  if (party == ALICE)
  {
    for (int j = 0; j < dim; j++)
    {
      w[j] = maskt_2 - w[j];
    }
  }
  Aux->comparison_with_eq(cmp, eq, w, dim, lambda); // Bob less and equal than Alice, result is 1. Greater  is 0.
  for (int j = 0; j < dim; j++)
  {
    if (z[j] < maskt_2)
    {
      b[j] = 1;
    }
    else
    {
      b[j] = 0;
    }
  }
  for (int j = 0; j < dim; j++)
  {
    cmp[j] = (cmp[j] ^ eq[j]) & 1;
  }
  // result = (b_a \xor b_b \xor (cmp_a + cmp_B))
  // flp = eq;
  Aux->equality(b, eq, dim, lambda);
  if (party == ALICE)
  {
    for (int j = 0; j < dim; j++)
    {
      eq[j] = 1 - eq[j];
    }
  }
  // if (party == ALICE)
  //     {
  //       uint8_t *tf =  new uint8_t[dim];
  //       Iot[0]->recv_data(tf, dim * sizeof(uint8_t));
  //       Iot[1]->send_data(cmp, dim * sizeof(uint8_t));
  //       for (int i = 0; i < dim; i++)
  //       {
  //         cout<<i<<"-th:w:"<<"w_A:"<<+cmp[i]<<",w_B:"<<+tf[i]<<endl;
  //       }
  //       delete[] tf;
  //       uint8_t *tb =  new uint8_t[dim];
  //       Iot[0]->recv_data(tb, dim * sizeof(uint8_t));
  //       Iot[1]->send_data(flp, dim * sizeof(uint8_t));
  //       for (int i = 0; i < dim; i++)
  //       {
  //         cout<<i<<"-th:b:"<<"b_A:"<<+flp[i]<<",b_B:"<<+tb[i]<<endl;
  //       }
  //       delete[] tb;
  //     }
  //     else
  //     {
  //       uint8_t *tf =  new uint8_t[dim];
  //       Iot[0]->send_data(cmp, dim * sizeof(uint8_t));
  //       Iot[1]->recv_data(tf, dim * sizeof(uint8_t));
  //       for (int i = 0; i < dim; i++)
  //       {
  //         cout<<i<<"-th:w:"<<"w_A:"<<+tf[i]<<",w_B:"<<+cmp[i]<<endl;
  //       }
  //       delete[] tf;
  //       uint8_t *tb =  new uint8_t[dim];
  //       Iot[0]->send_data(flp, dim * sizeof(uint8_t));
  //       Iot[1]->recv_data(tb, dim * sizeof(uint8_t));
  //       for (int i = 0; i < dim; i++)
  //       {
  //         cout<<i<<"-th:b:"<<"b_A:"<<+tb[i]<<",b_B:"<<+flp[i]<<endl;
  //       }
  //       delete[] tb;
  //     }
  for (int j = 0; j < dim; j++)
  {
    cmp[j] = (cmp[j] ^ eq[j]) & 1; // cmp \xor flp
  }
  // if (party == ALICE)
  //     {
  //       uint8_t *tf =  new uint8_t[dim];
  //       Iot[0]->recv_data(tf, dim * sizeof(uint8_t));
  //       Iot[1]->send_data(cmp, dim * sizeof(uint8_t));
  //       for (int i = 0; i < dim; i++)
  //       {
  //         cout<<i<<"-th:w:"<<"w_A:"<<+cmp[i]<<",w_B:"<<+tf[i]<<endl;
  //       }
  //       delete[] tf;
  //     }
  //     else
  //     {
  //       uint8_t *tf =  new uint8_t[dim];
  //       Iot[0]->send_data(cmp, dim * sizeof(uint8_t));
  //       Iot[1]->recv_data(tf, dim * sizeof(uint8_t));
  //       for (int i = 0; i < dim; i++)
  //       {
  //         cout<<i<<"-th:w:"<<"w_A:"<<+tf[i]<<",w_B:"<<+cmp[i]<<endl;
  //       }
  //       delete[] tf;
  //     }
  Aux->B2A(cmp, z, dim, lambda); // binary share to arithmetic share
  copy(z, z + dim, out.begin());
  // delete[] flp;
  delete[] b;
  delete[] w;
  delete[] z;
  delete[] eq;
  delete[] cmp;
}

void equality(int dim, uint64_t *in, uint64_t *&out, AuxProtocols *Aux)
{
  int bw_x = lambda;
  uint8_t *eq = new uint8_t[dim];

  Aux->equality(in, eq, dim, bw_x); // Bob equal than Alice ,result is 1. otherwise is 0.

  Aux->B2A(eq, out, dim, bw_x); // binary share to arithmetic share
  delete[] eq;
}

void loadP(vector<vector<uint32_t>> &p, vector<vector<uint32_t>> &zeroM, string data_path)
{
  ifstream inf;
  inf.open(data_path);
  string line;
  getline(inf, line);
  // cout << "o:" << line << endl;
  int row = 0;
  while (getline(inf, line))
  {
    istringstream iss(line);
    string cell;
    int col = 0;
    // while (getline(iss, cell, ','))
    // {
    //   lineArray.push_back(stoi(cell));
    // }
    while (getline(iss, cell, ','))
    {
      // int value;
      // istringstream(cell) >> value;
      p[row][col] = stoi(cell);
      if (p[row][col] == 0)
      {
        zeroM[row][col] = 0;
      }
      else
      {
        zeroM[row][col] = 1;
      }

      ++col;
    }
    ++row;
  }
  inf.close();
}

void Dom(vector<uint32_t> p1, vector<uint32_t> p2, vector<uint32_t> z1, vector<uint32_t> z2, int &rS)
{
  rS = 0;
  vector<uint32_t> cmp(m + 1);
  uint32_t sum1 = 0, sum2 = 0;
  for (int j = 0; j < m; ++j)
  {
    // zeroM[i][j]=0 means incomplete
    if (z1[j] == 0 || z2[j] == 0)
    {
      cmp[j] = 1;
    }
    else
    {
      if (p1[j] <= p2[j])
      {
        cmp[j] = 1;
      }
      sum1 += p1[j];
      sum2 += p2[j];
    }
  }
  if (sum1 < sum2)
  {
    cmp[m] = 1;
  }
  // cout << "[";
  // for (int j = 0; j <= m; ++j)
  // {
  //   cout << cmp[j] << ",";
  // }
  // cout << "]";
  if (accumulate(cmp.begin(), cmp.end(), 0) == m + 1)
  {
    rS = 1;
  }
}

void Dom_Relax(vector<uint32_t> p1, vector<uint32_t> p2, vector<uint32_t> z1, vector<uint32_t> z2, int &rS, uint64_t plain_mu)
{
  rS = 0;
  vector<uint32_t> cmp(m + 1);
  uint32_t sum1 = 0, sum2 = 0;
  int w = 0, v = 0;
  for (int j = 0; j < m + 1; ++j)
  {
    cmp[j] = 0;
  }
  for (int j = 0; j < m; ++j)
  {
    // zeroM[i][j]=0 means incomplete
    if (z1[j] == 0 || z2[j] == 0)
    {
      cmp[j] = 1;
    }
    else
    {
      w++;
      if (p1[j] <= p2[j])
      {
        cmp[j] = 1;
      }
      sum1 += p1[j];
      sum2 += p2[j];
    }
  }
  if (sum1 < sum2)
  {
    cmp[m] = 1;
  }
  // cout << "[";
  // for (int j = 0; j <= m; ++j)
  // {
  //   cout << cmp[j] << ",";
  // }
  // cout << "]";
  if (accumulate(cmp.begin(), cmp.end(), 0) == m + 1)
  {
    rS = 1;
  }
  if (plain_mu<=w)
  {
    v = 1;
  }
  rS = v * rS;
}

int plainB(vector<vector<uint32_t>> pt, vector<vector<uint32_t>> zeroM, vector<uint32_t> q)
{
  double startT = omp_get_wtime();
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<uint32_t> ind(n);
  iota(begin(ind), end(ind), 0);
  // process
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < m; ++j)
    {
      // q[j]=1 means large, q[j]=0 means less
      p[i][j] = (1 - q[j]) * pt[i][j] + q[j] * (domain - pt[i][j]);
    }
  }
  // for (int i = 0; i < n; ++i)
  // {
  //   for (int j = 0; j < m; j++)
  //   {
  //     cout << zeroM[i][j] << "\t";
  //   }
  //   cout << endl;
  // }
  // for (int i = 0; i < n; ++i)
  // {
  //   for (int j = 0; j < m; j++)
  //   {
  //     cout << pt[i][j] << "\t";
  //   }
  //   cout << endl;
  // }
  // for (int i = 0; i < n; ++i)
  // {
  //   for (int j = 0; j < m; j++)
  //   {
  //     cout << p[i][j] << "\t";
  //   }
  //   cout << endl;
  // }
  int countDom = 0;
  for (int i = 0; i < n; ++i)
  {
    if (!ind.empty())
    {
      auto it = begin(ind);
      while (it != end(ind))
      {
        int rs = 0;
        Dom(p[i], p[*it], zeroM[i], zeroM[*it], rs);
        // cout << i << " to " << (*it) << "-th:" << rs << endl;
        if (rs == 1)
        {
          it = ind.erase(it);
        }
        else
        {
          ++it;
        }
        countDom++;
      }
    }
  }
  double endT = omp_get_wtime();
  double time = endT - startT;
  cout << "Query Time\t" << RED << time << " s" << RESET << endl;
  cout << "len:" << ind.size() << ", num:" << countDom << endl;
  if (!ind.empty())
  {
    auto it = begin(ind);
    while (it != end(ind))
    {
      cout << "[";
      for (int j = 0; j < m - 1; j++)
      {
        cout << pt[*it][j] << ",";
      }
      cout << pt[*it][m - 1] << "];";
      ++it;
    }
  }
  cout << endl;
  return ind.size();
}

int plainB_min(vector<vector<uint32_t>> pt, vector<vector<uint32_t>> zeroM, vector<uint32_t> q)
{
  double startT = omp_get_wtime();
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<uint32_t> ind(n);
  iota(begin(ind), end(ind), 0);
  // process
  uint64_t *min_pos = new uint64_t[n];
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < m; ++j)
    {
      // q[j]=1 means large, q[j]=0 means less
      p[i][j] = (1 - q[j]) * pt[i][j] + q[j] * (domain - pt[i][j]);
    }
    min_pos[i] = 0;
    for (int j = 0; j < m; ++j)
    {
      min_pos[i] = (min_pos[i] + zeroM[i][j] * p[i][j]) & mask;
    }
  }
  // for (int i = 0; i < n; ++i)
  // {
  //   for (int j = 0; j < m; j++)
  //   {
  //     cout << zeroM[i][j] << "\t";
  //   }
  //   cout << endl;
  // }
  // for (int i = 0; i < n; ++i)
  // {
  //   for (int j = 0; j < m; j++)
  //   {
  //     cout << pt[i][j] << "\t";
  //   }
  //   cout << endl;
  // }
  // for (int i = 0; i < n; ++i)
  // {
  //   for (int j = 0; j < m; j++)
  //   {
  //     cout << p[i][j] << "\t";
  //   }
  //   cout << endl;
  // }
  countDom = 0;
  // unordered_set<uint32_t> pos;
  vector<int> indexArray(n);
  iota(begin(indexArray), end(indexArray), 0);
  auto compare = [&min_pos](int i, int j)
  {
    return min_pos[i] < min_pos[j];
  };
  sort(indexArray.begin(), indexArray.end(), compare);
  int pos_i = 0;
  for (int i = 0; i < n; ++i)
  {
    pos_i = indexArray[i];
    if (!ind.empty())
    {
      auto it = begin(ind);
      while (it != end(ind))
      {
        int rs = 0;
        Dom(p[pos_i], p[*it], zeroM[pos_i], zeroM[*it], rs);
        // cout << i << " to " << (*it) << "-th:" << rs << endl;
        if (rs == 1)
        {
          it = ind.erase(it);
        }
        else
        {
          ++it;
        }
        countDom++;
      }
    }
  }
  double endT = omp_get_wtime();
  double time = endT - startT;
  cout << "Query Time\t" << RED << time << " s" << RESET << endl;
  int len = ind.size();
  cout << "len:" << ind.size() << ", num:" << countDom << endl;
  if (!ind.empty())
  {
    auto it = begin(ind);
    while (it != end(ind))
    {
      cout << "[";
      for (int j = 0; j < m - 1; j++)
      {
        cout << pt[*it][j] << ",";
      }
      cout << pt[*it][m - 1] << "];";
      ++it;
    }
  }
  cout << endl;
  return len;
}

int plainB_Dim(vector<vector<uint32_t>> pt, vector<vector<uint32_t>> zeroM, vector<uint32_t> q, uint64_t plain_mu)
{
  double startT = omp_get_wtime();
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<uint32_t> ind(n);
  iota(begin(ind), end(ind), 0);
  // process
  uint64_t *min_pos = new uint64_t[n];
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < m; ++j)
    {
      // q[j]=1 means large, q[j]=0 means less
      p[i][j] = (1 - q[j]) * pt[i][j] + q[j] * (domain - pt[i][j]);
    }
    min_pos[i] = 0;
    for (int j = 0; j < m; ++j)
    {
      min_pos[i] = (min_pos[i] + zeroM[i][j] * p[i][j]) & mask;
    }
  }
  countDom = 0;
  // unordered_set<uint32_t> pos;
  vector<int> indexArray(n);
  iota(begin(indexArray), end(indexArray), 0);
  auto compare = [&min_pos](int i, int j)
  {
    return min_pos[i] < min_pos[j];
  };
  sort(indexArray.begin(), indexArray.end(), compare);
  int pos_i = 0;
  for (int i = 0; i < n; ++i)
  {
    pos_i = indexArray[i];
    if (!ind.empty())
    {
      auto it = begin(ind);
      while (it != end(ind))
      {
        int rs = 0;
        Dom_Relax(p[pos_i], p[*it], zeroM[pos_i], zeroM[*it], rs, plain_mu);
        if (rs == 1)
        {
          it = ind.erase(it);
        }
        else
        {
          ++it;
        }
        countDom++;
      }
    }
  }
  double endT = omp_get_wtime();
  double time = endT - startT;
  cout << "Query Time\t" << RED << time << " s" << RESET << endl;
  int len = ind.size();
  cout << "len:" << ind.size() << ", num:" << countDom << endl;
  if (!ind.empty())
  {
    auto it = begin(ind);
    while (it != end(ind))
    {
      cout << "[";
      for (int j = 0; j < m - 1; j++)
      {
        cout << pt[*it][j] << ",";
      }
      cout << pt[*it][m - 1] << "];";
      ++it;
    }
  }
  cout << endl;
  return len;
}

int plainB_Dom(vector<vector<uint32_t>> pt, vector<vector<uint32_t>> zeroM, vector<uint32_t> q, uint64_t plain_theta)
{
  double startT = omp_get_wtime();
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<uint32_t> ind(n);
  iota(begin(ind), end(ind), 0);
  // process
  uint64_t *min_pos = new uint64_t[n];
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < m; ++j)
    {
      // q[j]=1 means large, q[j]=0 means less
      p[i][j] = (1 - q[j]) * pt[i][j] + q[j] * (domain - pt[i][j]);
    }
    min_pos[i] = 0;
    for (int j = 0; j < m; ++j)
    {
      min_pos[i] = (min_pos[i] + zeroM[i][j] * p[i][j]) & mask;
    }
  }
  countDom = 0;
  // unordered_set<uint32_t> pos;
  vector<int> indexArray(n);
  iota(begin(indexArray), end(indexArray), 0);
  auto compare = [&min_pos](int i, int j)
  {
    return min_pos[i] < min_pos[j];
  };
  sort(indexArray.begin(), indexArray.end(), compare);
  int pos_i = 0;
  vector<uint64_t> xi(n, 0);
  for (int i = 0; i < n; ++i)
  {
    pos_i = indexArray[i];
    if (!ind.empty())
    {
      auto it = begin(ind);
      while (it != end(ind))
      {
        int rs = 0;
        Dom(p[pos_i], p[*it], zeroM[pos_i], zeroM[*it], rs);
        // cout << i << " to " << (*it) << "-th:" << rs << endl;
        xi[*it] += rs;
        if (xi[*it]>plain_theta)
        {
          it = ind.erase(it);
        }
        else
        {
          ++it;
        }
        countDom++;
      }
    }
  }
  double endT = omp_get_wtime();
  double time = endT - startT;
  cout << "Query Time\t" << RED << time << " s" << RESET << endl;
  int len = ind.size();
  cout << "len:" << ind.size() << ", num:" << countDom << endl;
  if (!ind.empty())
  {
    auto it = begin(ind);
    while (it != end(ind))
    {
      cout << "[";
      for (int j = 0; j < m - 1; j++)
      {
        cout << pt[*it][j] << ",";
      }
      cout << pt[*it][m - 1] << "];";
      ++it;
    }
  }
  cout << endl;
  return len;
}

int plainF(vector<vector<uint32_t>> pt, vector<vector<uint32_t>> zeroM, vector<uint32_t> q)
{
  double startT = omp_get_wtime();
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<uint32_t> ind(n, 0);
  // process
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < m; ++j)
    {
      // q[j]=1 means large, q[j]=0 means less
      p[i][j] = (1 - q[j]) * pt[i][j] + q[j] * (domain - pt[i][j]);
    }
  }
  countDom = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      if (i == j)
      {
        continue;
      }
      int rs = 0;
      Dom(p[i], p[j], zeroM[i], zeroM[j], rs);
      // cout << i << " to " << j << "-th:" << rs << endl;
      ind[j] += rs;
      countDom++;
    }
  }
  double endT = omp_get_wtime();
  double timeT = endT - startT;
  cout << "Query Time\t" << RED << timeT << " s" << RESET << endl;
  int len = count(ind.begin(), ind.end(), 0);
  cout << "len:" << count(ind.begin(), ind.end(), 0) << ", num:" << countDom << endl;
  // for (int i = 0; i < n; ++i)
  // {
  //   cout << ind[i] << "\t";
  // }
  // cout << endl;
  for (int i = 0; i < n; ++i)
  {
    if (ind[i] == 0)
    {
      cout << "[";
      for (int j = 0; j < m - 1; j++)
      {
        cout << pt[i][j] << ",";
      }
      cout << pt[i][m - 1] << "];";
    }
  }
  cout << endl;
  return len;
}

int plainF_Dim(vector<vector<uint32_t>> pt, vector<vector<uint32_t>> zeroM, vector<uint32_t> q, uint64_t plain_mu)
{
  double startT = omp_get_wtime();
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<uint32_t> ind(n, 0);
  // process
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < m; ++j)
    {
      // q[j]=1 means large, q[j]=0 means less
      p[i][j] = (1 - q[j]) * pt[i][j] + q[j] * (domain - pt[i][j]);
    }
  }
  countDom = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      if (i == j)
      {
        continue;
      }
      int rs = 0;
      Dom_Relax(p[i], p[j], zeroM[i], zeroM[j], rs, plain_mu);
      // cout << i << " to " << j << "-th:" << rs << endl;
      ind[j] += rs;
      countDom++;
    }
  }
  double endT = omp_get_wtime();
  double timeT = endT - startT;
  cout << "Query Time\t" << RED << timeT << " s" << RESET << endl;
  int len = count(ind.begin(), ind.end(), 0);
  cout << "len:" << count(ind.begin(), ind.end(), 0) << ", num:" << countDom << endl;
  // for (int i = 0; i < n; ++i)
  // {
  //   cout << ind[i] << "\t";
  // }
  // cout << endl;
  for (int i = 0; i < n; ++i)
  {
    if (ind[i] == 0)
    {
      cout << "[";
      for (int j = 0; j < m - 1; j++)
      {
        cout << pt[i][j] << ",";
      }
      cout << pt[i][m - 1] << "];";
    }
  }
  cout << endl;
  return len;
}

int plainF_Dom(vector<vector<uint32_t>> pt, vector<vector<uint32_t>> zeroM, vector<uint32_t> q, uint64_t plain_theta)
{
  double startT = omp_get_wtime();
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<uint32_t> ind(n, 0);
  // process
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < m; ++j)
    {
      // q[j]=1 means large, q[j]=0 means less
      p[i][j] = (1 - q[j]) * pt[i][j] + q[j] * (domain - pt[i][j]);
    }
  }
  countDom = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      if (i == j)
      {
        continue;
      }
      int rs = 0;
      Dom(p[i], p[j], zeroM[i], zeroM[j], rs);
      // cout << i << " to " << j << "-th:" << rs << endl;
      ind[j] += rs;
      countDom++;
    }
  }
  double endT = omp_get_wtime();
  double timeT = endT - startT;
  cout << "Query Time\t" << RED << timeT << " s" << RESET << endl;
  vector<uint64_t> xi(n, 0);
  for (int j = 0; j < n; ++j)
  {
    if (ind[j]<=plain_theta)
    {
      xi[j] = 1;
    }
  }
  int len = count(xi.begin(), xi.end(), 1);
  cout << "len:" << len << ", num:" << countDom << endl;
  // for (int i = 0; i < n; ++i)
  // {
  //   cout << ind[i] << "\t";
  // }
  // cout << endl;
  for (int i = 0; i < n; ++i)
  {
    if (xi[i] == 1)
    {
      cout << "[";
      for (int j = 0; j < m - 1; j++)
      {
        cout << pt[i][j] << ",";
      }
      cout << pt[i][m - 1] << "];";
    }
  }
  cout << endl;
  return len;
}

void Prod_H(int dim, uint64_t *inA, uint64_t *inB, uint64_t *&out, LinearOT *Prod)
{
  int bw_x = lambda;
  // Prod->hadamard_product(dim, inA, inB, out, bw_x, bw_x, bw_x, false);
  Prod->hadamard_cross_terms(dim, inA, inB, out, bw_x, bw_x, bw_x);
  for (int i = 0; i < dim; i++)
  {
    out[i] = (out[i] + inA[i] * inB[i]) & mask;
  }
}

void Prod_M(int dim1, int dim2, int dim3, uint64_t *inA, uint64_t *inB, uint64_t *&out, LinearOT *Prod, MultMode mode)
{
  /*
  uint8_t *msbA = nullptr;
  uint8_t *msbB = nullptr;
  int bw_x = lambda;
  uint64_t *outC = new uint64_t[dim1 * dim2 * dim3];
  Prod->matrix_multiplication(dim1, dim2, dim3, inA, inB, outC, bw_x, bw_x, bw_x, false, true, false, mode, msbA, msbB);
  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim3; j++) {
      out[i * dim3 + j] = 0;
      for (int k = 0; k < dim2; k++) {
        out[i * dim3 + j] = (out[i * dim3 + j] + outC[dim2 * dim3 * i + dim2 * j + k]) & mask;
      }
    }
  }
  delete []outC;
  */
  int32_t bw_x = 64;
  uint64_t *outC = new uint64_t[dim1 * dim2 * dim3];
  // Prod->matrix_multiplication(dim1, dim2, dim3, inA, inB, outC, bw_x, bw_x, bw_x, false, false, false, mode);
  Prod->matmul_cross_terms(dim1, dim2, dim3, inA, inB, outC, bw_x, bw_x, bw_x, false, mode);
  uint64_t *local_terms = new uint64_t[dim1 * dim2 * dim3];
  // uint64_t *local_terms = new uint64_t[dim1 * dim3];
  if (party == ALICE &&
      (mode == MultMode::Alice_has_A || mode == MultMode::Alice_has_B))
  {
    Prod->matmul_cleartext(dim1, dim2, dim3, inA, inB, local_terms, false);
  }
  else if (party == BOB &&
           (mode == MultMode::Bob_has_A || mode == MultMode::Bob_has_B))
  {
    Prod->matmul_cleartext(dim1, dim2, dim3, inA, inB, local_terms, false);
  }
  else if (mode == MultMode::None)
  {
    Prod->matmul_cleartext(dim1, dim2, dim3, inA, inB, local_terms, false);
  }
  else
  {
    memset(local_terms, 0, dim1 * dim2 * dim3 * sizeof(uint64_t));
  }
  // for (int i = 0; i < dim1 * dim3; i++) {
  //   out[i] = (out[i] + local_terms[i]) & mask;
  // }
  for (int i = 0; i < dim1; i++)
  {
    for (int j = 0; j < dim3; j++)
    {
      out[i * dim3 + j] = 0;
      for (int k = 0; k < dim2; k++)
      {
        out[i * dim3 + j] = (out[i * dim3 + j] + local_terms[dim2 * dim3 * i + dim2 * j + k] + outC[dim2 * dim3 * i + dim2 * j + k]) & mask;
      }
    }
  }
  delete[] outC;
  delete[] local_terms;
}

uint64_t HashP(uint64_t in)
{
  block128 inB = toBlock(in);
  // block128 CRH.H(block128)
  block128 outB = crh.H(inB);
  uint64_t out = (uint64_t)_mm_extract_epi64(outB, 0);
  // cout<<(uint64_t)_mm_extract_epi64(outB, 0)<<endl;
  // cout<<(uint64_t)_mm_extract_epi64(outB, 1)<<endl;
  return out;
}

void share_point(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q)
{
  SS_p = new uint64_t *[n];
  SS_zeroM = new uint64_t *[n];
  SS_q = new uint64_t[m];
  uint64_t *x = new uint64_t[m];
  uint64_t *y = new uint64_t[m];
  uint64_t *z = new uint64_t[m];
  for (int i = 0; i < n; ++i)
  {
    SS_p[i] = new uint64_t[m];
    SS_zeroM[i] = new uint64_t[m];
    prg.random_data(x, m * sizeof(uint64_t));
    prg.random_data(y, m * sizeof(uint64_t));
    if (party == ALICE)
    {
      Iot[0]->send_data(x, m * sizeof(uint64_t));
      Iot[1]->recv_data(SS_zeroM[i], m * sizeof(uint64_t));
      for (int j = 0; j < m; j++)
      {
        SS_p[i][j] = (p[i][j] - x[j]) & mask;
      }
    }
    else
    {
      Iot[0]->recv_data(SS_p[i], m * sizeof(uint64_t));
      Iot[1]->send_data(y, m * sizeof(uint64_t));
      for (int j = 0; j < m; j++)
      {
        SS_zeroM[i][j] = (zeroM[i][j] - y[j]) & mask;
      }
    }
  }
  prg.random_data(z, m * sizeof(uint64_t));
  if (party == ALICE)
  {
    Iot[0]->send_data(z, m * sizeof(uint64_t));
    for (int j = 0; j < m; j++)
    {
      SS_q[j] = (q[j] - z[j]) & mask;
    }
  }
  else
  {
    Iot[0]->recv_data(SS_q, m * sizeof(uint64_t));
  }
  delete[] x;
  delete[] y;
  delete[] z;
}

void share_point_bucket(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q)
{
  vector<uint64_t> SS_pt(m);
  vector<uint64_t> SS_zeroMt(m);
  SS_q = new uint64_t[m];
  uint64_t *SS_1 = new uint64_t[m];
  uint64_t *SS_2 = new uint64_t[m];
  unordered_map<uint32_t, uint32_t> keyMapping;
  uint32_t T_ind = 0;
  for (int i = 0; i < n; ++i)
  {
    prg.random_data(SS_1, m * sizeof(uint64_t));
    if (party == ALICE)
    {
      Iot[0]->send_data(SS_1, m * sizeof(uint64_t));
      for (int j = 0; j < m; j++)
      {
        SS_1[j] = (p[i][j] - SS_1[j]) & mask;
      }
    }
    else
    {
      Iot[0]->recv_data(SS_1, m * sizeof(uint64_t));
    }
    uint64_t sum = 0;
    for (int j = 0; j < m; ++j)
    {
      sum += zeroM[i][j] * pow(2, m - 1 - j);
    }
    uint32_t index = 0;
    if (keyMapping.find(sum) != keyMapping.end())
    {
      index = keyMapping[sum];
    }
    else
    {
      keyMapping.insert({sum, T_ind});
      index = T_ind;
      T_ind++;
    }
    auto it = SS_pB.find(index);
    if (it != SS_pB.end())
    {
      copy(SS_1, SS_1 + m, SS_pt.begin());
      it->second.push_back(SS_pt);
    }
    else
    {
      prg.random_data(SS_2, m * sizeof(uint64_t));
      if (party == ALICE)
      {
        Iot[0]->recv_data(SS_2, m * sizeof(uint64_t));
      }
      else
      {
        Iot[0]->send_data(SS_2, m * sizeof(uint64_t));
        for (int j = 0; j < m; j++)
        {
          SS_2[j] = (zeroM[i][j] - SS_2[j]) & mask;
        }
      }
      vector<vector<uint64_t>> St;
      copy(SS_2, SS_2 + m, SS_zeroMt.begin());
      copy(SS_1, SS_1 + m, SS_pt.begin());
      St.push_back(SS_zeroMt);
      St.push_back(SS_pt);
      // SS_pB.insert({index, St});
      SS_pB[index] = St;
    }
  }
  // q
  prg.random_data(SS_q, m * sizeof(uint64_t));
  if (party == ALICE)
  {
    Iot[0]->send_data(SS_q, m * sizeof(uint64_t));
    for (int j = 0; j < m; j++)
    {
      SS_q[j] = (q[j] - SS_q[j]) & mask;
    }
  }
  else
  {
    Iot[0]->recv_data(SS_q, m * sizeof(uint64_t));
  }
  delete[] SS_1;
  delete[] SS_2;
}

void SS_Dom_Complete(uint64_t *p1, uint64_t *p2, uint64_t &rS)
{
  rS = 0;
  vector<uint32_t> cmp(m + 1);
  uint64_t *delta = new uint64_t[m];
  uint64_t *st = new uint64_t[m];
  uint64_t *t1 = new uint64_t[1];
  uint64_t *t2 = new uint64_t[1];
  uint64_t *sigma = new uint64_t[1];
  comparison(m, p1, p2, delta, Auxt[0]);
  t1[0] = 0;
  t2[0] = 0;
  for (int j = 0; j < m; ++j)
  {
    // S_p2 <= S_p1 means S_p2 - S_p1 <= 0
    st[j] = (p2[j] - p1[j]) & mask;
  }
  for (int j = 0; j < m; ++j)
  {
    t2[0] = (t2[0] + st[j]) & mask;
  }
  comparison(1, t2, t1, sigma, Auxt[0]);
  sigma[0] = (SS_one - sigma[0]) & mask;
  uint64_t *rt = new uint64_t[1];
  t1[0] = sigma[0];
  for (int j = 0; j < m; ++j)
  {
    t2[0] = delta[j];
    Prod_H(1, t1, t2, rt, Prodt[0]);
    t1[0] = rt[0];
  }
  rS = t1[0];
  delete[] delta;
  delete[] st;
  delete[] t1;
  delete[] t2;
  delete[] rt;
}

void SS_Dom_bymul(uint64_t *p1, uint64_t *p2, uint64_t *z1, uint64_t *z2, uint64_t &rS)
{
  rS = 0;
  vector<uint32_t> cmp(m + 1);
  uint64_t *delta = new uint64_t[m];
  uint64_t *beta = new uint64_t[m];
  uint64_t *phi = new uint64_t[m];
  uint64_t *st1 = new uint64_t[m];
  uint64_t *st2 = new uint64_t[m];
  uint64_t *t1 = new uint64_t[1];
  uint64_t *t2 = new uint64_t[1];
  uint64_t *sigma = new uint64_t[1];
  comparison(m, p1, p2, delta, Auxt[0]);
  // z[j]=0 means incomplete
  Prod_H(m, z1, z2, beta, Prodt[0]);
  Prod_H(m, beta, delta, phi, Prodt[0]);
  // Ret(ind, m);
  t1[0] = 0;
  t2[0] = 0;
  for (int j = 0; j < m; ++j)
  {
    // S_p2 <= S_p1 means S_p2 - S_p1 <= 0
    st1[j] = (p2[j] - p1[j]) & mask;
    phi[j] = (phi[j] + SS_one - beta[j]) & mask;
  }
  Prod_H(m, beta, st1, st2, Prodt[0]);
  for (int j = 0; j < m; ++j)
  {
    t2[0] = (t2[0] + st2[j]) & mask;
  }
  comparison(1, t2, t1, sigma, Auxt[0]);
  sigma[0] = (SS_one - sigma[0]) & mask;
  // Ret(delta, m);
  // Ret(beta, m);
  // Ret(phi, m);
  // Ret(sigma, 1);
  // mul-, first choice
  uint64_t *rt = new uint64_t[1];
  t1[0] = sigma[0];
  for (int j = 0; j < m; ++j)
  {
    t2[0] = phi[j];
    Prod_H(1, t1, t2, rt, Prodt[0]);
    t1[0] = rt[0];
  }
  rS = t1[0];
  delete[] delta;
  delete[] beta;
  delete[] phi;
  delete[] st1;
  delete[] st2;
  delete[] t1;
  delete[] t2;
  delete[] rt;
}

void SS_Dom_bymul_Relax(uint64_t *p1, uint64_t *p2, uint64_t *z1, uint64_t *z2, uint64_t &rS, uint64_t *mu)
{
  rS = 0;
  vector<uint32_t> cmp(m + 1);
  uint64_t *delta = new uint64_t[m];
  uint64_t *beta = new uint64_t[m];
  uint64_t *phi = new uint64_t[m];
  uint64_t *st1 = new uint64_t[m];
  uint64_t *st2 = new uint64_t[m];
  uint64_t *t1 = new uint64_t[1];
  uint64_t *t2 = new uint64_t[1];
  uint64_t *sigma = new uint64_t[1];
  comparison(m, p1, p2, delta, Auxt[0]);
  // z[j]=0 means incomplete
  Prod_H(m, z1, z2, beta, Prodt[0]);
  Prod_H(m, beta, delta, phi, Prodt[0]);
  // Ret(ind, m);
  t1[0] = 0;
  t2[0] = 0;
  for (int j = 0; j < m; ++j)
  {
    // S_p2 <= S_p1 means S_p2 - S_p1 <= 0
    st1[j] = (p2[j] - p1[j]) & mask;
    phi[j] = (phi[j] + SS_one - beta[j]) & mask;
  }
  Prod_H(m, beta, st1, st2, Prodt[0]);
  for (int j = 0; j < m; ++j)
  {
    t2[0] = (t2[0] + st2[j]) & mask;
  }
  comparison(1, t2, t1, sigma, Auxt[0]);
  sigma[0] = (SS_one - sigma[0]) & mask;
  // Ret(delta, m);
  // Ret(beta, m);
  // Ret(phi, m);
  // Ret(sigma, 1);
  // mul-, first choice
  uint64_t *rt = new uint64_t[1];
  t1[0] = sigma[0];
  for (int j = 0; j < m; ++j)
  {
    t2[0] = phi[j];
    Prod_H(1, t1, t2, rt, Prodt[0]);
    t1[0] = rt[0];
  }
  uint64_t *w = new uint64_t[1];
  uint64_t *v = new uint64_t[1];
  w[0] = 0;
  for (int j = 0; j < m; ++j)
  {
    w[0] = (w[0] + beta[j]) & mask;
  }
  comparison(1, mu, w, v, Auxt[0]);
  Prod_H(1, v, t1, rt, Prodt[0]);
  rS = rt[0];
  delete[] delta;
  delete[] beta;
  delete[] phi;
  delete[] st1;
  delete[] st2;
  delete[] t1;
  delete[] t2;
  delete[] rt;
  delete[] w;
  delete[] v;
}

void SS_Dom_bymul_T(uint64_t *p1, uint64_t *p2, uint64_t *z1, uint64_t *z2, uint64_t &rS, int th)
{
  rS = 0;
  uint64_t *delta = new uint64_t[m];
  uint64_t *beta = new uint64_t[m];
  uint64_t *phi = new uint64_t[m];
  uint64_t *st1 = new uint64_t[m];
  uint64_t *st2 = new uint64_t[m];
  uint64_t *t1 = new uint64_t[1];
  uint64_t *t2 = new uint64_t[1];
  uint64_t *sigma = new uint64_t[1];
  comparison(m, p1, p2, delta, Auxt[th]);
  // z[j]=0 means incomplete
  Prod_H(m, z1, z2, beta, Prodt[th]);
  Prod_H(m, beta, delta, phi, Prodt[th]);
  // Ret(ind, m);
  t1[0] = 0;
  t2[0] = 0;
  for (int j = 0; j < m; ++j)
  {
    // S_p2 <= S_p1 means S_p2 - S_p1 <= 0
    st1[j] = (p2[j] - p1[j]) & mask;
    phi[j] = (phi[j] + SS_one - beta[j]) & mask;
  }
  // reduce a prod_H
  Prod_H(m, beta, st1, st2, Prodt[th]);
  for (int j = 0; j < m; ++j)
  {
    t2[0] = (t2[0] + st2[j]) & mask;
  }
  comparison(1, t2, t1, sigma, Auxt[th]);
  // S_p1 < S_p2 = 1- (S_p2 <= S_p1)
  sigma[0] = (SS_one - sigma[0]) & mask;
  // Ret(delta, m);
  // Ret(beta, m);
  // Ret(phi, m);
  // Ret(sigma, 1);
  // mul-, first choice
  uint64_t *rt = new uint64_t[1];
  t1[0] = sigma[0];
  for (int j = 0; j < m; ++j)
  {
    t2[0] = phi[j];
    Prod_H(1, t1, t2, rt, Prodt[th]);
    t1[0] = rt[0];
  }
  rS = t1[0];
  delete[] delta;
  delete[] beta;
  delete[] phi;
  delete[] st1;
  delete[] st2;
  delete[] t1;
  delete[] t2;
  delete[] rt;
}

void SS_Dom_bymul_T_Relax(uint64_t *p1, uint64_t *p2, uint64_t *z1, uint64_t *z2, uint64_t &rS, uint64_t *mu, int th)
{
  rS = 0;
  uint64_t *delta = new uint64_t[m];
  uint64_t *beta = new uint64_t[m];
  uint64_t *phi = new uint64_t[m];
  uint64_t *st1 = new uint64_t[m];
  uint64_t *st2 = new uint64_t[m];
  uint64_t *t1 = new uint64_t[1];
  uint64_t *t2 = new uint64_t[1];
  uint64_t *sigma = new uint64_t[1];
  comparison(m, p1, p2, delta, Auxt[th]);
  // z[j]=0 means incomplete
  Prod_H(m, z1, z2, beta, Prodt[th]);
  Prod_H(m, beta, delta, phi, Prodt[th]);
  // Ret(ind, m);
  t1[0] = 0;
  t2[0] = 0;
  for (int j = 0; j < m; ++j)
  {
    // S_p2 <= S_p1 means S_p2 - S_p1 <= 0
    st1[j] = (p2[j] - p1[j]) & mask;
    phi[j] = (phi[j] + SS_one - beta[j]) & mask;
  }
  // reduce a prod_H
  Prod_H(m, beta, st1, st2, Prodt[th]);
  for (int j = 0; j < m; ++j)
  {
    t2[0] = (t2[0] + st2[j]) & mask;
  }
  comparison(1, t2, t1, sigma, Auxt[th]);
  // S_p1 < S_p2 = 1- (S_p2 <= S_p1)
  sigma[0] = (SS_one - sigma[0]) & mask;
  // Ret(delta, m);
  // Ret(beta, m);
  // Ret(phi, m);
  // Ret(sigma, 1);
  // mul-, first choice
  uint64_t *rt = new uint64_t[1];
  t1[0] = sigma[0];
  for (int j = 0; j < m; ++j)
  {
    t2[0] = phi[j];
    Prod_H(1, t1, t2, rt, Prodt[th]);
    t1[0] = rt[0];
  }
  uint64_t *w = new uint64_t[1];
  uint64_t *v = new uint64_t[1];
  w[0] = 0;
  for (int j = 0; j < m; ++j)
  {
    w[0] = (w[0] + beta[j]) & mask;
  }
  comparison(1, mu, w, v, Auxt[th]);
  Prod_H(1, v, t1, rt, Prodt[th]);
  rS = rt[0];
  delete[] delta;
  delete[] beta;
  delete[] phi;
  delete[] st1;
  delete[] st2;
  delete[] t1;
  delete[] t2;
  delete[] rt;
  delete[] w;
  delete[] v;
}

void SS_Dom_byeql(uint64_t *p1, uint64_t *p2, uint64_t *z1, uint64_t *z2, uint64_t &rS)
{
  rS = 0;
  vector<uint32_t> cmp(m + 1);
  uint64_t *delta = new uint64_t[m];
  uint64_t *beta = new uint64_t[m];
  uint64_t *phi = new uint64_t[m];
  uint64_t *st1 = new uint64_t[m];
  uint64_t *st2 = new uint64_t[m];
  uint64_t *t1 = new uint64_t[1];
  uint64_t *t2 = new uint64_t[1];
  uint64_t *sigma = new uint64_t[1];
  comparison(m, p1, p2, delta, Auxt[0]);
  // z[j]=0 means incomplete
  Prod_H(m, z1, z2, beta, Prodt[0]);
  Prod_H(m, beta, p1, st1, Prodt[0]);
  Prod_H(m, beta, p2, st2, Prodt[0]);
  Prod_H(m, beta, delta, phi, Prodt[0]);
  for (int j = 0; j < m; ++j)
  {
    t1[0] = (t1[0] + st1[j]) & mask;
    t2[0] = (t2[0] + st2[j]) & mask;
    phi[j] = (phi[j] + SS_one - beta[j]) & mask;
  }
  comparison(1, t2, t1, sigma, Auxt[0]);
  sigma[0] = (SS_one - sigma[0]) & mask;
  // cmp-, second choice
  t1[0] = sigma[0];
  for (int j = 0; j < m; ++j)
  {
    t1[0] = (t1[0] + phi[j]) & mask;
  }
  if (party == ALICE)
  {
    t1[0] = (t1[0] - m - 1) & mask;
  }
  else
  {
    t1[0] = (0 - t1[0]) & mask;
  }
  equality(1, t1, t2, Auxt[0]);
  rS = t2[0];
  delete[] delta;
  delete[] beta;
  delete[] phi;
  delete[] st1;
  delete[] st2;
  delete[] t1;
  delete[] t2;
}

void BSS_minLoop(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q)
{
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  share_point(p, zeroM, q);
  // uint64_t rs = 0;
  // Ret(SS_q, m);
  // Ret(SS_p[6], m);
  // Ret(SS_p[1], m);
  // double start1 = omp_get_wtime();
  // for (int i = 0; i < 1000; i++)
  //   SS_Dom_byeql(SS_p[6], SS_p[1], SS_zeroM[6], SS_zeroM[1], rs);
  // double end1 = omp_get_wtime();
  // double time1 = end1 - start1;
  // Ret(&rs, 1);
  // double start2 = omp_get_wtime();
  // cout << "Query Time\t" << RED << time1 << " s" << RESET << endl;
  // for (int i = 0; i < 1000; i++)
  //   SS_Dom_bymul(SS_p[6], SS_p[1], SS_zeroM[6], SS_zeroM[1], rs);
  // double end2 = omp_get_wtime();
  // Ret(&rs, 1);
  // double time2 = end2 - start2;
  // cout << "Query Time\t" << RED << time2 << " s" << RESET << endl;

  double start = omp_get_wtime();
  uint64_t **SS_pq = new uint64_t *[n];
  vector<uint32_t> ind(n);
  iota(begin(ind), end(ind), 0);
  // process
  uint64_t *min_pos = new uint64_t[n];
  uint64_t *min_t = new uint64_t[m];
  for (int i = 0; i < n; ++i)
  {
    SS_pq[i] = new uint64_t[m];
    Prod_H(m, SS_p[i], SS_q, SS_pq[i], Prodt[0]); // q[j] * p[i][j]
    for (int j = 0; j < m; ++j)
    {
      // q[j]=1 means large, q[j]=0 means less, (1 - q[j]) * p[i][j] + q[j] * (domain - p[i][j]);
      SS_pq[i][j] = (SS_p[i][j] + domain * SS_q[j] - 2 * SS_pq[i][j]) & mask;
    }
    Prod_H(m, SS_zeroM[i], SS_pq[i], min_t, Prodt[0]);
    min_pos[i] = 0;
    for (int j = 0; j < m; ++j)
    {
      min_pos[i] = (min_pos[i] + min_t[j]) & mask;
    }
  }
  double phase = omp_get_wtime();
  cout << "process Time\t" << RED << (phase - start) << " s" << RESET << endl;
  int numCmp1 = 0;
  double t1 = omp_get_wtime(), t2 = 0;
  uint64_t *plain_rs = new uint64_t[1];
  uint64_t *ss_in = new uint64_t[2];
  uint64_t *ss_out = new uint64_t[2];
  uint64_t *ss_pos = new uint64_t[2];
  uint64_t *min_v = new uint64_t[1];
  uint64_t *min_i = new uint64_t[1];
  uint64_t *tmp_v = new uint64_t[1];
  unordered_set<uint32_t> pos;
  int pos_i = 0;
  for (int i = 0; i < n; ++i)
  {
    min_v[0] = domain * m;
    min_i[0] = n;
    for (int i1 = 0; i1 < n; ++i1)
    {
      if (pos.find(i1) != pos.end())
      {
        continue;
      }
      tmp_v[0] = min_pos[i1];
      comparison(1, tmp_v, min_v, ss_pos, Auxt[0]);
      // Ret(ss_pos, plain_min, 1);
      // if (plain_min[0] == 1)
      // {
      //   pos_i = i1;
      //   min_v = min_pos[i1];
      // }
      ss_in[0] = (min_pos[i1] - min_v[0]) & mask;
      if (party == ALICE)
      {
        ss_in[1] = (i1 - min_i[0]) & mask;
      }
      else
      {
        ss_in[1] = (0 - min_i[0]) & mask;
      }
      ss_pos[1] = ss_pos[0];
      Prod_H(2, ss_pos, ss_in, ss_out, Prodt[0]);
      min_v[0] = (ss_out[0] + min_v[0]) & mask;
      min_i[0] = (ss_out[1] + min_i[0]) & mask;
      // Ret(ss_pos, 2);
      // Ret(tmp_v, 1);
      // Ret(min_i, 1);
      // Ret(min_v, 1);
    }
    Ret(min_i, ss_pos, 1);
    pos_i = ss_pos[0];
    pos.insert(pos_i);
    cout << pos_i << ";";
    if (!ind.empty())
    {
      unordered_set<uint32_t> indt;
      auto it = begin(ind);
      while (it != end(ind))
      {
        uint64_t rs = 0;
        SS_Dom_bymul(SS_pq[pos_i], SS_pq[*it], SS_zeroM[pos_i], SS_zeroM[*it], rs);
        Ret(&rs, plain_rs, 1);
        // cout << i << " to " << (*it) << "-th:" << plain_rs[0] << endl;
        if (plain_rs[0] == 1)
        {
          indt.insert(*it);
        }
        ++numCmp1;
        ++it;
      }
      it = begin(ind);
      while (it != end(ind))
      {
        if (indt.find(*it) != indt.end())
        {
          it = ind.erase(it);
        }
        else
        {
          ++it;
        }
      }
    }
    if (i % 10 == 0 && i != 0)
    {
      t2 = omp_get_wtime();
      cout << i << "th:" << ind.size() << ", " << (t2 - t1) << " s" << endl;
      t1 = omp_get_wtime();
    }
  }
  cout << "len:" << ind.size() << endl;
  if (!ind.empty())
  {
    auto it = begin(ind);
    while (it != end(ind))
    {
      Ret(SS_p[*it], m);
      cout << endl;
      ++it;
    }
  }
  cout << "num:" << numCmp1 << endl;
  double end = omp_get_wtime();
  double time = end - start;
  cout << "Query Time\t" << RED << time << " s" << RESET << endl;
}

int partition(vector<uint64_t> arr, vector<uint64_t> &index, vector<int> ind, vector<int> &ind1, vector<int> &ind2, int offset_p)
{
  // cout << "offset:" << offset_p << endl;
  int size = accumulate(ind.begin(), ind.end(), 0);
  uint64_t pos_t, ind_t;
  int j = 0;
  for (int i1 = 0; i1 < index.size(); ++i1)
  {
    if (ind[i1] != 0)
    {
      pos_t = arr[i1];
      ind_t = i1;
      break;
    }
  }
  if (size == 1)
  {
    index[offset_p] = ind_t;
    // cout << offset_p << "record-1:" << ind_t << endl;
    return offset_p;
  }
  else
  {
    uint64_t *arr_pos = new uint64_t[size - 1];
    uint64_t *ss_ind = new uint64_t[size - 1];
    vector<uint64_t> plain_ind(size - 1);
    uint64_t *pos = new uint64_t[size - 1];
    for (int i = ind_t + 1; i < index.size(); ++i)
    {
      if (ind[i] != 0)
      {
        arr_pos[j] = arr[i];
        j++;
      }
    }
    for (int i = 0; i < size - 1; ++i)
    {
      pos[i] = pos_t;
    }
    comparison(size - 1, arr_pos, pos, ss_ind, Auxt[0]);
    // Ret(ss_ind, size - 1);
    Ret(ss_ind, plain_ind, size - 1);
    j = 0;
    for (int i = ind_t + 1; i < index.size(); ++i)
    {
      if (ind[i] != 0)
      {
        if (plain_ind[j] == 1)
        {
          ind1[i] = 1;
        }
        else
        {
          ind2[i] = 1;
        }
        j++;
      }
    }
    int ind_pos = accumulate(plain_ind.begin(), plain_ind.end(), 0);
    index[offset_p + ind_pos] = ind_t;
    // cout << offset_p + ind_pos << "record-2:" << ind_t << endl;
    return offset_p + ind_pos;
  }
}

void quickSortWithIndex(vector<uint64_t> arr, vector<uint64_t> &index, vector<int> ind, int offset)
{
  vector<int> ind1(index.size(), 0);
  vector<int> ind2(index.size(), 0);
  int ind_pos = partition(arr, index, ind, ind1, ind2, offset);
  if ((accumulate(ind1.begin(), ind1.end(), 0) > 0))
  {
    quickSortWithIndex(arr, index, ind1, offset);
  }
  if ((accumulate(ind2.begin(), ind2.end(), 0) > 0))
  {
    quickSortWithIndex(arr, index, ind2, ind_pos + 1);
  }
}

int partition_T(vector<uint64_t> arr, vector<uint64_t> &index, vector<int> ind, vector<int> &ind1, vector<int> &ind2, int offset_p)
{
  // cout << "offset:" << offset_p << endl;
  int size = accumulate(ind.begin(), ind.end(), 0);
  uint64_t pos_t, ind_t;
  int j = 0;
  for (int i1 = 0; i1 < index.size(); ++i1)
  {
    if (ind[i1] != 0)
    {
      pos_t = arr[i1];
      ind_t = i1;
      break;
    }
  }
  if (size == 1)
  {
    index[offset_p] = ind_t;
    // cout << offset_p << "record-1:" << ind_t << endl;
    return offset_p;
  }
  else
  {
    uint64_t *arr_pos = new uint64_t[size - 1];
    uint64_t *ss_ind = new uint64_t[size - 1];
    vector<uint64_t> plain_ind(size - 1);
    uint64_t *pos = new uint64_t[size - 1];
    for (int i = ind_t + 1; i < index.size(); ++i)
    {
      if (ind[i] != 0)
      {
        arr_pos[j] = arr[i];
        j++;
      }
    }
    for (int i = 0; i < size - 1; ++i)
    {
      pos[i] = pos_t;
    }
    comparison(size - 1, arr_pos, pos, ss_ind, Auxt[0]);
    // Ret(ss_ind, size - 1);
    Ret(ss_ind, plain_ind, size - 1);
    j = 0;
    for (int i = ind_t + 1; i < index.size(); ++i)
    {
      if (ind[i] != 0)
      {
        if (plain_ind[j] == 1)
        {
          ind1[i] = 1;
        }
        else
        {
          ind2[i] = 1;
        }
        j++;
      }
    }
    int ind_pos = accumulate(plain_ind.begin(), plain_ind.end(), 0);
    index[offset_p + ind_pos] = ind_t;
    // cout << offset_p + ind_pos << "record-2:" << ind_t << endl;
    return offset_p + ind_pos;
  }
}

void quickSortWithIndex_T(vector<uint64_t> arr, vector<uint64_t> &index, vector<int> ind, int offset)
{
  vector<int> ind1(index.size(), 0);
  vector<int> ind2(index.size(), 0);
  int ind_pos = partition_T(arr, index, ind, ind1, ind2, offset);
#pragma omp parallel num_threads(2)
  {
// #pragma omp sections
#pragma omp single
    {
#pragma omp task firstprivate(arr, index)
      {
        if ((accumulate(ind1.begin(), ind1.end(), 0) > 0))
        {
          quickSortWithIndex_T(arr, index, ind1, offset);
        }
      }
#pragma omp task firstprivate(arr, index)
      {
        if ((accumulate(ind2.begin(), ind2.end(), 0) > 0))
        {
          quickSortWithIndex_T(arr, index, ind2, ind_pos + 1);
        }
      }
#pragma omp taskwait
    }
  }
}

int BSS_min(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q)
{
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  share_point(p, zeroM, q);
  double startT = omp_get_wtime();
  uint64_t **SS_pq = new uint64_t *[n];
  vector<uint32_t> ind(n);
  iota(begin(ind), end(ind), 0);
  // process
  vector<uint64_t> min_pos(n, 0);
  uint64_t *min_t = new uint64_t[m];
  for (int i = 0; i < n; ++i)
  {
    SS_pq[i] = new uint64_t[m];
    Prod_H(m, SS_p[i], SS_q, SS_pq[i], Prodt[0]); // q[j] * p[i][j]
    for (int j = 0; j < m; ++j)
    {
      // q[j]=1 means large, q[j]=0 means less, (1 - q[j]) * p[i][j] + q[j] * (domain - p[i][j]);
      SS_pq[i][j] = (SS_p[i][j] + domain * SS_q[j] - 2 * SS_pq[i][j]) & mask;
    }
    Prod_H(m, SS_zeroM[i], SS_pq[i], min_t, Prodt[0]);
    for (int j = 0; j < m; ++j)
    {
      min_pos[i] = (min_pos[i] + min_t[j]) & mask;
    }
  }
  double phaseT = omp_get_wtime();
  pre_time = phaseT - startT;
  cout << "process Time\t" << RED << pre_time << " s" << RESET << endl;
  int numCmp1 = 0;
  double t1 = omp_get_wtime();
  uint64_t *plain_rs = new uint64_t[1];
  vector<uint64_t> plain_pos(n);
  vector<int> ind_pos(n);
  fill(begin(ind_pos), end(ind_pos), 1);
  quickSortWithIndex(min_pos, plain_pos, ind_pos, 0);
  // for (int i = 0; i < n; ++i)
  // {
  //   cout << plain_pos[i] << ";";
  //   if (i % 10 == 0 && i != 0)
  //   {
  //     cout << endl;
  //   }
  // }
  // cout << endl;
  double t2 = omp_get_wtime(), t3 = 0;
  sort_time = t2 - t1;
  cout << "sort Time\t" << RED << sort_time << " s" << RESET << endl;
  int pos_i = 0;
  for (int i = 0; i < n; ++i)
  {
    pos_i = plain_pos[i];
    if (!ind.empty())
    {
      unordered_set<uint32_t> indt;
      auto it = begin(ind);
      while (it != end(ind))
      {
        uint64_t rs = 0;
        SS_Dom_bymul(SS_pq[pos_i], SS_pq[*it], SS_zeroM[pos_i], SS_zeroM[*it], rs);
        Ret(&rs, plain_rs, 1);
        // cout << i << " to " << (*it) << "-th:" << plain_rs[0] << endl;
        if (plain_rs[0] == 1)
        {
          indt.insert(*it);
        }
        ++numCmp1;
        ++it;
      }
      it = begin(ind);
      while (it != end(ind))
      {
        if (indt.find(*it) != indt.end())
        {
          it = ind.erase(it);
        }
        else
        {
          ++it;
        }
      }
      // cout << "delete:" << indt.size() << endl;
    }
    // if (i % 10 == 0 && i != 0)
    // {
    //   t3 = omp_get_wtime();
    //   cout << i << "th:" << numCmp1 << ", " <<  "rest:" << ind.size() << ", " << (t3 - t2) << " s" << endl;
    //   t2 = omp_get_wtime();
    // }
  }
  double endT = omp_get_wtime();
  que_time = endT - t1;
  cout << "Query Time\t" << RED << que_time << " s" << RESET << endl;
  int len = ind.size();
  cout << "len:" << ind.size() << ", num:" << numCmp1 << endl;
  if (!ind.empty())
  {
    auto it = begin(ind);
    while (it != end(ind))
    {
      Ret(SS_p[*it], m);
      cout << ";";
      ++it;
    }
  }
  cout << endl;
  return len;
}

int BSS(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q)
{
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  share_point(p, zeroM, q);
  // uint64_t rs = 0;
  // Ret(SS_q, m);
  // Ret(SS_p[6], m);
  // Ret(SS_p[1], m);
  // double start1 = omp_get_wtime();
  // for (int i = 0; i < 1000; i++)
  //   SS_Dom_byeql(SS_p[6], SS_p[1], SS_zeroM[6], SS_zeroM[1], rs);
  // double end1 = omp_get_wtime();
  // double time1 = end1 - start1;
  // Ret(&rs, 1);
  // double start2 = omp_get_wtime();
  // cout << "Query Time\t" << RED << time1 << " s" << RESET << endl;
  // for (int i = 0; i < 1000; i++)
  //   SS_Dom_bymul(SS_p[6], SS_p[1], SS_zeroM[6], SS_zeroM[1], rs);
  // double end2 = omp_get_wtime();
  // Ret(&rs, 1);
  // double time2 = end2 - start2;
  // cout << "Query Time\t" << RED << time2 << " s" << RESET << endl;

  double startT = omp_get_wtime();
  uint64_t **SS_pq = new uint64_t *[n];
  vector<uint32_t> ind(n);
  iota(begin(ind), end(ind), 0);
  // process
  for (int i = 0; i < n; ++i)
  {
    SS_pq[i] = new uint64_t[m];
    Prod_H(m, SS_p[i], SS_q, SS_pq[i], Prodt[0]); // q[j] * p[i][j]
    for (int j = 0; j < m; ++j)
    {
      // q[j]=1 means large, q[j]=0 means less, (1 - q[j]) * p[i][j] + q[j] * (domain - p[i][j]);
      SS_pq[i][j] = (SS_p[i][j] + domain * SS_q[j] - 2 * SS_pq[i][j]) & mask;
    }
  }
  double phaseT = omp_get_wtime();
  pre_time = phaseT - startT;
  cout << "process Time\t" << RED << pre_time << " s" << RESET << endl;
  int numCmp1 = 0;
  double t1 = omp_get_wtime(), t2 = 0;
  uint64_t *plain_rs = new uint64_t[1];
  for (int i = 0; i < n; ++i)
  {
    if (!ind.empty())
    {
      unordered_set<uint32_t> indt;
      auto it = begin(ind);
      while (it != end(ind))
      {
        uint64_t rs = 0;
        SS_Dom_bymul(SS_pq[i], SS_pq[*it], SS_zeroM[i], SS_zeroM[*it], rs);
        Ret(&rs, plain_rs, 1);
        // cout << i << " to " << (*it) << "-th:" << plain_rs[0] << endl;
        if (plain_rs[0] == 1)
        {
          indt.insert(*it);
        }
        ++numCmp1;
        ++it;
      }
      it = begin(ind);
      while (it != end(ind))
      {
        if (indt.find(*it) != indt.end())
        {
          it = ind.erase(it);
        }
        else
        {
          ++it;
        }
      }
    }
    // if (i % 10 == 0 && i != 0)
    // {
    //   t2 = omp_get_wtime();
    //   cout << i << "th:" << numCmp1 << ", " <<  "rest:" << ind.size() << ", " << (t2 - t1) << " s" << endl;
    //   t1 = omp_get_wtime();
    // }
  }
  double endT = omp_get_wtime();
  que_time = endT - t1;
  cout << "Query Time\t" << RED << que_time << " s" << RESET << endl;
  int len = ind.size();
  cout << "len:" << ind.size() << ", num:" << numCmp1 << endl;
  if (!ind.empty())
  {
    auto it = begin(ind);
    while (it != end(ind))
    {
      Ret(SS_p[*it], m);
      cout << ";";
      ++it;
    }
  }
  cout << endl;
  return len;
}

int BSS_min_complete(vector<vector<uint64_t>> &SS_pt, uint64_t **SS_pq, int len, vector<uint32_t> &ind)
{
  double satrtT = omp_get_wtime();
  vector<uint64_t> min_pos(len, 0);
  uint64_t *min_t = new uint64_t[m];
  for (int i = 0; i < len; ++i)
  {
    for (int j = 0; j < m; ++j)
    {
      min_pos[i] = (min_pos[i] + SS_pq[i][j]) & mask;
    }
  }
  double t1 = omp_get_wtime();
  uint64_t *plain_rs = new uint64_t[1];
  vector<uint64_t> plain_pos(len);
  vector<int> ind_pos(len);
  fill(begin(ind_pos), end(ind_pos), 1);
  quickSortWithIndex(min_pos, plain_pos, ind_pos, 0);
  double t2 = omp_get_wtime();
  double sort_t = t2 - t1;
  cout << "sort Time\t" << RED << sort_t << " s" << RESET << endl;
  int pos_i = 0;
  for (int i = 0; i < len; ++i)
  {
    pos_i = plain_pos[i];
    if (!ind.empty())
    {
      unordered_set<uint32_t> indt;
      auto it = begin(ind);
      while (it != end(ind))
      {
        uint64_t rs = 0;
        SS_Dom_Complete(SS_pq[pos_i], SS_pq[*it], rs);
        Ret(&rs, plain_rs, 1);
        // cout << i << " to " << (*it) << "-th:" << plain_rs[0] << endl;
        if (plain_rs[0] == 1)
        {
          indt.insert(*it);
        }
        ++countDom;
        ++it;
      }
      it = begin(ind);
      while (it != end(ind))
      {
        if (indt.find(*it) != indt.end())
        {
          it = ind.erase(it);
        }
        else
        {
          ++it;
        }
      }
    }
  }
  double endT = omp_get_wtime();
  cout << "Query Time in Bucket\t" << RED << (endT - satrtT) << " s" << RESET << endl;
  que_time += endT - satrtT;
  int size = ind.size();
  cout << "size:" << ind.size() << ", num:" << countDom << endl;
  // if (!ind.empty())
  // {
  //   auto it = begin(ind);
  //   while (it != end(ind))
  //   {
  //     cout << *it + 1 << ":";
  //     Ret(SS_pt[*it + 1], m);
  //     cout << ";";
  //     ++it;
  //   }
  // }
  // cout << endl;
  return size;
}

int BSS_min_bucket(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q)
{
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  share_point_bucket(p, zeroM, q);
  cout << "len_bucket:" << SS_pB.size() << endl;
  int size = 0;
  double startT = omp_get_wtime();
  vector<vector<uint32_t>> ind_bucket(SS_pB.size());
  vector<uint64_t **> SS_pqt(SS_pB.size());
  pre_time = 0;
  que_time = 0;
  for (int k = 0; k < SS_pB.size(); k++)
  {
    cout << "bucket-" << k + 1 << " th:" << endl;
    vector<vector<uint64_t>> t = SS_pB[k];
    int len = t.size();
    cout << "old size:" << len << endl;
    uint64_t **SS_pq = new uint64_t *[len - 1];
    uint64_t *SS_pt = new uint64_t[m];
    double t1 = omp_get_wtime();
    for (int i = 0; i < len - 1; ++i)
    {
      SS_pq[i] = new uint64_t[m];
      copy(t[i + 1].begin(), t[i + 1].end(), SS_pt); // first row is zeroM
      Prod_H(m, SS_pt, SS_q, SS_pq[i], Prodt[0]);    // q[j] * p[i][j]
      for (int j = 0; j < m; ++j)
      {
        // q[j]=1 means large, q[j]=0 means less, (1 - q[j]) * p[i][j] + q[j] * (domain - p[i][j]);
        SS_pq[i][j] = (SS_pt[j] + domain * SS_q[j] - 2 * SS_pq[i][j]) & mask;
      }
    }
    double t2 = omp_get_wtime();
    pre_time += t2 - t1;
    cout << "process Time\t" << RED << pre_time << " s" << RESET << endl;
    vector<uint32_t> indt(len - 1);
    iota(begin(indt), end(indt), 0);
    int sizet = BSS_min_complete(t, SS_pq, len - 1, indt);
    size += sizet;
    cout << "new size:" << sizet << endl;
    ind_bucket[k] = indt;
    SS_pqt[k] = SS_pq;
    delete[] SS_pt;
  }
  cout << "remaining size:" << size << endl;
  SS_p = new uint64_t *[size];
  SS_zeroM = new uint64_t *[size];
  uint64_t **SS_pq = new uint64_t *[size];
  int ll = 0;
  for (int k = 0; k < SS_pB.size(); k++)
  {
    auto it = begin(ind_bucket[k]);
    while (it != end(ind_bucket[k]))
    {
      int pos = *it;
      // cout << pos << endl;
      SS_p[ll] = new uint64_t[m];
      copy(SS_pB[k][pos + 1].begin(), SS_pB[k][pos + 1].end(), SS_p[ll]);
      // Ret(SS_pB[k][pos+1], m);
      SS_zeroM[ll] = new uint64_t[m];
      copy(SS_pB[k][0].begin(), SS_pB[k][0].end(), SS_zeroM[ll]);
      // Ret(SS_pB[k][0], m);
      SS_pq[ll] = new uint64_t[m];
      copy(SS_pqt[k][pos], SS_pqt[k][pos] + m, SS_pq[ll]);
      // Ret(SS_pqt[k][pos], m);
      ++it;
      ++ll;
    }
  }
  // for (int i = 0; i < size; ++i)
  // {
  //   Ret(SS_p[i], m);
  // }
  // cout << endl;
  // for (int i = 0; i < size; ++i)
  // {
  //   Ret(SS_pq[i], m);
  // }
  // cout << endl;
  // for (int i = 0; i < size; ++i)
  // {
  //   Ret(SS_zeroM[i], m);
  // }
  // cout << endl;
  double t1 = omp_get_wtime();
  vector<uint64_t> min_pos(size, 0);
  uint64_t *min_t = new uint64_t[m];
  for (int i = 0; i < size; ++i)
  {
    Prod_H(m, SS_zeroM[i], SS_pq[i], min_t, Prodt[0]);
    for (int j = 0; j < m; ++j)
    {
      min_pos[i] = (min_pos[i] + min_t[j]) & mask;
    }
  }
  int numCmp1 = countDom;
  uint64_t *plain_rs = new uint64_t[1];
  vector<uint64_t> plain_pos(size);
  vector<int> ind_pos(size);
  fill(begin(ind_pos), end(ind_pos), 1);
  quickSortWithIndex(min_pos, plain_pos, ind_pos, 0);
  double t2 = omp_get_wtime(), t3 = 0;
  sort_time = t2 - t1;
  cout << "sort Time\t" << RED << sort_time << " s" << RESET << endl;
  int pos_i = 0;
  vector<uint32_t> ind(size);
  iota(begin(ind), end(ind), 0);
  for (int i = 0; i < size; ++i)
  {
    pos_i = plain_pos[i];
    if (!ind.empty())
    {
      unordered_set<uint32_t> indt;
      auto it = begin(ind);
      while (it != end(ind))
      {
        uint64_t rs = 0;
        SS_Dom_bymul(SS_pq[pos_i], SS_pq[*it], SS_zeroM[pos_i], SS_zeroM[*it], rs);
        Ret(&rs, plain_rs, 1);
        // cout << i << " to " << (*it) << "-th:" << plain_rs[0] << endl;
        if (plain_rs[0] == 1)
        {
          indt.insert(*it);
        }
        ++numCmp1;
        ++it;
      }
      it = begin(ind);
      while (it != end(ind))
      {
        if (indt.find(*it) != indt.end())
        {
          it = ind.erase(it);
        }
        else
        {
          ++it;
        }
      }
    }
    // if (i % 100 == 0 && i != 0)
    // {
    //   t3 = omp_get_wtime();
    //   cout << i << "th:" << numCmp1 << ", " <<  "rest:" << ind.size() << ", " << (t3 - t2) << " s" << endl;
    //   t2 = omp_get_wtime();
    // }
  }
  double endT = omp_get_wtime();
  double que_time_buc = endT - t1;
  cout << "Query Time\t" << RED << que_time_buc << " s" << RESET << endl;
  que_time += que_time_buc;
  int lenT = ind.size();
  cout << "len:" << ind.size() << ", num:" << numCmp1 << endl;
  if (!ind.empty())
  {
    auto it = begin(ind);
    while (it != end(ind))
    {
      Ret(SS_p[*it], m);
      cout << ";";
      ++it;
    }
  }
  cout << endl;
  return lenT;
}

int BSS_min_T(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q)
{
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  share_point(p, zeroM, q);
  double startT = omp_get_wtime();
  uint64_t **SS_pq = new uint64_t *[n];
  vector<uint32_t> ind(n);
  iota(begin(ind), end(ind), 0);
  // process
  vector<uint64_t> min_pos(n, 0);
#pragma omp parallel num_threads(THs)
  {
#pragma omp single
    {
      for (int itr = 0; itr < THs; itr++)
      {
#pragma omp task firstprivate(itr, THs, mask, SS_p, SS_q, domain)
        {
          int lendt = (((n - 1) * itr) / THs) + 1;
          int lenut = ((n - 1) * (itr + 1)) / THs;
          if (itr == 0)
          {
            lendt = 0;
          }
          for (int i = lendt; i <= lenut; i++)
          {
            uint64_t *min_t = new uint64_t[m];
            SS_pq[i] = new uint64_t[m];
            Prod_H(m, SS_p[i], SS_q, SS_pq[i], Prodt[itr]); // q[j] * p[i][j]
            for (int j = 0; j < m; ++j)
            {
              // q[j]=1 means large, q[j]=0 means less, (1 - q[j]) * p[i][j] + q[j] * (domain - p[i][j]);
              SS_pq[i][j] = (SS_p[i][j] + domain * SS_q[j] - 2 * SS_pq[i][j]) & mask;
            }
            Prod_H(m, SS_zeroM[i], SS_pq[i], min_t, Prodt[itr]);
            for (int j = 0; j < m; ++j)
            {
              min_pos[i] = (min_pos[i] + min_t[j]) & mask;
            }
            delete[] min_t;
          }
        }
      }
#pragma omp taskwait
    }
  }
  double phaseT = omp_get_wtime();
  pre_time = phaseT - startT;
  cout << "process Time\t" << RED << pre_time << " s" << RESET << endl;
  countDom = 0;
  double t1 = omp_get_wtime();
  vector<uint64_t> plain_pos(n);
  vector<int> ind_pos(n);
  fill(begin(ind_pos), end(ind_pos), 1);
  quickSortWithIndex(min_pos, plain_pos, ind_pos, 0);
  // for (int i = 0; i < n; ++i)
  // {
  //   cout << plain_pos[i] << ";";
  //   if (i % 10 == 0 && i != 0)
  //   {
  //     cout << endl;
  //   }
  // }
  // cout << endl;
  double t2 = omp_get_wtime(), t3 = 0;
  sort_time = t2 - t1;
  cout << "sort Time\t" << RED << sort_time << " s" << RESET << endl;
  int pos_i = 0;
  vector<uint64_t> xi(n, 0);
  for (int i = 0; i < n; ++i)
  {
    pos_i = plain_pos[i];
    if (!ind.empty())
    {
#pragma omp parallel num_threads(THs)
      {
#pragma omp single
        {
          for (int itr = 0; itr < THs; itr++)
          {
#pragma omp task firstprivate(itr, THs, SS_pq, SS_zeroM, ind, pos_i)
            {
              int lendt = (((ind.size() - 1) * itr) / THs) + 1;
              int lenut = ((ind.size() - 1) * (itr + 1)) / THs;
              if (itr == 0)
              {
                lendt = 0;
              }
              uint64_t *plain_rs = new uint64_t[1];
              // for (int j = lendt; j <= lenut; j++)
              // auto it = begin(ind);
              for (int it = lendt; it != lenut + 1; ++it)
              {
                uint64_t rs = 0;
                SS_Dom_bymul_T(SS_pq[pos_i], SS_pq[ind[it]], SS_zeroM[pos_i], SS_zeroM[ind[it]], rs, itr);
                // cout << pos_i<<"," << ind[it] << ";";
                // if (it == 1)
                // {
                //   uint64_t *t1 = new uint64_t[m];
                //   uint64_t *t2 = new uint64_t[m];
                //   uint64_t *t3 = new uint64_t[m];
                //   uint64_t *t4 = new uint64_t[m];
                //   Ret_T(SS_pq[pos_i], t1, m, itr);
                //   Ret_T(SS_pq[it], t2, m, itr);
                //   Ret_T(SS_zeroM[pos_i], t3, m, itr);
                //   Ret_T(SS_zeroM[it], t4, m, itr);
                //   for (int ii = 0; ii < m; ++ii)
                //     cout << t1[ii] << ";";
                //   cout << endl;
                //   for (int ii = 0; ii < m; ++ii)
                //     cout << t2[ii] << ";";
                //   cout << endl;
                //   for (int ii = 0; ii < m; ++ii)
                //     cout << t3[ii] << ";";
                //   cout << endl;
                //   for (int ii = 0; ii < m; ++ii)
                //     cout << t4[ii] << ";";
                //   cout << endl;
                //   Ret_T(&rs, plain_rs, 1, itr);
                //   cout << plain_rs[0] << endl;
                // }
                Ret_T(&rs, plain_rs, 1, itr);
                xi[ind[it]] += plain_rs[0];
                //   if (plain_rs[0] == 1)
                //   {
                //     // indt.insert(*it);
                //     indt[ind[it]] = 1;
                //   }
                ++countDom;
              }
            }
          }
#pragma omp taskwait
        }
      }
      auto it = begin(ind);
      while (it != end(ind))
      {
        if (xi[*it] > 0)
        {
          it = ind.erase(it);
        }
        else
        {
          ++it;
        }
      }
    }
    // if (i % 10 == 0 && i != 0)
    // {
    //   t3 = omp_get_wtime();
    //   cout << i << "th:" << countDom << ", "
    //        << "rest:" << ind.size() << ", " << (t3 - t2) << " s" << endl;
    //   t2 = omp_get_wtime();
    // }
  }
  double endT = omp_get_wtime();
  que_time = endT - t1;
  cout << "Query Time\t" << RED << que_time << " s" << RESET << endl;
  int len = ind.size();
  cout << "len:" << ind.size() << ", num:" << countDom << endl;
  if (!ind.empty())
  {
    auto it = begin(ind);
    while (it != end(ind))
    {
      Ret(SS_p[*it], m);
      cout << ";";
      ++it;
    }
  }
  cout << endl;
  return len;
}

int BSS_T_Dim(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q, uint64_t *mu)
{
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  share_point(p, zeroM, q);
  double startT = omp_get_wtime();
  uint64_t **SS_pq = new uint64_t *[n];
  vector<uint32_t> ind(n);
  iota(begin(ind), end(ind), 0);
  // process
  vector<uint64_t> min_pos(n, 0);
#pragma omp parallel num_threads(THs)
  {
#pragma omp single
    {
      for (int itr = 0; itr < THs; itr++)
      {
#pragma omp task firstprivate(itr, THs, mask, SS_p, SS_q, domain)
        {
          int lendt = (((n - 1) * itr) / THs) + 1;
          int lenut = ((n - 1) * (itr + 1)) / THs;
          if (itr == 0)
          {
            lendt = 0;
          }
          for (int i = lendt; i <= lenut; i++)
          {
            uint64_t *min_t = new uint64_t[m];
            SS_pq[i] = new uint64_t[m];
            Prod_H(m, SS_p[i], SS_q, SS_pq[i], Prodt[itr]); // q[j] * p[i][j]
            for (int j = 0; j < m; ++j)
            {
              // q[j]=1 means large, q[j]=0 means less, (1 - q[j]) * p[i][j] + q[j] * (domain - p[i][j]);
              SS_pq[i][j] = (SS_p[i][j] + domain * SS_q[j] - 2 * SS_pq[i][j]) & mask;
            }
            Prod_H(m, SS_zeroM[i], SS_pq[i], min_t, Prodt[itr]);
            for (int j = 0; j < m; ++j)
            {
              min_pos[i] = (min_pos[i] + min_t[j]) & mask;
            }
            delete[] min_t;
          }
        }
      }
#pragma omp taskwait
    }
  }
  double phaseT = omp_get_wtime();
  pre_time = phaseT - startT;
  cout << "process Time\t" << RED << pre_time << " s" << RESET << endl;
  countDom = 0;
  double t1 = omp_get_wtime();
  vector<uint64_t> plain_pos(n);
  vector<int> ind_pos(n);
  fill(begin(ind_pos), end(ind_pos), 1);
  quickSortWithIndex(min_pos, plain_pos, ind_pos, 0);
  double t2 = omp_get_wtime(), t3 = 0;
  sort_time = t2 - t1;
  cout << "sort Time\t" << RED << sort_time << " s" << RESET << endl;
  int pos_i = 0;
  vector<uint64_t> xi(n, 0);
  for (int i = 0; i < n; ++i)
  {
    pos_i = plain_pos[i];
    if (!ind.empty())
    {
#pragma omp parallel num_threads(THs)
      {
#pragma omp single
        {
          for (int itr = 0; itr < THs; itr++)
          {
#pragma omp task firstprivate(itr, THs, SS_pq, SS_zeroM, ind, pos_i)
            {
              int lendt = (((ind.size() - 1) * itr) / THs) + 1;
              int lenut = ((ind.size() - 1) * (itr + 1)) / THs;
              if (itr == 0)
              {
                lendt = 0;
              }
              uint64_t *plain_rs = new uint64_t[1];
              // for (int j = lendt; j <= lenut; j++)
              // auto it = begin(ind);
              for (int it = lendt; it != lenut + 1; ++it)
              {
                uint64_t rs = 0;
                SS_Dom_bymul_T_Relax(SS_pq[pos_i], SS_pq[ind[it]], SS_zeroM[pos_i], SS_zeroM[ind[it]], rs, mu, itr);
                Ret_T(&rs, plain_rs, 1, itr);
                xi[ind[it]] += plain_rs[0];
                ++countDom;
              }
            }
          }
#pragma omp taskwait
        }
      }
      auto it = begin(ind);
      while (it != end(ind))
      {
        if (xi[*it] > 0)
        {
          it = ind.erase(it);
        }
        else
        {
          ++it;
        }
      }
    }
    // if (i % 10 == 0 && i != 0)
    // {
    //   t3 = omp_get_wtime();
    //   cout << i << "th:" << countDom << ", "
    //        << "rest:" << ind.size() << ", " << (t3 - t2) << " s" << endl;
    //   t2 = omp_get_wtime();
    // }
  }
  double endT = omp_get_wtime();
  que_time = endT - t1;
  cout << "Query Time\t" << RED << que_time << " s" << RESET << endl;
  int len = ind.size();
  cout << "len:" << ind.size() << ", num:" << countDom << endl;
  if (!ind.empty())
  {
    auto it = begin(ind);
    while (it != end(ind))
    {
      Ret(SS_p[*it], m);
      cout << ";";
      ++it;
    }
  }
  cout << endl;
  return len;
}

int BSS_T_Dom(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q, uint64_t plain_theta)
{
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  share_point(p, zeroM, q);
  double startT = omp_get_wtime();
  uint64_t **SS_pq = new uint64_t *[n];
  vector<uint32_t> ind(n);
  iota(begin(ind), end(ind), 0);
  // process
  vector<uint64_t> min_pos(n, 0);
#pragma omp parallel num_threads(THs)
  {
#pragma omp single
    {
      for (int itr = 0; itr < THs; itr++)
      {
#pragma omp task firstprivate(itr, THs, mask, SS_p, SS_q, domain)
        {
          int lendt = (((n - 1) * itr) / THs) + 1;
          int lenut = ((n - 1) * (itr + 1)) / THs;
          if (itr == 0)
          {
            lendt = 0;
          }
          for (int i = lendt; i <= lenut; i++)
          {
            uint64_t *min_t = new uint64_t[m];
            SS_pq[i] = new uint64_t[m];
            Prod_H(m, SS_p[i], SS_q, SS_pq[i], Prodt[itr]); // q[j] * p[i][j]
            for (int j = 0; j < m; ++j)
            {
              // q[j]=1 means large, q[j]=0 means less, (1 - q[j]) * p[i][j] + q[j] * (domain - p[i][j]);
              SS_pq[i][j] = (SS_p[i][j] + domain * SS_q[j] - 2 * SS_pq[i][j]) & mask;
            }
            Prod_H(m, SS_zeroM[i], SS_pq[i], min_t, Prodt[itr]);
            for (int j = 0; j < m; ++j)
            {
              min_pos[i] = (min_pos[i] + min_t[j]) & mask;
            }
            delete[] min_t;
          }
        }
      }
#pragma omp taskwait
    }
  }
  double phaseT = omp_get_wtime();
  pre_time = phaseT - startT;
  cout << "process Time\t" << RED << pre_time << " s" << RESET << endl;
  countDom = 0;
  double t1 = omp_get_wtime();
  vector<uint64_t> plain_pos(n);
  vector<int> ind_pos(n);
  fill(begin(ind_pos), end(ind_pos), 1);
  quickSortWithIndex(min_pos, plain_pos, ind_pos, 0);
  double t2 = omp_get_wtime(), t3 = 0;
  sort_time = t2 - t1;
  cout << "sort Time\t" << RED << sort_time << " s" << RESET << endl;
  int pos_i = 0;
  vector<uint64_t> xi(n, 0);
  for (int i = 0; i < n; ++i)
  {
    pos_i = plain_pos[i];
    if (!ind.empty())
    {
#pragma omp parallel num_threads(THs)
      {
#pragma omp single
        {
          for (int itr = 0; itr < THs; itr++)
          {
#pragma omp task firstprivate(itr, THs, SS_pq, SS_zeroM, ind, pos_i)
            {
              int lendt = (((ind.size() - 1) * itr) / THs) + 1;
              int lenut = ((ind.size() - 1) * (itr + 1)) / THs;
              if (itr == 0)
              {
                lendt = 0;
              }
              uint64_t *plain_rs = new uint64_t[1];
              // for (int j = lendt; j <= lenut; j++)
              // auto it = begin(ind);
              for (int it = lendt; it != lenut + 1; ++it)
              {
                uint64_t rs = 0;
                SS_Dom_bymul_T(SS_pq[pos_i], SS_pq[ind[it]], SS_zeroM[pos_i], SS_zeroM[ind[it]], rs, itr);
                Ret_T(&rs, plain_rs, 1, itr);
                xi[ind[it]] += plain_rs[0];
                ++countDom;
              }
            }
          }
#pragma omp taskwait
        }
      }
      auto it = begin(ind);
      while (it != end(ind))
      {
        if (xi[*it]>plain_theta)
        {
          it = ind.erase(it);
        }
        else
        {
          ++it;
        }
      }
    }
    // if (i % 10 == 0 && i != 0)
    // {
    //   t3 = omp_get_wtime();
    //   cout << i << "th:" << countDom << ", "
    //        << "rest:" << ind.size() << ", " << (t3 - t2) << " s" << endl;
    //   t2 = omp_get_wtime();
    // }
  }
  double endT = omp_get_wtime();
  que_time = endT - t1;
  cout << "Query Time\t" << RED << que_time << " s" << RESET << endl;
  int len = ind.size();
  cout << "len:" << ind.size() << ", num:" << countDom << endl;
  if (!ind.empty())
  {
    auto it = begin(ind);
    while (it != end(ind))
    {
      Ret(SS_p[*it], m);
      cout << ";";
      ++it;
    }
  }
  cout << endl;
  return len;
}

int FSS(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q)
{
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  share_point(p, zeroM, q);
  double startT = omp_get_wtime();
  uint64_t **SS_pq = new uint64_t *[n];
  // process
  for (int i = 0; i < n; ++i)
  {
    SS_pq[i] = new uint64_t[m];
    Prod_H(m, SS_p[i], SS_q, SS_pq[i], Prodt[0]); // q[j] * p[i][j]
    for (int j = 0; j < m; ++j)
    {
      // q[j]=1 means large, q[j]=0 means less, (1 - q[j]) * p[i][j] + q[j] * (domain - p[i][j]);
      SS_pq[i][j] = (SS_p[i][j] + domain * SS_q[j] - 2 * SS_pq[i][j]) & mask;
    }
  }
  double phaseT = omp_get_wtime();
  pre_time = phaseT - startT;
  cout << "process Time\t" << RED << pre_time << " s" << RESET << endl;
  int numCmp1 = 0;
  double t1 = omp_get_wtime(), t2 = omp_get_wtime(), t3 = 0;
  uint64_t *indt = new uint64_t[n];
  uint64_t *ss_ind = new uint64_t[n];
  int offset = 3 - 2 * party; // +1 and -1
  for (int i = 0; i < n; ++i)
  {
    indt[i] = 0;
  }
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      if (i == j)
      {
        continue;
      }
      uint64_t rs = 0;
      SS_Dom_bymul(SS_pq[i], SS_pq[j], SS_zeroM[i], SS_zeroM[j], rs);
      // uint64_t *plain_rs = new uint64_t[1];
      // Ret(&rs, plain_rs, 1);
      // cout << i << " to " << j << "-th:" << plain_rs[0] << endl;
      // if (party == ALICE)
      // {
      //   indt[j] = (indt[j] + rs) & mask;
      // }
      // else
      // {
      //   indt[j] = (indt[j] - rs) & mask;
      // }
      indt[j] = (indt[j] + offset * rs) & mask;
      ++numCmp1;
    }
    if (i % 100 == 0 && i != 0)
    {
      t3 = omp_get_wtime();
      cout << i << "th:" << numCmp1 << ", " << (t3 - t2) << " s" << endl;
      t2 = omp_get_wtime();
    }
  }
  equality(n, indt, ss_ind, Auxt[0]);
  Skyline = new uint64_t *[n];
  uint64_t *index = new uint64_t[m];
  for (int i = 0; i < n; ++i)
  {
    Skyline[i] = new uint64_t[m];
    for (int j = 0; j < m; ++j)
    {
      index[j] = ss_ind[i];
    }
    Prod_H(m, SS_p[i], index, Skyline[i], Prodt[0]);
  }
  double endT = omp_get_wtime();
  que_time = endT - t1;
  cout << "Query Time\t" << RED << que_time << " s" << RESET << endl;
  vector<uint64_t> ind(n);
  Ret(ss_ind, ind, n);
  int len = count(ind.begin(), ind.end(), 1);
  cout << "len:" << len << ", num:" << numCmp1 << endl;
  for (int i = 0; i < n; ++i)
  {
    if (ind[i] == 1)
    {
      Ret(Skyline[i], m);
      cout << ";";
    }
  }
  cout << endl;
  delete[] indt;
  delete[] index;
  delete[] ss_ind;
  delete[] SS_pq;
  return len;
}

int FSS_Dim(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q, uint64_t *mu)
{
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  share_point(p, zeroM, q);
  double startT = omp_get_wtime();
  uint64_t **SS_pq = new uint64_t *[n];
  // process
  for (int i = 0; i < n; ++i)
  {
    SS_pq[i] = new uint64_t[m];
    Prod_H(m, SS_p[i], SS_q, SS_pq[i], Prodt[0]); // q[j] * p[i][j]
    for (int j = 0; j < m; ++j)
    {
      // q[j]=1 means large, q[j]=0 means less, (1 - q[j]) * p[i][j] + q[j] * (domain - p[i][j]);
      SS_pq[i][j] = (SS_p[i][j] + domain * SS_q[j] - 2 * SS_pq[i][j]) & mask;
    }
  }
  double phaseT = omp_get_wtime();
  pre_time = phaseT - startT;
  cout << "process Time\t" << RED << pre_time << " s" << RESET << endl;
  double t1 = omp_get_wtime(), t2 = omp_get_wtime(), t3 = 0;
  uint64_t *indt = new uint64_t[n];
  uint64_t *ss_ind = new uint64_t[n];
  int offset = 3 - 2 * party; // +1 and -1
  for (int i = 0; i < n; ++i)
  {
    indt[i] = 0;
  }
  countDom = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      if (i == j)
      {
        continue;
      }
      uint64_t rs = 0;
      SS_Dom_bymul_Relax(SS_pq[i], SS_pq[j], SS_zeroM[i], SS_zeroM[j], rs, mu);
      indt[j] = (indt[j] + offset * rs) & mask;
      ++countDom;
    }
    if (i % 100 == 0 && i != 0)
    {
      t3 = omp_get_wtime();
      cout << i << "th:" << countDom << ", " << (t3 - t2) << " s" << endl;
      t2 = omp_get_wtime();
    }
  }
  equality(n, indt, ss_ind, Auxt[0]);
  Skyline = new uint64_t *[n];
  uint64_t *index = new uint64_t[m];
  for (int i = 0; i < n; ++i)
  {
    Skyline[i] = new uint64_t[m];
    for (int j = 0; j < m; ++j)
    {
      index[j] = ss_ind[i];
    }
    Prod_H(m, SS_p[i], index, Skyline[i], Prodt[0]);
  }
  double endT = omp_get_wtime();
  que_time = endT - t1;
  cout << "Query Time\t" << RED << que_time << " s" << RESET << endl;
  vector<uint64_t> ind(n);
  Ret(ss_ind, ind, n);
  int len = count(ind.begin(), ind.end(), 1);
  cout << "len:" << len << ", num:" << countDom << endl;
  for (int i = 0; i < n; ++i)
  {
    if (ind[i] == 1)
    {
      Ret(Skyline[i], m);
      cout << ";";
    }
  }
  cout << endl;
  delete[] indt;
  delete[] index;
  delete[] ss_ind;
  delete[] SS_pq;
  return len;
}

int FSS_Dom(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q, uint64_t *theta)
{
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  share_point(p, zeroM, q);
  double startT = omp_get_wtime();
  uint64_t **SS_pq = new uint64_t *[n];
  // process
  for (int i = 0; i < n; ++i)
  {
    SS_pq[i] = new uint64_t[m];
    Prod_H(m, SS_p[i], SS_q, SS_pq[i], Prodt[0]); // q[j] * p[i][j]
    for (int j = 0; j < m; ++j)
    {
      // q[j]=1 means large, q[j]=0 means less, (1 - q[j]) * p[i][j] + q[j] * (domain - p[i][j]);
      SS_pq[i][j] = (SS_p[i][j] + domain * SS_q[j] - 2 * SS_pq[i][j]) & mask;
    }
  }
  double phaseT = omp_get_wtime();
  pre_time = phaseT - startT;
  cout << "process Time\t" << RED << pre_time << " s" << RESET << endl;
  double t1 = omp_get_wtime(), t2 = omp_get_wtime(), t3 = 0;
  uint64_t *indt = new uint64_t[n];
  uint64_t *ss_ind = new uint64_t[n];
  for (int j = 0; j < n; ++j)
  {
    indt[j] = 0;
    ss_ind[j] = theta[0];
  }
  countDom = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      if (i == j)
      {
        continue;
      }
      uint64_t rs = 0;
      SS_Dom_bymul(SS_pq[i], SS_pq[j], SS_zeroM[i], SS_zeroM[j], rs);
      indt[j] = (indt[j] + rs) & mask;
      ++countDom;
    }
    if (i % 100 == 0 && i != 0)
    {
      t3 = omp_get_wtime();
      cout << i << "th:" << countDom << ", " << (t3 - t2) << " s" << endl;
      t2 = omp_get_wtime();
    }
  }
  uint64_t *xi = new uint64_t[n];
  comparison(n, indt, ss_ind, xi, Auxt[0]);
  Skyline = new uint64_t *[n];
  uint64_t *index = new uint64_t[m];
  for (int i = 0; i < n; ++i)
  {
    Skyline[i] = new uint64_t[m];
    for (int j = 0; j < m; ++j)
    {
      index[j] = xi[i];
    }
    Prod_H(m, SS_p[i], index, Skyline[i], Prodt[0]);
  }
  double endT = omp_get_wtime();
  que_time = endT - t1;
  cout << "Query Time\t" << RED << que_time << " s" << RESET << endl;
  vector<uint64_t> ind(n);
  Ret(xi, ind, n);
  int len = count(ind.begin(), ind.end(), 1);
  cout << "len:" << len << ", num:" << countDom << endl;
  for (int i = 0; i < n; ++i)
  {
    if (ind[i] == 1)
    {
      Ret(Skyline[i], m);
      cout << ";";
    }
  }
  cout << endl;
  delete[] indt;
  delete[] index;
  delete[] ss_ind;
  delete[] SS_pq;
  return len;
}

int FSS_T(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q)
{
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  share_point(p, zeroM, q);
  double startT = omp_get_wtime();
  uint64_t **SS_pq = new uint64_t *[n];
  // process
#pragma omp parallel num_threads(THs)
  {
#pragma omp single
    {
      for (int itr = 0; itr < THs; itr++)
      {
#pragma omp task firstprivate(itr, THs, mask, SS_p, SS_q, domain)
        {
          int lendt = (((n - 1) * itr) / THs) + 1;
          int lenut = ((n - 1) * (itr + 1)) / THs;
          if (itr == 0)
          {
            lendt = 0;
          }
          for (int i = lendt; i <= lenut; i++)
          {
            SS_pq[i] = new uint64_t[m];
            Prod_H(m, SS_p[i], SS_q, SS_pq[i], Prodt[itr]); // q[j] * p[i][j]
            for (int j = 0; j < m; ++j)
            {
              // q[j]=1 means large, q[j]=0 means less, (1 - q[j]) * p[i][j] + q[j] * (domain - p[i][j]);
              SS_pq[i][j] = (SS_p[i][j] + domain * SS_q[j] - 2 * SS_pq[i][j]) & mask;
            }
          }
        }
      }
#pragma omp taskwait
    }
  }
  double phaseT = omp_get_wtime();
  pre_time = phaseT - startT;
  cout << "process Time\t" << RED << pre_time << " s" << RESET << endl;
  countDom = 0;
  double t1 = omp_get_wtime(), t2 = omp_get_wtime(), t3 = 0;
  uint64_t *indt = new uint64_t[n];
  uint64_t *ss_ind = new uint64_t[n];
  for (int j = 0; j < n; ++j)
  {
    indt[j] = 0;
  }
  int offset = 3 - 2 * party; // +1 and -1
#pragma omp parallel num_threads(THs)
  {
#pragma omp single
    {
      for (int itr = 0; itr < THs; itr++)
      {
#pragma omp task firstprivate(itr, THs, indt, SS_pq, SS_zeroM, offset)
        {
          int lendt = (((n - 1) * itr) / THs) + 1;
          int lenut = ((n - 1) * (itr + 1)) / THs;
          if (itr == 0)
          {
            lendt = 0;
          }
          for (int j = lendt; j <= lenut; ++j)
          {
            for (int i = 0; i < n; i++)
            {
              if (i == j)
              {
                continue;
              }
              uint64_t rs = 0;
              SS_Dom_bymul_T(SS_pq[i], SS_pq[j], SS_zeroM[i], SS_zeroM[j], rs, itr);
              // uint64_t *plain_rs = new uint64_t[1];
              // Ret(&rs, plain_rs, 1);
              // cout << i << " to " << j << "-th:" << plain_rs[0] << endl;
              // indt[j] = (indt[j] + rs)& mask;
              // if (party == ALICE)
              // {
              //   indt[j] = (indt[j] + rs) & mask;
              // }
              // else
              // {
              //   indt[j] = (indt[j] - rs) & mask;
              // }
              indt[j] = (indt[j] + offset * rs) & mask;
              ++countDom;
            }
            if (j % 100 == 0 && j != 0)
            {
              t3 = omp_get_wtime();
              cout << j << "th:" << (t3 - t2) << " s" << endl;
              t2 = omp_get_wtime();
            }
          }
        }
      }
#pragma omp taskwait
    }
  }
  equality(n, indt, ss_ind, Auxt[0]);
  Skyline = new uint64_t *[n];
  uint64_t *index = new uint64_t[m];
  for (int i = 0; i < n; ++i)
  {
    Skyline[i] = new uint64_t[m];
    for (int j = 0; j < m; ++j)
    {
      index[j] = ss_ind[i];
    }
    Prod_H(m, SS_p[i], index, Skyline[i], Prodt[0]);
  }
  double endT = omp_get_wtime();
  que_time = endT - t1;
  cout << "Query Time\t" << RED << que_time << " s" << RESET << endl;
  vector<uint64_t> ind(n);
  Ret(ss_ind, ind, n);
  int len = count(ind.begin(), ind.end(), 1);
  cout << "len:" << len << ", num:" << countDom << endl;
  for (int i = 0; i < n; ++i)
  {
    if (ind[i] == 1)
    {
      Ret(Skyline[i], m);
      cout << ";";
    }
  }
  cout << endl;
  delete[] indt;
  delete[] index;
  delete[] ss_ind;
  delete[] SS_pq;
  return len;
}

int FSS_T_Dim(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q, uint64_t *mu)
{
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  share_point(p, zeroM, q);
  double startT = omp_get_wtime();
  uint64_t **SS_pq = new uint64_t *[n];
  // process
#pragma omp parallel num_threads(THs)
  {
#pragma omp single
    {
      for (int itr = 0; itr < THs; itr++)
      {
#pragma omp task firstprivate(itr, THs, mask, SS_p, SS_q, domain)
        {
          int lendt = (((n - 1) * itr) / THs) + 1;
          int lenut = ((n - 1) * (itr + 1)) / THs;
          if (itr == 0)
          {
            lendt = 0;
          }
          for (int i = lendt; i <= lenut; i++)
          {
            SS_pq[i] = new uint64_t[m];
            Prod_H(m, SS_p[i], SS_q, SS_pq[i], Prodt[itr]); // q[j] * p[i][j]
            for (int j = 0; j < m; ++j)
            {
              // q[j]=1 means large, q[j]=0 means less, (1 - q[j]) * p[i][j] + q[j] * (domain - p[i][j]);
              SS_pq[i][j] = (SS_p[i][j] + domain * SS_q[j] - 2 * SS_pq[i][j]) & mask;
            }
          }
        }
      }
#pragma omp taskwait
    }
  }
  double phaseT = omp_get_wtime();
  pre_time = phaseT - startT;
  cout << "process Time\t" << RED << pre_time << " s" << RESET << endl;
  countDom = 0;
  double t1 = omp_get_wtime(), t2 = omp_get_wtime(), t3 = 0;
  uint64_t *indt = new uint64_t[n];
  uint64_t *ss_ind = new uint64_t[n];
  for (int j = 0; j < n; ++j)
  {
    indt[j] = 0;
  }
  int offset = 3 - 2 * party; // +1 and -1
#pragma omp parallel num_threads(THs)
  {
#pragma omp single
    {
      for (int itr = 0; itr < THs; itr++)
      {
#pragma omp task firstprivate(itr, THs, indt, SS_pq, SS_zeroM, offset)
        {
          int lendt = (((n - 1) * itr) / THs) + 1;
          int lenut = ((n - 1) * (itr + 1)) / THs;
          if (itr == 0)
          {
            lendt = 0;
          }
          for (int j = lendt; j <= lenut; ++j)
          {
            for (int i = 0; i < n; i++)
            {
              if (i == j)
              {
                continue;
              }
              uint64_t rs = 0;
              SS_Dom_bymul_T_Relax(SS_pq[i], SS_pq[j], SS_zeroM[i], SS_zeroM[j], rs, mu, itr);
              indt[j] = (indt[j] + offset * rs) & mask;
              ++countDom;
            }
            if (j % 100 == 0 && j != 0)
            {
              t3 = omp_get_wtime();
              cout << j << "th:" << (t3 - t2) << " s" << endl;
              t2 = omp_get_wtime();
            }
          }
        }
      }
#pragma omp taskwait
    }
  }
  equality(n, indt, ss_ind, Auxt[0]);
  Skyline = new uint64_t *[n];
  uint64_t *index = new uint64_t[m];
  for (int i = 0; i < n; ++i)
  {
    Skyline[i] = new uint64_t[m];
    for (int j = 0; j < m; ++j)
    {
      index[j] = ss_ind[i];
    }
    Prod_H(m, SS_p[i], index, Skyline[i], Prodt[0]);
  }
  double endT = omp_get_wtime();
  que_time = endT - t1;
  cout << "Query Time\t" << RED << que_time << " s" << RESET << endl;
  vector<uint64_t> ind(n);
  Ret(ss_ind, ind, n);
  int len = count(ind.begin(), ind.end(), 1);
  cout << "len:" << len << ", num:" << countDom << endl;
  for (int i = 0; i < n; ++i)
  {
    if (ind[i] == 1)
    {
      Ret(Skyline[i], m);
      cout << ";";
    }
  }
  cout << endl;
  delete[] indt;
  delete[] index;
  delete[] ss_ind;
  delete[] SS_pq;
  return len;
}

int FSS_T_Dom(vector<vector<uint32_t>> p, vector<vector<uint32_t>> zeroM, vector<uint32_t> q, uint64_t *theta)
{
  if (party == ALICE)
  {
    prg.random_data(&SS_one, sizeof(uint64_t));
    Iot[0]->send_data(&SS_one, sizeof(uint64_t));
    Iot[1]->recv_data(&SS_zero, sizeof(uint64_t));
    SS_zero = (0 - SS_zero) & mask;
  }
  else
  {
    prg.random_data(&SS_zero, sizeof(uint64_t));
    Iot[0]->recv_data(&SS_one, sizeof(uint64_t));
    Iot[1]->send_data(&SS_zero, sizeof(uint64_t));
    SS_one = (1 - SS_one) & mask;
  }
  share_point(p, zeroM, q);
  double startT = omp_get_wtime();
  uint64_t **SS_pq = new uint64_t *[n];
  // process
#pragma omp parallel num_threads(THs)
  {
#pragma omp single
    {
      for (int itr = 0; itr < THs; itr++)
      {
#pragma omp task firstprivate(itr, THs, mask, SS_p, SS_q, domain)
        {
          int lendt = (((n - 1) * itr) / THs) + 1;
          int lenut = ((n - 1) * (itr + 1)) / THs;
          if (itr == 0)
          {
            lendt = 0;
          }
          for (int i = lendt; i <= lenut; i++)
          {
            SS_pq[i] = new uint64_t[m];
            Prod_H(m, SS_p[i], SS_q, SS_pq[i], Prodt[itr]); // q[j] * p[i][j]
            for (int j = 0; j < m; ++j)
            {
              // q[j]=1 means large, q[j]=0 means less, (1 - q[j]) * p[i][j] + q[j] * (domain - p[i][j]);
              SS_pq[i][j] = (SS_p[i][j] + domain * SS_q[j] - 2 * SS_pq[i][j]) & mask;
            }
          }
        }
      }
#pragma omp taskwait
    }
  }
  double phaseT = omp_get_wtime();
  pre_time = phaseT - startT;
  cout << "process Time\t" << RED << pre_time << " s" << RESET << endl;
  countDom = 0;
  double t1 = omp_get_wtime(), t2 = omp_get_wtime(), t3 = 0;
  uint64_t *indt = new uint64_t[n];
  uint64_t *ss_ind = new uint64_t[n];
  uint64_t *xi = new uint64_t[n];
  for (int j = 0; j < n; ++j)
  {
    indt[j] = 0;
    xi[j] = theta[0];
  }
#pragma omp parallel num_threads(THs)
  {
#pragma omp single
    {
      for (int itr = 0; itr < THs; itr++)
      {
#pragma omp task firstprivate(itr, THs, indt, SS_pq, SS_zeroM)
        {
          int lendt = (((n - 1) * itr) / THs) + 1;
          int lenut = ((n - 1) * (itr + 1)) / THs;
          if (itr == 0)
          {
            lendt = 0;
          }
          for (int j = lendt; j <= lenut; ++j)
          {
            for (int i = 0; i < n; i++)
            {
              if (i == j)
              {
                continue;
              }
              uint64_t rs = 0;
              SS_Dom_bymul_T(SS_pq[i], SS_pq[j], SS_zeroM[i], SS_zeroM[j], rs, itr);
              indt[j] = (indt[j] + rs) & mask;
              ++countDom;
            }
            if (j % 100 == 0 && j != 0)
            {
              t3 = omp_get_wtime();
              cout << j << "th:" << (t3 - t2) << " s" << endl;
              t2 = omp_get_wtime();
            }
          }
        }
      }
#pragma omp taskwait
    }
  }
  comparison(n, indt, xi, ss_ind, Auxt[0]);
  Skyline = new uint64_t *[n];
  uint64_t *index = new uint64_t[m];
  for (int i = 0; i < n; ++i)
  {
    Skyline[i] = new uint64_t[m];
    for (int j = 0; j < m; ++j)
    {
      index[j] = ss_ind[i];
    }
    Prod_H(m, SS_p[i], index, Skyline[i], Prodt[0]);
  }
  double endT = omp_get_wtime();
  que_time = endT - t1;
  cout << "Query Time\t" << RED << que_time << " s" << RESET << endl;
  vector<uint64_t> ind(n);
  Ret(ss_ind, ind, n);
  int len = count(ind.begin(), ind.end(), 1);
  cout << "len:" << len << ", num:" << countDom << endl;
  for (int i = 0; i < n; ++i)
  {
    if (ind[i] == 1)
    {
      Ret(Skyline[i], m);
      cout << ";";
    }
  }
  cout << endl;
  delete[] indt;
  delete[] index;
  delete[] ss_ind;
  delete[] SS_pq;
  return len;
}

void test()
{
  int t = 15;
  uint64_t x[t] = {2, 3, 5, 2, 1, 25, 4, 85, 8, 96, 4, 5, 6, 785, 2};
  vector<uint64_t> min_pos(t, 0);
  vector<uint64_t> plain_pos(t);
  vector<int> ind_pos(t);
  fill(begin(ind_pos), end(ind_pos), 1);
  for (int i = 0; i < t; i++)
  {
    min_pos[i] = x[i];
  }
  quickSortWithIndex(min_pos, plain_pos, ind_pos, 0);
  vector<int> indexArray(t);
  iota(begin(indexArray), end(indexArray), 0);
  auto compare = [&min_pos](int i, int j)
  {
    return min_pos[i] <= min_pos[j];
  };
  sort(indexArray.begin(), indexArray.end(), compare);
  for (int i = 0; i < indexArray.size(); ++i)
  {
    cout << indexArray[i] << ",";
  }
  cout << endl;
  for (int i = 0; i < plain_pos.size(); ++i)
  {
    cout << plain_pos[i] << ",";
  }
  cout << endl;
}

// skyline query
int mainS_O(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("Pdim", Pd, "data dim");
  amap.arg("Psize", Pst, "data size");
  amap.arg("Prate", Pr, "data missing rate");
  amap.arg("Pname", Pn, "data name");
  amap.arg("Pthread", Pth, "omp multi-thread");
  amap.parse(argc, argv);
  if (Pth == 0)
  {
    THs = 2;
  }
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0) + party);
  // shar();
  cout << "Party: CS_" << party << endl;

  vector<int> input_dim = {2, 3, 4, 5, 6};                                // size = 4
  vector<int> input_size = {1000, 3000, 5000, 7000, 9000, 11000, 100000}; // size = 6
  vector<int> rate = {10, 20, 30, 40, 50};                                // size = 4
  vector<string> path = {"small_anti_corr.txt", "small_correlate.txt", "small_uniformly.txt", "nba.txt", "hotel.txt"};
  // Pd = 0, Pst = 0, Pr = 0, Pn = 4;

  m = input_dim[Pd];
  n = input_size[Pst];
  int r = rate[Pr];
  filename = path[Pn];
  data_path = "./data/data" + to_string(m) + "/size=" + to_string(n) + "/rate=" + to_string(r) + "/" + filename;
  // m = 3;
  // n = 9;
  // data_path = "./10.txt";
  cout << data_path << endl;
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<vector<uint32_t>> zeroM(n, vector<uint32_t>(m));
  loadP(p, zeroM, data_path);
  double Qtime = 0, QCom = 0, Qpre = 0, Qque = 0, Qsort = 0, Qlen = 0;
  int itrs = 3;
  for (int itr = 0; itr < itrs; itr++)
  {
    vector<uint32_t> q(m);
    for (int j = 0; j < m; j++)
    {
      q[j] = rand() % 2;
      // q[j] = 1;
    }
    if (party == ALICE)
    {
      uint32_t *q0 = new uint32_t[m];
      for (int j = 0; j < m; j++)
      {
        q0[j] = q[j];
      }
      Iot[0]->send_data(q0, m * sizeof(uint32_t));
      delete[] q0;
    }
    else
    {
      uint32_t *q0 = new uint32_t[m];
      Iot[0]->recv_data(q0, m * sizeof(uint32_t));
      for (int j = 0; j < m; j++)
      {
        q[j] = q0[j];
      }
      delete[] q0;
    }
    cout << itr << "-itr:\t" << endl;
    cout << "q: ";
    for (int j = 0; j < m; j++)
    {
      cout << q[j] << "\t";
    }
    cout << endl;
    plainB(p, zeroM, q);
    // plainF(p, zeroM, q);
    cout << "+++" << endl;
    // BSS(p, zeroM, q);
    // BSS_minPre(p, zeroM, q);
    // BSS_T(p, zeroM, q);
    uint64_t comm_start = 0;
    for (int j = 0; j < THs; j++)
    {
      comm_start += Iot[j]->counter;
    }
    double start = omp_get_wtime();
    int len;
    if (Pth == 0)
    {
      len = BSS_min(p, zeroM, q);
    }
    else
    {
      len = BSS_min_T(p, zeroM, q);
    }
    // int len = BSS_min(p, zeroM, q);
    // int len = BSS_min_T(p, zeroM, q);
    // int len = BSS_min_bucket(p, zeroM, q);
    // int len = FSS(p, zeroM, q);
    // int len = FSS_T(p, zeroM, q);
    double end = omp_get_wtime();
    Qtime += end - start;
    uint64_t comm_end = 0;
    for (int j = 0; j < THs; j++)
    {
      comm_end += Iot[j]->counter;
    }
    QCom += comm_end - comm_start;
    Qpre += pre_time;
    Qsort += sort_time;
    Qque += que_time;
    Qlen += len;
    delete[] SS_p;
    delete[] SS_zeroM;
    delete[] Skyline;
    delete[] SS_q;
    SS_pB.clear();
  }
  Qtime = Qtime / itrs;
  QCom = QCom / itrs / 1024 / 1024;
  Qpre = Qpre / itrs;
  Qsort = Qsort / itrs;
  Qque = Qque / itrs;
  Qlen = Qlen / itrs;
  cout << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) << endl;
  cout << "CS_" + to_string(party) + " Skyline Number\t" << RED << Qlen << RESET << endl;
  cout << "CS_" + to_string(party) + " Total Time\t" << RED << Qtime << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " process Time\t" << RED << Qpre << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " sort Time\t" << RED << Qsort << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " Query Time\t" << RED << Qque << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " Communication\t" << BLUE << QCom << " MB" << RESET << endl;
  ofstream outfile;
  string pp = "../../tests/out_" + to_string(party) + ".txt";
  if (Pth == 1)
  {
    pp = "../../tests/out_" + to_string(party) + "_T.txt";
  }
  outfile.open(pp, ios::app | ios::in);
  outfile << " -------------------------------------" << endl;
  outfile << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) + " " + data_path << endl;
  outfile << "CS_" + to_string(party) + " Skyline Number:" << Qlen << endl;
  outfile << "CS_" + to_string(party) + " Total Time:" << Qtime << " s" << endl;
  outfile << "CS_" + to_string(party) + " process Time:" << Qpre << " s" << endl;
  outfile << "CS_" + to_string(party) + " sort Time:" << Qsort << " s" << endl;
  outfile << "CS_" + to_string(party) + " Query Time:" << Qque << " s" << endl;
  outfile << "CS_" + to_string(party) + " Communication:" << QCom << " MB" << endl;
  outfile << " -------------------------------------" << endl;
  outfile.close();
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

int mainF_O(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("Pdim", Pd, "data dim");
  amap.arg("Psize", Pst, "data size");
  amap.arg("Prate", Pr, "data missing rate");
  amap.arg("Pname", Pn, "data name");
  amap.arg("Pthread", Pth, "omp multi-thread");
  amap.parse(argc, argv);
  if (Pth == 0)
  {
    THs = 2;
  }
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0) + party);
  // shar();
  cout << "Party: CS_" << party << endl;

  vector<int> input_dim = {2, 3, 4, 5, 6};                                // size = 4
  vector<int> input_size = {1000, 3000, 5000, 7000, 9000, 11000, 100000}; // size = 6
  vector<int> rate = {10, 20, 30, 40, 50};                                // size = 4
  vector<string> path = {"small_anti_corr.txt", "small_correlate.txt", "small_uniformly.txt", "nba.txt", "hotel.txt"};
  // Pd = 0, Pst = 0, Pr = 0, Pn = 4;

  m = input_dim[Pd];
  n = input_size[Pst];
  int r = rate[Pr];
  filename = path[Pn];
  data_path = "./data/data" + to_string(m) + "/size=" + to_string(n) + "/rate=" + to_string(r) + "/" + filename;
  // m = 3;
  // n = 9;
  // data_path = "./10.txt";
  cout << data_path << endl;
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<vector<uint32_t>> zeroM(n, vector<uint32_t>(m));
  loadP(p, zeroM, data_path);
  double Qtime = 0, QCom = 0, Qpre = 0, Qque = 0, Qsort = 0, Qlen = 0;
  int itrs = 3;
  for (int itr = 0; itr < itrs; itr++)
  {
    vector<uint32_t> q(m);
    for (int j = 0; j < m; j++)
    {
      q[j] = rand() % 2;
      // q[j] = 1;
    }
    if (party == ALICE)
    {
      uint32_t *q0 = new uint32_t[m];
      for (int j = 0; j < m; j++)
      {
        q0[j] = q[j];
      }
      Iot[0]->send_data(q0, m * sizeof(uint32_t));
      delete[] q0;
    }
    else
    {
      uint32_t *q0 = new uint32_t[m];
      Iot[0]->recv_data(q0, m * sizeof(uint32_t));
      for (int j = 0; j < m; j++)
      {
        q[j] = q0[j];
      }
      delete[] q0;
    }
    cout << itr << "-itr:\t" << endl;
    cout << "q: ";
    for (int j = 0; j < m; j++)
    {
      cout << q[j] << "\t";
    }
    cout << endl;
    // plainB(p, zeroM, q);
    plainF(p, zeroM, q);
    cout << "+++" << endl;
    // BSS(p, zeroM, q);
    // BSS_minPre(p, zeroM, q);
    // BSS_T(p, zeroM, q);
    uint64_t comm_start = 0;
    for (int j = 0; j < THs; j++)
    {
      comm_start += Iot[j]->counter;
    }
    double start = omp_get_wtime();
    int len;
    if (Pth == 0)
    {
      len = FSS(p, zeroM, q);
    }
    else
    {
      len = FSS_T(p, zeroM, q);
    }
    // int len = BSS_min(p, zeroM, q);
    // int len = BSS_min_T(p, zeroM, q);
    // int len = BSS_min_bucket(p, zeroM, q);
    // int len = FSS(p, zeroM, q);
    // int len = FSS_T(p, zeroM, q);
    double end = omp_get_wtime();
    Qtime += end - start;
    uint64_t comm_end = 0;
    for (int j = 0; j < THs; j++)
    {
      comm_end += Iot[j]->counter;
    }
    QCom += comm_end - comm_start;
    Qpre += pre_time;
    Qsort += sort_time;
    Qque += que_time;
    Qlen += len;
    delete[] SS_p;
    delete[] SS_zeroM;
    delete[] Skyline;
    delete[] SS_q;
    SS_pB.clear();
  }
  Qtime = Qtime / itrs;
  QCom = QCom / itrs / 1024 / 1024;
  Qpre = Qpre / itrs;
  Qsort = Qsort / itrs;
  Qque = Qque / itrs;
  Qlen = Qlen / itrs;
  cout << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) << endl;
  cout << "CS_" + to_string(party) + " Skyline Number\t" << RED << Qlen << RESET << endl;
  cout << "CS_" + to_string(party) + " Total Time\t" << RED << Qtime << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " process Time\t" << RED << Qpre << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " sort Time\t" << RED << Qsort << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " Query Time\t" << RED << Qque << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " Communication\t" << BLUE << QCom << " MB" << RESET << endl;
  ofstream outfile;
  string pp = "../../tests/out_" + to_string(party) + "_F.txt";
  if (Pth == 1)
  {
    pp = "../../tests/out_" + to_string(party) + "_FT.txt";
  }
  outfile.open(pp, ios::app | ios::in);
  outfile << " -------------------------------------" << endl;
  outfile << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) + " " + data_path << endl;
  outfile << "CS_" + to_string(party) + " Skyline Number:" << Qlen << endl;
  outfile << "CS_" + to_string(party) + " Total Time:" << Qtime << " s" << endl;
  outfile << "CS_" + to_string(party) + " process Time:" << Qpre << " s" << endl;
  outfile << "CS_" + to_string(party) + " sort Time:" << Qsort << " s" << endl;
  outfile << "CS_" + to_string(party) + " Query Time:" << Qque << " s" << endl;
  outfile << "CS_" + to_string(party) + " Communication:" << QCom << " MB" << endl;
  outfile << " -------------------------------------" << endl;
  outfile.close();
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

// skyline num
int mainNum(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("Pdim", Pd, "data dim");
  amap.arg("Psize", Pst, "data size");
  amap.arg("Prate", Pr, "data missing rate");
  amap.arg("Pname", Pn, "data name");
  amap.arg("Pthread", Pth, "omp multi-thread");
  amap.parse(argc, argv);
  THs = 2;
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0) + party);
  // shar();
  cout << "Party: CS_" << party << endl;

  vector<int> input_dim = {2, 3, 4, 5, 6};                                // size = 4
  vector<int> input_size = {1000, 3000, 5000, 7000, 9000, 11000, 100000}; // size = 6
  vector<int> rate = {10, 20, 30, 40, 50};                                // size = 4
  vector<string> path = {"small_anti_corr.txt", "small_correlate.txt", "small_uniformly.txt", "nba.txt", "hotel.txt"};
  // Pd = 0, Pst = 0, Pr = 0, Pn = 4;

  m = input_dim[Pd];
  n = input_size[Pst];
  int r = rate[Pr];
  filename = path[Pn];
  data_path = "./data/data" + to_string(m) + "/size=" + to_string(n) + "/rate=" + to_string(r) + "/" + filename;
  // m = 3;
  // n = 9;
  // data_path = "./10.txt";
  cout << data_path << endl;
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<vector<uint32_t>> zeroM(n, vector<uint32_t>(m));
  loadP(p, zeroM, data_path);
  double Qnum = 0, Qlen = 0;
  int itrs = 10;
  for (int itr = 0; itr < itrs; itr++)
  {
    vector<uint32_t> q(m);
    for (int j = 0; j < m; j++)
    {
      q[j] = rand() % 2;
      // q[j] = 1;
    }
    if (party == ALICE)
    {
      uint32_t *q0 = new uint32_t[m];
      for (int j = 0; j < m; j++)
      {
        q0[j] = q[j];
      }
      Iot[0]->send_data(q0, m * sizeof(uint32_t));
      delete[] q0;
    }
    else
    {
      uint32_t *q0 = new uint32_t[m];
      Iot[0]->recv_data(q0, m * sizeof(uint32_t));
      for (int j = 0; j < m; j++)
      {
        q[j] = q0[j];
      }
      delete[] q0;
    }
    cout << itr << "-itr:\t" << endl;
    cout << "q: ";
    for (int j = 0; j < m; j++)
    {
      cout << q[j] << "\t";
    }
    cout << endl;
    int len;
    if (Pth == 0)
    {
      len = plainB_min(p, zeroM, q);
    }
    else
    {
      len = plainF(p, zeroM, q);
    }
    Qnum += countDom;
    Qlen += len;
    delete[] Skyline;
    SS_pB.clear();
  }
  Qnum = Qnum / itrs;
  Qlen = Qlen / itrs;
  cout << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) << endl;
  cout << "CS_" + to_string(party) + " Skyline Number\t" << RED << Qlen << RESET << endl;
  cout << "CS_" + to_string(party) + " Dom Number\t" << RED << Qnum << RESET << endl;
  ofstream outfile;
  string pp = "../../tests/out_" + to_string(party) + "_Num.txt";
  if (Pth == 1)
  {
    pp = "../../tests/out_" + to_string(party) + "_Num_F.txt";
  }
  outfile.open(pp, ios::app | ios::in);
  outfile << " -------------------------------------" << endl;
  outfile << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) + " " + data_path << endl;
  outfile << "CS_" + to_string(party) + " Skyline Number:" << Qlen << endl;
  outfile << "CS_" + to_string(party) + " Dom Number:" << Qnum << endl;
  outfile << " -------------------------------------" << endl;
  outfile.close();
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

int mainRNum(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("Pdim", Pd, "data dim");
  amap.arg("Psize", Pst, "data size");
  amap.arg("Prate", Pr, "data missing rate");
  amap.arg("Pname", Pn, "data name");
  amap.arg("Pthread", Pth, "omp multi-thread");
  amap.parse(argc, argv);
  THs = 2;
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0) + party);
  // shar();
  cout << "Party: CS_" << party << endl;

  vector<int> input_dim = {2, 3, 4, 5, 6};                                // size = 4
  vector<int> input_size = {1000, 3000, 5000, 7000, 9000, 11000, 100000}; // size = 6
  vector<int> rate = {10, 20, 30, 40, 50};                                // size = 4
  vector<string> path = {"small_anti_corr.txt", "small_correlate.txt", "small_uniformly.txt", "nba.txt", "hotel.txt"};
  Pd = 1, Pst = 0, Pr = 1, Pn = 1;

  m = input_dim[Pd];
  n = input_size[Pst];
  int r = rate[Pr];
  filename = path[Pn];
  data_path = "./data/data" + to_string(m) + "/size=" + to_string(n) + "/rate=" + to_string(r) + "/" + filename;
  // m = 3;
  // n = 9;
  // data_path = "./10.txt";
  cout << data_path << endl;
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<vector<uint32_t>> zeroM(n, vector<uint32_t>(m));
  loadP(p, zeroM, data_path);
  uint64_t plain_mu = 1, plain_theta = 1;
  uint64_t *mu = new uint64_t[1];
  uint64_t *theta = new uint64_t[1];
  prg.random_data(mu, sizeof(uint64_t));
  prg.random_data(theta, sizeof(uint64_t));
  if (party == ALICE)
  {
    Iot[0]->send_data(mu, sizeof(uint64_t));
    Iot[1]->recv_data(theta, sizeof(uint64_t));
    theta[0] = (plain_theta - theta[0]) & mask;
  }
  else
  {
    Iot[0]->recv_data(mu, sizeof(uint64_t));
    Iot[1]->send_data(theta, sizeof(uint64_t));
    mu[0] = (plain_mu - mu[0]) & mask;
  }
  double len_0 = 0, num_0 = 0, len_1 = 0, num_1 = 0, len_2 = 0, num_2 = 0, len_3 = 0, num_3 = 0;
  int itrs = 1 << m;
  cout << itrs << endl;
  Pth = 1;
  for (int itr = 0; itr < itrs; itr++)
  {
    vector<uint32_t> q(m);
    for (int j = 0; j < m; j++)
    {
      q[j] = (itr >> j) & 1;
    }
    if (party == ALICE)
    {
      uint32_t *q0 = new uint32_t[m];
      for (int j = 0; j < m; j++)
      {
        q0[j] = q[j];
      }
      Iot[0]->send_data(q0, m * sizeof(uint32_t));
      delete[] q0;
    }
    else
    {
      uint32_t *q0 = new uint32_t[m];
      Iot[0]->recv_data(q0, m * sizeof(uint32_t));
      for (int j = 0; j < m; j++)
      {
        q[j] = q0[j];
      }
      delete[] q0;
    }
    cout << itr << "-itr:\t" << endl;
    cout << "q: ";
    for (int j = 0; j < m; j++)
    {
      cout << q[j] << "\t";
    }
    cout << endl;
    // plainB(p, zeroM, q);
    int len0, len1, len2, len3;
    if (Pth == 0)
    {
      cout << "CS_" + to_string(party) + " Dim:" << plain_mu << endl;
      cout << "BSS_" + to_string(party) + " Dim" << endl;
      len0 = plainB_Dim(p, zeroM, q, plain_mu);
      cout << len0 << "+++" << countDom << endl;
      len_0 += len0;
      num_0 += countDom;
      cout << "FSS_" + to_string(party) + " Dim" << endl;
      len1 = plainF_Dim(p, zeroM, q, plain_mu);
      cout << len1 << "+++" << countDom << endl;
      len_1 += len1;
      num_1 += countDom;
    }
    else
    {
      cout << "CS_" + to_string(party) + " Dom:" << plain_theta << endl;
      cout << "BSS_" + to_string(party) + " Dom" << endl;
      len0 = plainB_Dom(p, zeroM, q, plain_theta);
      cout << len0 << "+++" << countDom << endl;
      len_0 += len0;
      num_0 += countDom;
      cout << "FSS_" + to_string(party) + " Dom" << endl;
      len1 = plainF_Dom(p, zeroM, q, plain_theta);
      cout << len1 << "+++" << countDom << endl;
      len_1 += len1;
      num_1 += countDom;
    }
    len2 = plainB_min(p, zeroM, q);
    cout << "BSS:"<< len2 << "+++" << countDom << endl;
    len_2 += len2;
    num_2 += countDom;
    len3 = plainF(p, zeroM, q);
    cout << "FSS:"<< len3 << "+++" << countDom << endl;
    cout << "+++" << endl;
    len_3 += len3;
    num_3 += countDom;
  }
  cout << "CS_" + to_string(party) + " BSS_ Skyline Number\t" << RED << len_0/itrs << RESET << endl;
  cout << "CS_" + to_string(party) + " BSS_ Dom Number\t" << RED << num_0/itrs << RESET << endl;
  cout << "CS_" + to_string(party) + " FSS_ Skyline Number\t" << RED << len_1/itrs << RESET << endl;
  cout << "CS_" + to_string(party) + " FSS_ Dom Number\t" << RED << num_1/itrs << RESET << endl;
  cout << "CS_" + to_string(party) + " BSS: Skyline Number\t" << RED << len_2/itrs << RESET << endl;
  cout << "CS_" + to_string(party) + " BSS: Dom Number\t" << RED << num_2/itrs << RESET << endl;
  cout << "CS_" + to_string(party) + " FSS: Skyline Number\t" << RED << len_3/itrs << RESET << endl;
  cout << "CS_" + to_string(party) + " FSS: Dom Number\t" << RED << num_3/itrs << RESET << endl;
  return 0;
}

int mainRF(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("Pdim", Pd, "data dim");
  amap.arg("Psize", Pst, "data size");
  amap.arg("Prate", Pr, "data missing rate");
  amap.arg("Pname", Pn, "data name");
  amap.arg("Pthread", Pth, "omp multi-thread");
  amap.arg("Ptheta", Pta, "relax theta");
  amap.arg("Pmu", Pm, "relax mu");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0) + party);
  // shar();
  cout << "Party: CS_" << party << endl;

  vector<int> input_dim = {2, 3, 4, 5, 6};                                // size = 4
  vector<int> input_size = {1000, 3000, 5000, 7000, 9000, 11000, 100000}; // size = 6
  vector<int> rate = {10, 20, 30, 40, 50};                                // size = 4
  vector<string> path = {"small_anti_corr.txt", "small_correlate.txt", "small_uniformly.txt", "nba.txt", "hotel.txt"};
  // Pd = 0, Pst = 0, Pr = 0, Pn = 4;

  m = input_dim[Pd];
  n = input_size[Pst];
  int r = rate[Pr];
  filename = path[Pn];
  data_path = "./data/data" + to_string(m) + "/size=" + to_string(n) + "/rate=" + to_string(r) + "/" + filename;
  // m = 3;
  // n = 100;
  // data_path = "./data/test.txt";
  cout << data_path << endl;
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<vector<uint32_t>> zeroM(n, vector<uint32_t>(m));
  loadP(p, zeroM, data_path);
  uint64_t plain_mu = Pm, plain_theta = Pta;
  uint64_t *mu = new uint64_t[1];
  uint64_t *theta = new uint64_t[1];
  prg.random_data(mu, sizeof(uint64_t));
  prg.random_data(theta, sizeof(uint64_t));
  if (party == ALICE)
  {
    Iot[0]->send_data(mu, sizeof(uint64_t));
    Iot[1]->recv_data(theta, sizeof(uint64_t));
    theta[0] = (plain_theta - theta[0]) & mask;
  }
  else
  {
    Iot[0]->recv_data(mu, sizeof(uint64_t));
    Iot[1]->send_data(theta, sizeof(uint64_t));
    mu[0] = (plain_mu - mu[0]) & mask;
  }
  double o_len = 0, o_num = 0, p_len = 0, p_num = 0; 
  double Qtime = 0, QCom = 0, Qpre = 0, Qque = 0, Qsort = 0, Qlen = 0, Qnum = 0;
  int itrs = 1 << m;
  cout << itrs << endl;
  for (int itr = 0; itr < itrs; itr++)
  {
    vector<uint32_t> q(m);
    for (int j = 0; j < m; j++)
    {
      q[j] = (itr >> j) & 1;
    }
    if (party == ALICE)
    {
      uint32_t *q0 = new uint32_t[m];
      for (int j = 0; j < m; j++)
      {
        q0[j] = q[j];
      }
      Iot[0]->send_data(q0, m * sizeof(uint32_t));
      delete[] q0;
    }
    else
    {
      uint32_t *q0 = new uint32_t[m];
      Iot[0]->recv_data(q0, m * sizeof(uint32_t));
      for (int j = 0; j < m; j++)
      {
        q[j] = q0[j];
      }
      delete[] q0;
    }
    cout << itr << "-itr:\t" << endl;
    cout << "q: ";
    for (int j = 0; j < m; j++)
    {
      cout << q[j] << "\t";
    }
    cout << endl;
    int len0 = plainF(p, zeroM, q);
    o_len += len0;
    o_num += countDom;
    cout << "origin:" << len0 << "+++" << countDom << endl;
    if (Pth == 0)
    {
      cout << "CS_" + to_string(party) + " Dim:" << plain_mu << endl;
      len0 = plainF_Dim(p, zeroM, q, plain_mu);
      cout << "plain:" << len0 << "+++" << countDom << endl;
    }
    else
    {
      cout << "CS_" + to_string(party) + " Dom:" << plain_theta << endl;
      len0 = plainF_Dom(p, zeroM, q, plain_theta);
      cout << "plain:" << len0 << "+++" << countDom << endl;
    }
    p_len += len0;
    p_num += countDom;
    uint64_t comm_start = 0;
    for (int j = 0; j < THs; j++)
    {
      comm_start += Iot[j]->counter;
    }
    double start = omp_get_wtime();
    int len;
    if (Pth == 0)
    {
      len = FSS_T_Dim(p, zeroM, q, mu);
    }
    else
    {
      len = FSS_T_Dom(p, zeroM, q, theta);
    }
    double end = omp_get_wtime();
    Qtime += end - start;
    uint64_t comm_end = 0;
    for (int j = 0; j < THs; j++)
    {
      comm_end += Iot[j]->counter;
    }
    QCom += comm_end - comm_start;
    Qpre += pre_time;
    Qsort += sort_time;
    Qque += que_time;
    Qlen += len;
    Qnum += countDom;
    delete[] SS_p;
    delete[] SS_zeroM;
    delete[] Skyline;
    delete[] SS_q;
    SS_pB.clear();
    cout << "CS_" + to_string(party) + " Skyline Number\t" << RED << len << RESET << endl;
    cout << "CS_" + to_string(party) + " Dom Number\t" << RED << countDom << RESET << endl;
    cout << "CS_" + to_string(party) + " Total Time\t" << RED << (end - start) << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " process Time\t" << RED << pre_time << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " sort Time\t" << RED << sort_time << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " Query Time\t" << RED << que_time << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " Communication\t" << BLUE <<  (comm_end - comm_start) / 1024 / 1024 << " MB" << RESET << endl;   
  }
  Qtime = Qtime / itrs;
  QCom = QCom / itrs / 1024 / 1024;
  Qpre = Qpre / itrs;
  Qsort = Qsort / itrs;
  Qque = Qque / itrs;
  Qlen = Qlen / itrs;
  Qnum = Qnum / itrs;
  o_len = o_len / itrs;
  o_num = o_num / itrs;
  p_len = p_len / itrs;
  p_num = p_num / itrs;
  cout << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) << endl;
  if (Pth == 0)
  {
    cout << "Dim:" << plain_mu << endl;
  } else {
    cout << "Dom:" << plain_theta << endl;
  }
  cout << "CS_" + to_string(party) + " Origin Skyline Number\t" << RED << o_len << RESET << endl;
  cout << "CS_" + to_string(party) + " Origin Dom Number\t" << RED << o_num << RESET << endl;
  cout << "CS_" + to_string(party) + " Plain Skyline Number\t" << RED << p_len << RESET << endl;
  cout << "CS_" + to_string(party) + " Plain Dom Number\t" << RED << p_num << RESET << endl;
  cout << "CS_" + to_string(party) + " Relax Skyline Number\t" << RED << Qlen << RESET << endl;
  cout << "CS_" + to_string(party) + " Relax Dom Number\t" << RED << Qnum << RESET << endl;
  cout << "CS_" + to_string(party) + " Total Time\t" << RED << Qtime << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " process Time\t" << RED << Qpre << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " sort Time\t" << RED << Qsort << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " Query Time\t" << RED << Qque << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " Communication\t" << BLUE << QCom << " MB" << RESET << endl;
  ofstream outfile;
  string pp = "../../tests/out_" + to_string(party) + "_Re.txt";
  outfile.open(pp, ios::app | ios::in);
  outfile << " -------------------------------------" << endl;
  outfile << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) + " " + data_path;
  if (Pth == 0)
  {
    outfile << "-Dim:" << plain_mu << endl;
  } else {
    outfile << "-Dom:" << plain_theta << endl;
  }
  outfile << "CS_" + to_string(party) + " Origin Skyline Number\t" << o_len << endl;
  outfile << "CS_" + to_string(party) + " Origin Dom Number\t" << o_num << endl;
  outfile << "CS_" + to_string(party) + " Plain Skyline Number\t" << p_len << endl;
  outfile << "CS_" + to_string(party) + " Plain Dom Number\t" << p_num << endl;
  outfile << "CS_" + to_string(party) + " Relax Skyline Number:" << Qlen << endl;
  outfile << "CS_" + to_string(party) + " Relax Dom Number:" << Qnum << endl;
  outfile << "CS_" + to_string(party) + " Total Time:" << Qtime << " s" << endl;
  outfile << "CS_" + to_string(party) + " process Time:" << Qpre << " s" << endl;
  outfile << "CS_" + to_string(party) + " sort Time:" << Qsort << " s" << endl;
  outfile << "CS_" + to_string(party) + " Query Time:" << Qque << " s" << endl;
  outfile << "CS_" + to_string(party) + " Communication:" << QCom << " MB" << endl;
  outfile << " -------------------------------------" << endl;
  outfile.close();
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

int mainRB(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("Pdim", Pd, "data dim");
  amap.arg("Psize", Pst, "data size");
  amap.arg("Prate", Pr, "data missing rate");
  amap.arg("Pname", Pn, "data name");
  amap.arg("Pthread", Pth, "omp multi-thread");
  amap.arg("Ptheta", Pta, "relax theta");
  amap.arg("Pmu", Pm, "relax mu");
  amap.parse(argc, argv);
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0) + party);
  // shar();
  cout << "Party: CS_" << party << endl;

  vector<int> input_dim = {2, 3, 4, 5, 6};                                // size = 4
  vector<int> input_size = {1000, 3000, 5000, 7000, 9000, 11000, 100000}; // size = 6
  vector<int> rate = {10, 20, 30, 40, 50};                                // size = 4
  vector<string> path = {"small_anti_corr.txt", "small_correlate.txt", "small_uniformly.txt", "nba.txt", "hotel.txt"};
  // Pd = 0, Pst = 0, Pr = 0, Pn = 4;

  m = input_dim[Pd];
  n = input_size[Pst];
  int r = rate[Pr];
  filename = path[Pn];
  data_path = "./data/data" + to_string(m) + "/size=" + to_string(n) + "/rate=" + to_string(r) + "/" + filename;
  // m = 3;
  // n = 9;
  // data_path = "./10.txt";
  cout << data_path << endl;
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<vector<uint32_t>> zeroM(n, vector<uint32_t>(m));
  loadP(p, zeroM, data_path);
  uint64_t plain_mu = Pm, plain_theta = Pta;
  uint64_t *mu = new uint64_t[1];
  uint64_t *theta = new uint64_t[1];
  prg.random_data(mu, sizeof(uint64_t));
  prg.random_data(theta, sizeof(uint64_t));
  if (party == ALICE)
  {
    Iot[0]->send_data(mu, sizeof(uint64_t));
    Iot[1]->recv_data(theta, sizeof(uint64_t));
    theta[0] = (plain_theta - theta[0]) & mask;
  }
  else
  {
    Iot[0]->recv_data(mu, sizeof(uint64_t));
    Iot[1]->send_data(theta, sizeof(uint64_t));
    mu[0] = (plain_mu - mu[0]) & mask;
  }
  double o_len = 0, o_num = 0, p_len = 0, p_num = 0; 
  double Qtime = 0, QCom = 0, Qpre = 0, Qque = 0, Qsort = 0, Qlen = 0, Qnum = 0;
  int itrs = 1 << m;
  cout << itrs << endl;
  for (int itr = 0; itr < itrs; itr++)
  {
    vector<uint32_t> q(m);
    for (int j = 0; j < m; j++)
    {
      q[j] = (itr >> j) & 1;
    }
    if (party == ALICE)
    {
      uint32_t *q0 = new uint32_t[m];
      for (int j = 0; j < m; j++)
      {
        q0[j] = q[j];
      }
      Iot[0]->send_data(q0, m * sizeof(uint32_t));
      delete[] q0;
    }
    else
    {
      uint32_t *q0 = new uint32_t[m];
      Iot[0]->recv_data(q0, m * sizeof(uint32_t));
      for (int j = 0; j < m; j++)
      {
        q[j] = q0[j];
      }
      delete[] q0;
    }
    cout << itr << "-itr:\t" << endl;
    cout << "q: ";
    for (int j = 0; j < m; j++)
    {
      cout << q[j] << "\t";
    }
    cout << endl;
    int len0 = plainB_min(p, zeroM, q);
    o_len += len0;
    o_num += countDom;
    cout << "origin:" << len0 << "+++" << countDom << endl;
    if (Pth == 0)
    {
      cout << "CS_" + to_string(party) + " Dim:" << plain_mu << endl;
      len0 = plainB_Dim(p, zeroM, q, plain_mu);
      cout << "plain:" << len0 << "+++" << countDom << endl;
    }
    else
    {
      cout << "CS_" + to_string(party) + " Dom:" << plain_theta << endl;
      len0 = plainB_Dom(p, zeroM, q, plain_theta);
      cout << "plain:" << len0 << "+++" << countDom << endl;
    }
    p_len += len0;
    p_num += countDom;
    uint64_t comm_start = 0;
    for (int j = 0; j < THs; j++)
    {
      comm_start += Iot[j]->counter;
    }
    double start = omp_get_wtime();
    int len;
    if (Pth == 0)
    {
      len = BSS_T_Dim(p, zeroM, q, mu);
    }
    else
    {
      len = BSS_T_Dom(p, zeroM, q, plain_theta);
    }
    double end = omp_get_wtime();
    Qtime += end - start;
    uint64_t comm_end = 0;
    for (int j = 0; j < THs; j++)
    {
      comm_end += Iot[j]->counter;
    }
    QCom += comm_end - comm_start;
    Qpre += pre_time;
    Qsort += sort_time;
    Qque += que_time;
    Qlen += len;
    Qnum += countDom;
    delete[] SS_p;
    delete[] SS_zeroM;
    delete[] Skyline;
    delete[] SS_q;
    SS_pB.clear();
    cout << "CS_" + to_string(party) + " Skyline Number\t" << RED << len << RESET << endl;
    cout << "CS_" + to_string(party) + " Dom Number\t" << RED << countDom << RESET << endl;
    cout << "CS_" + to_string(party) + " Total Time\t" << RED << (end - start) << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " process Time\t" << RED << pre_time << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " sort Time\t" << RED << sort_time << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " Query Time\t" << RED << que_time << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " Communication\t" << BLUE <<  (comm_end - comm_start) / 1024 / 1024 << " MB" << RESET << endl;   
  }
  Qtime = Qtime / itrs;
  QCom = QCom / itrs / 1024 / 1024;
  Qpre = Qpre / itrs;
  Qsort = Qsort / itrs;
  Qque = Qque / itrs;
  Qlen = Qlen / itrs;
  Qnum = Qnum / itrs;
  o_len = o_len / itrs;
  o_num = o_num / itrs;
  p_len = p_len / itrs;
  p_num = p_num / itrs;
  cout << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) << endl;
  if (Pth == 0)
  {
    cout << "Dim:" << plain_mu << endl;
  } else {
    cout << "Dom:" << plain_theta << endl;
  }
  cout << "CS_" + to_string(party) + " Origin Skyline Number\t" << RED << o_len << RESET << endl;
  cout << "CS_" + to_string(party) + " Origin Dom Number\t" << RED << o_num << RESET << endl;
  cout << "CS_" + to_string(party) + " Plain Skyline Number\t" << RED << p_len << RESET << endl;
  cout << "CS_" + to_string(party) + " Plain Dom Number\t" << RED << p_num << RESET << endl;
  cout << "CS_" + to_string(party) + " Relax Skyline Number\t" << RED << Qlen << RESET << endl;
  cout << "CS_" + to_string(party) + " Relax Dom Number\t" << RED << Qnum << RESET << endl;
  cout << "CS_" + to_string(party) + " Total Time\t" << RED << Qtime << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " process Time\t" << RED << Qpre << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " sort Time\t" << RED << Qsort << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " Query Time\t" << RED << Qque << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " Communication\t" << BLUE << QCom << " MB" << RESET << endl;
  ofstream outfile;
  string pp = "../../tests/out_" + to_string(party) + "_Re_B.txt";
  outfile.open(pp, ios::app | ios::in);
  outfile << " -------------------------------------" << endl;
  outfile << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) + " " + data_path;
  if (Pth == 0)
  {
    outfile << "-Dim:" << plain_mu << endl;
  } else {
    outfile << "-Dom:" << plain_theta << endl;
  }
  outfile << "CS_" + to_string(party) + " Origin Skyline Number\t" << o_len << endl;
  outfile << "CS_" + to_string(party) + " Origin Dom Number\t" << o_num << endl;
  outfile << "CS_" + to_string(party) + " Plain Skyline Number\t" << p_len << endl;
  outfile << "CS_" + to_string(party) + " Plain Dom Number\t" << p_num << endl;
  outfile << "CS_" + to_string(party) + " Relax Skyline Number:" << Qlen << endl;
  outfile << "CS_" + to_string(party) + " Relax Dom Number:" << Qnum << endl;
  outfile << "CS_" + to_string(party) + " Total Time:" << Qtime << " s" << endl;
  outfile << "CS_" + to_string(party) + " process Time:" << Qpre << " s" << endl;
  outfile << "CS_" + to_string(party) + " sort Time:" << Qsort << " s" << endl;
  outfile << "CS_" + to_string(party) + " Query Time:" << Qque << " s" << endl;
  outfile << "CS_" + to_string(party) + " Communication:" << QCom << " MB" << endl;
  outfile << " -------------------------------------" << endl;
  outfile.close();
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

int mainF(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("Pdim", Pd, "data dim");
  amap.arg("Psize", Pst, "data size");
  amap.arg("Prate", Pr, "data missing rate");
  amap.arg("Pname", Pn, "data name");
  amap.arg("Pthread", Pth, "omp multi-thread");
  amap.parse(argc, argv);
  if (Pth == 0)
  {
    THs = 2;
  }
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0) + party);
  // shar();
  cout << "Party: CS_" << party << endl;

  vector<int> input_dim = {2, 3, 4, 5, 6};                                // size = 4
  vector<int> input_size = {1000, 3000, 5000, 7000, 9000, 11000, 100000}; // size = 6
  vector<int> rate = {10, 20, 30, 40, 50};                                // size = 4
  vector<string> path = {"small_anti_corr.txt", "small_correlate.txt", "small_uniformly.txt", "nba.txt", "hotel.txt"};
  // Pd = 0, Pst = 0, Pr = 0, Pn = 4;

  m = input_dim[Pd];
  n = input_size[Pst];
  int r = rate[Pr];
  filename = path[Pn];
  data_path = "./data/data" + to_string(m) + "/size=" + to_string(n) + "/rate=" + to_string(r) + "/" + filename;
  // m = 3;
  // n = 100;
  // data_path = "./data/test.txt";
  cout << data_path << endl;
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<vector<uint32_t>> zeroM(n, vector<uint32_t>(m));
  loadP(p, zeroM, data_path);
  double Qtime = 0, QCom = 0, Qpre = 0, Qque = 0, Qsort = 0, Qlen = 0, Qnum = 0;
  int itrs = 1 << m;
  cout << itrs << endl;
  for (int itr = 0; itr < itrs; itr++)
  {
    vector<uint32_t> q(m);
    for (int j = 0; j < m; j++)
    {
      q[j] = (itr >> j) & 1;
    }
    if (party == ALICE)
    {
      uint32_t *q0 = new uint32_t[m];
      for (int j = 0; j < m; j++)
      {
        q0[j] = q[j];
      }
      Iot[0]->send_data(q0, m * sizeof(uint32_t));
      delete[] q0;
    }
    else
    {
      uint32_t *q0 = new uint32_t[m];
      Iot[0]->recv_data(q0, m * sizeof(uint32_t));
      for (int j = 0; j < m; j++)
      {
        q[j] = q0[j];
      }
      delete[] q0;
    }
    cout << itr << "-itr:\t" << endl;
    cout << "q: ";
    for (int j = 0; j < m; j++)
    {
      cout << q[j] << "\t";
    }
    cout << endl;
    int len;
    len = plainF(p, zeroM, q);
    cout << "CS_" + to_string(party) + " Skyline Number\t" << RED << len << RESET << endl;
    cout << "CS_" + to_string(party) + " Dom Number\t" << RED << countDom << RESET << endl;
    cout << "+++" << endl;
    // BSS(p, zeroM, q);
    // BSS_minPre(p, zeroM, q);
    // BSS_T(p, zeroM, q);
    uint64_t comm_start = 0;
    for (int j = 0; j < THs; j++)
    {
      comm_start += Iot[j]->counter;
    }
    double start = omp_get_wtime();
    if (Pth == 0)
    {
      len = FSS(p, zeroM, q);
    }
    else
    {
      len = FSS_T(p, zeroM, q);
    }
    // int len = BSS_min(p, zeroM, q);
    // int len = BSS_min_T(p, zeroM, q);
    // int len = BSS_min_bucket(p, zeroM, q);
    // int len = FSS(p, zeroM, q);
    // int len = FSS_T(p, zeroM, q);
    double end = omp_get_wtime();
    Qtime += end - start;
    uint64_t comm_end = 0;
    for (int j = 0; j < THs; j++)
    {
      comm_end += Iot[j]->counter;
    }
    QCom += comm_end - comm_start;
    Qpre += pre_time;
    Qsort += sort_time;
    Qque += que_time;
    Qlen += len;
    Qnum += countDom;
    delete[] SS_p;
    delete[] SS_zeroM;
    delete[] Skyline;
    delete[] SS_q;
    SS_pB.clear();
    cout << "CS_" + to_string(party) + " Skyline Number\t" << RED << len << RESET << endl;
    cout << "CS_" + to_string(party) + " Total Time\t" << RED << (end - start) << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " process Time\t" << RED << pre_time << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " sort Time\t" << RED << sort_time << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " Query Time\t" << RED << que_time << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " Communication\t" << BLUE <<  (comm_end - comm_start) / 1024 / 1024 << " MB" << RESET << endl;
  }
  Qtime = Qtime / itrs;
  QCom = QCom / itrs / 1024 / 1024;
  Qpre = Qpre / itrs;
  Qsort = Qsort / itrs;
  Qque = Qque / itrs;
  Qlen = Qlen / itrs;
  Qnum = Qnum / itrs;
  cout << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) << endl;
  cout << "CS_" + to_string(party) + " Skyline Number\t" << RED << Qlen << RESET << endl;
  cout << "CS_" + to_string(party) + " Dom Number\t" << RED << Qnum << RESET << endl;
  cout << "CS_" + to_string(party) + " Total Time\t" << RED << Qtime << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " process Time\t" << RED << Qpre << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " sort Time\t" << RED << Qsort << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " Query Time\t" << RED << Qque << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " Communication\t" << BLUE << QCom << " MB" << RESET << endl;
  ofstream outfile;
  string pp = "../../tests/out_" + to_string(party) + "_F.txt";
  if (Pth == 1)
  {
    pp = "../../tests/out_" + to_string(party) + "_FT.txt";
  }
  outfile.open(pp, ios::app | ios::in);
  outfile << " -------------------------------------" << endl;
  outfile << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) + " " + data_path << endl;
  outfile << "CS_" + to_string(party) + " Skyline Number:" << Qlen << endl;
  outfile << "CS_" + to_string(party) + " Dom Number:" << Qnum << endl;
  outfile << "CS_" + to_string(party) + " Total Time:" << Qtime << " s" << endl;
  outfile << "CS_" + to_string(party) + " process Time:" << Qpre << " s" << endl;
  outfile << "CS_" + to_string(party) + " sort Time:" << Qsort << " s" << endl;
  outfile << "CS_" + to_string(party) + " Query Time:" << Qque << " s" << endl;
  outfile << "CS_" + to_string(party) + " Communication:" << QCom << " MB" << endl;
  outfile << " -------------------------------------" << endl;
  outfile.close();
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}

int main(int argc, char **argv)
{
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("Pdim", Pd, "data dim");
  amap.arg("Psize", Pst, "data size");
  amap.arg("Prate", Pr, "data missing rate");
  amap.arg("Pname", Pn, "data name");
  amap.arg("Pthread", Pth, "omp multi-thread");
  amap.parse(argc, argv);
  if (Pth == 0)
  {
    THs = 2;
  }
  for (int i = 0; i < THs; i++)
  {
    Iot[i] = new NetIO(party == 1 ? nullptr : address.c_str(), port + i * 3);
    Otpackt[i] = new OTPack<NetIO>(Iot[i], party);
    Auxt[i] = new AuxProtocols(party, Iot[i], Otpackt[i]);
    Prodt[i] = new LinearOT(party, Iot[i], Otpackt[i]);
  }
  srand(time(0) + party);
  // shar();
  cout << "Party: CS_" << party << endl;

  vector<int> input_dim = {2, 3, 4, 5, 6};                                // size = 4
  vector<int> input_size = {1000, 3000, 5000, 7000, 9000, 11000, 100000}; // size = 6
  vector<int> rate = {10, 20, 30, 40, 50};                                // size = 4
  vector<string> path = {"small_anti_corr.txt", "small_correlate.txt", "small_uniformly.txt", "nba.txt", "hotel.txt"};
  // Pd = 0, Pst = 0, Pr = 0, Pn = 4;

  m = input_dim[Pd];
  n = input_size[Pst];
  int r = rate[Pr];
  filename = path[Pn];
  data_path = "./data/data" + to_string(m) + "/size=" + to_string(n) + "/rate=" + to_string(r) + "/" + filename;
  // m = 3;
  // n = 9;
  // data_path = "./10.txt";
  cout << data_path << endl;
  vector<vector<uint32_t>> p(n, vector<uint32_t>(m));
  vector<vector<uint32_t>> zeroM(n, vector<uint32_t>(m));
  loadP(p, zeroM, data_path);
  double Qtime = 0, QCom = 0, Qpre = 0, Qque = 0, Qsort = 0, Qlen = 0, Qnum = 0;
  int itrs = 1 << m;
  cout << itrs << endl;
  for (int itr = 0; itr < itrs; itr++)
  {
    vector<uint32_t> q(m);
    for (int j = 0; j < m; j++)
    {
      q[j] = (itr >> j) & 1;
    }
    if (party == ALICE)
    {
      uint32_t *q0 = new uint32_t[m];
      for (int j = 0; j < m; j++)
      {
        q0[j] = q[j];
      }
      Iot[0]->send_data(q0, m * sizeof(uint32_t));
      delete[] q0;
    }
    else
    {
      uint32_t *q0 = new uint32_t[m];
      Iot[0]->recv_data(q0, m * sizeof(uint32_t));
      for (int j = 0; j < m; j++)
      {
        q[j] = q0[j];
      }
      delete[] q0;
    }
    cout << itr << "-itr:\t" << endl;
    cout << "q: ";
    for (int j = 0; j < m; j++)
    {
      cout << q[j] << "\t";
    }
    cout << endl;
    int len;
    len = plainB(p, zeroM, q);
    cout << "CS_" + to_string(party) + " Skyline Number\t" << RED << len << RESET << endl;
    cout << "CS_" + to_string(party) + " Dom Number\t" << RED << countDom << RESET << endl;
    cout << "+++" << endl;
    // len = plainB_min(p, zeroM, q);
    // cout << "CS_" + to_string(party) + " Skyline Number\t" << RED << len << RESET << endl;
    // cout << "CS_" + to_string(party) + " Dom Number\t" << RED << countDom << RESET << endl;
    // cout << "+++" << endl;
    // len = plainB_Dom(p, zeroM, q, 1);
    // cout << "CS_" + to_string(party) + " Skyline Number\t" << RED << len << RESET << endl;
    // cout << "CS_" + to_string(party) + " Dom Number\t" << RED << countDom << RESET << endl;
    // cout << "+++" << endl;
    // BSS(p, zeroM, q);
    // BSS_minPre(p, zeroM, q);
    // BSS_T(p, zeroM, q);
    uint64_t comm_start = 0;
    for (int j = 0; j < THs; j++)
    {
      comm_start += Iot[j]->counter;
    }
    double start = omp_get_wtime();
    if (Pth == 0)
    {
      len = BSS_min(p, zeroM, q);
    }
    else
    {
      len = BSS_min_T(p, zeroM, q);
    }
    // int len = BSS_min(p, zeroM, q);
    // int len = BSS_min_T(p, zeroM, q);
    // int len = BSS_min_bucket(p, zeroM, q);
    // int len = FSS(p, zeroM, q);
    // int len = FSS_T(p, zeroM, q);
    double end = omp_get_wtime();
    Qtime += end - start;
    uint64_t comm_end = 0;
    for (int j = 0; j < THs; j++)
    {
      comm_end += Iot[j]->counter;
    }
    QCom += comm_end - comm_start;
    Qpre += pre_time;
    Qsort += sort_time;
    Qque += que_time;
    Qlen += len;
    Qnum += countDom;
    delete[] SS_p;
    delete[] SS_zeroM;
    delete[] Skyline;
    delete[] SS_q;
    SS_pB.clear();
    cout << "CS_" + to_string(party) + " Skyline Number\t" << RED << len << RESET << endl;
    cout << "CS_" + to_string(party) + " Total Time\t" << RED << (end - start) << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " process Time\t" << RED << pre_time << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " sort Time\t" << RED << sort_time << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " Query Time\t" << RED << que_time << " s" << RESET << endl;
    cout << "CS_" + to_string(party) + " Communication\t" << BLUE <<  (comm_end - comm_start) / 1024 / 1024 << " MB" << RESET << endl;
  }
  Qtime = Qtime / itrs;
  QCom = QCom / itrs / 1024 / 1024;
  Qpre = Qpre / itrs;
  Qsort = Qsort / itrs;
  Qque = Qque / itrs;
  Qlen = Qlen / itrs;
  Qnum = Qnum / itrs;
  cout << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) << endl;
  cout << "CS_" + to_string(party) + " Skyline Number\t" << RED << Qlen << RESET << endl;
  cout << "CS_" + to_string(party) + " Dom Number\t" << RED << Qnum << RESET << endl;
  cout << "CS_" + to_string(party) + " Total Time\t" << RED << Qtime << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " process Time\t" << RED << Qpre << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " sort Time\t" << RED << Qsort << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " Query Time\t" << RED << Qque << " s" << RESET << endl;
  cout << "CS_" + to_string(party) + " Communication\t" << BLUE << QCom << " MB" << RESET << endl;
  ofstream outfile;
  string pp = "../../tests/out_" + to_string(party) + ".txt";
  if (Pth == 1)
  {
    pp = "../../tests/out_" + to_string(party) + "_T.txt";
  }
  outfile.open(pp, ios::app | ios::in);
  outfile << " -------------------------------------" << endl;
  outfile << "n = " + to_string(n) + ", m = " + to_string(m) + ", rate = " + to_string(r) + " " + data_path << endl;
  outfile << "CS_" + to_string(party) + " Skyline Number:" << Qlen << endl;
  outfile << "CS_" + to_string(party) + " Dom Number:" << Qnum << endl;
  outfile << "CS_" + to_string(party) + " Total Time:" << Qtime << " s" << endl;
  outfile << "CS_" + to_string(party) + " process Time:" << Qpre << " s" << endl;
  outfile << "CS_" + to_string(party) + " sort Time:" << Qsort << " s" << endl;
  outfile << "CS_" + to_string(party) + " Query Time:" << Qque << " s" << endl;
  outfile << "CS_" + to_string(party) + " Communication:" << QCom << " MB" << endl;
  outfile << " -------------------------------------" << endl;
  outfile.close();
  for (int i = 0; i < THs; i++)
  {
    delete Auxt[i];
    delete Prodt[i];
    delete Otpackt[i];
    delete Iot[i];
  }
  return 0;
}
