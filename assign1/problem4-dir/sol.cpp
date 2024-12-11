#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>
#include <papi.h>
#include <vector>

using namespace std;
using namespace std::chrono;

using HR = high_resolution_clock;
using HRTimer = HR::time_point;

#define N (2048)
int BLK_A, BLK_B, BLK_C;

void matmul_ijk(const uint32_t *A, const uint32_t *B, uint32_t *C, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      uint32_t sum = 0.0;
      for (int k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      C[i * SIZE + j] += sum;
    }
  }
}

void matmul_ijk_blocking(const uint32_t *A, const uint32_t *B, uint32_t *C, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      uint32_t sum = 0.0;
      for (int k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      C[i * SIZE + j] += sum;
    }
  }
}

void matmul_ijk_block(const uint32_t *A, const uint32_t *B, uint32_t *C, const int SIZE) {
        for (int i = 0; i < SIZE; i+=BLK_A) {
                for (int j = 0; j < SIZE; j+=BLK_B) {
                        for (int k = 0; k < SIZE; k+=BLK_C) {
                        /* B×B mini -matrix (blocks) multiplications */
                                for (int i1 = i; i1 < i+BLK_A; i1++) {
                                        for (int j1 = j; j1 < j+BLK_B; j1++){
                                                uint32_t sum = 0;
                                                for (int k1 = k; k1 < k+BLK_C; k1++) {
                                                        sum += A[i1 * SIZE + k1] * B[k1 * SIZE + j1];
                                                }
                                                C[i1 * SIZE + j1] += sum;
                                        }
                                }
                        }
                }
        }
}

void init(uint32_t *mat, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      mat[i * SIZE + j] = 1;
    }
  }
}

void print_matrix(const uint32_t *mat, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      cout << mat[i * SIZE + j] << "\t";
    }
    cout << "\n";
  }
}

void check_result(const uint32_t *ref, const uint32_t *opt, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      if (ref[i * SIZE + j] != opt[i * SIZE + j]) {
        assert(false && "Diff found between sequential and blocked versions!\n");
      }
    }
  }
}

int main(int argc, char *argv[]) {
        int retval = PAPI_library_init(PAPI_VER_CURRENT);
        if(retval != PAPI_VER_CURRENT and retval > 0){
                cerr << "PAPI library version mismatch\n";
                exit(EXIT_FAILURE);
        }
        else if(retval < 0){
                cerr << "PAPI library initialiation error\n";
                exit(EXIT_FAILURE);
        }

        int eventset = PAPI_NULL;
        retval = PAPI_create_eventset(&eventset);
        if(retval != PAPI_OK){
                cerr << "Error at PAPI create eventset\n";
                exit(EXIT_FAILURE);
        }

        // vector<long long> papi_events = {PAPI_L1_DCH, PAPI_L2_DCH, PAPI_L1_DCA, PAPI_L2_DCA, PAPI_L3_DCA};
        vector<int> papi_events = {PAPI_L1_DCM, PAPI_L2_DCM};
        for(auto &ev: papi_events){
                if(PAPI_add_event(eventset, ev) != PAPI_OK){
                        cerr << "Error in adding event " << ev << "\n";
                        exit(EXIT_FAILURE);
                }
        }

        uint32_t *A = new uint32_t[N * N];
        uint32_t *B = new uint32_t[N * N];
        uint32_t *C_seq = new uint32_t[N * N];

        init(A, N);
        init(B, N);
        init(C_seq, N);

        BLK_A = stoi(argv[1]);
        BLK_B = stoi(argv[2]);
        BLK_C = stoi(argv[3]);

        retval = PAPI_start(eventset);
        if(retval != PAPI_OK){
                cerr << "Error at PAPI_start" << endl;
                exit(EXIT_FAILURE);
        }

        HRTimer start = HR::now();
        matmul_ijk(A, B, C_seq, N);
        HRTimer end = HR::now();
        auto duration1 = duration_cast<microseconds>(end - start).count();
  		  cout << "Time without blocking (us): " << duration1 << "\n";

        long long int values_non_blocking[papi_events.size()];
        retval = PAPI_stop(eventset, values_non_blocking);
        if(retval != PAPI_OK){
                cerr << "Error at PAPI_stop" << endl;
                exit(EXIT_FAILURE);
        }


        uint32_t *C_blk = new uint32_t[N * N];
        init(C_blk, N);

	      retval = PAPI_start(eventset);
        if(retval != PAPI_OK){
                cerr << "Error at PAPI_start" << endl;
                exit(EXIT_FAILURE);
        }
        
        start = HR::now();
        matmul_ijk_block(A, B, C_blk, N);
        end = HR::now();
        auto duration2 = duration_cast<microseconds>(end - start).count();
  		  cout << "Time with blocking (us): " << duration2 << "\n";

        long long int values_blocking[papi_events.size()];
        retval = PAPI_stop(eventset, values_blocking);
        if(retval != PAPI_OK){
                cerr << "Error at PAPI_stop" << endl;
                exit(EXIT_FAILURE);
        }

        ofstream fout("./data_csews.csv", ios::app);
        fout << BLK_A << ',' << BLK_B << ',' << BLK_C << ',' << duration1 << ',' << duration2;
        for(int i=0; i<papi_events.size(); i++){
                fout << ',' << values_non_blocking[i] << ',' << values_blocking[i];
        }
        fout << endl;
        fout.close();
  
        check_result(C_seq, C_blk, N);

        return EXIT_SUCCESS;
}