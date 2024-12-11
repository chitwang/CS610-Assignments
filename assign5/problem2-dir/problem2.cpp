#include <iostream>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>
#include <random>
#include <cassert>
#include <atomic>
#include <chrono>
#include "cstack.hpp" 

using std::chrono::duration_cast;
using std::chrono::milliseconds;

void read_data(const std::filesystem::path& path, uint64_t num_ops, uint32_t* buffer) {
    std::ifstream file(path, std::ios::binary);
    assert(file.is_open());
    file.read(reinterpret_cast<char*>(buffer), num_ops * sizeof(uint32_t));
    file.close();
}


void thread_task(LFStack& stack, uint32_t* insert_vals, uint64_t num_ops_per_thread, double push_prob) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    int push_ptr = 0;
    for (uint64_t i = 0; i < num_ops_per_thread; ++i) {
        double random_value = dist(gen);
        if (random_value < push_prob) {
            // std::cout << "Pushing " << insert_vals[push_ptr] << std::endl;
            stack.push(insert_vals[push_ptr++]);
        } else {
            // std::cout << "Popping " << std::endl;
            stack.pop();
        }
    }
}


int main(int argc, char* argv[]) {
    uint64_t NUM_OPS = std::stod(argv[1]);
    uint64_t NUM_THREADS = std::stoll(argv[2]);
    double push_probability = 0.6; 
    std::cout << "NUM OPS " << NUM_OPS << std::endl;
    std::cout << "NUM THREADS " << NUM_THREADS << std::endl;

    std::filesystem::path cwd = std::filesystem::current_path();
    // cwd = "/tmp/";
    std::filesystem::path path_insert_values = cwd / "random_values_insert.bin";

    assert(std::filesystem::exists(path_insert_values));

    auto* insert_vals = new uint32_t[NUM_OPS];
    read_data(path_insert_values, NUM_OPS, insert_vals);

    LFStack stack;

    uint64_t num_ops_per_thread = NUM_OPS / NUM_THREADS;
    std::vector<std::thread> threads;

    auto start = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(thread_task, std::ref(stack), insert_vals + i * num_ops_per_thread, num_ops_per_thread, push_probability);
    }

    for (auto& thread : threads) {
        thread.join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    float elapsed = duration_cast<milliseconds>(end - start).count();
    // std::chrono::duration<double> elapsed = end - start;

    delete[] insert_vals;
    std::cout << "All operations completed in " << elapsed << " milliseconds." << std::endl;

    return 0;
}
