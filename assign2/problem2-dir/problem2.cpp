#include<iostream>
#include<mutex>
#include<queue>
#include<vector>
#include<string>
#include<condition_variable>
#include<atomic>
#include<fstream>
#include<thread>
#include<unistd.h>


std::mutex file_mtx;
std::mutex shared_buffer_mutex;
std::mutex producer_mutex;
std::queue<std::string> shared_buffer;
std::condition_variable cv;
std::atomic<bool> done(false);
std::ifstream inputstream;

int bufferSize;
int lines_per_thread;

void producer_function(int tid)
{
    // std::cout << tid << std::endl;
    std::vector<std::string> local_buffer;
    while (true)   // never stop reading the file
    {
        // contend for lock on ifstream
        {
            std::lock_guard<std::mutex> lock(file_mtx);
            std::string line;
            for(int i=0; i < lines_per_thread && std::getline(inputstream, line); i++){
                local_buffer.push_back(line);
            }
        }
        if(local_buffer.empty()){   // file is completely read
            break;
        }
        std::unique_lock<std::mutex> producer_lock(producer_mutex);   // lock the producer mutex so that no other producer can enter this section
        for(const auto &line: local_buffer){
            std::unique_lock<std::mutex> consumer_lock(shared_buffer_mutex);
            cv.wait(consumer_lock, []{
                return shared_buffer.size() < bufferSize;
            });
            shared_buffer.push(line);
            cv.notify_all();
            // sleep(rand()%5);    //used to test various contention scenarios   
        }
        producer_lock.unlock();
        local_buffer.clear();
    }
}

void consumer_function(std::string output_file)
{
    std::ofstream fout(output_file);
    if(!fout){
        std::cerr << "Unable to open output file\n";
        return;
    }
    while (!done.load() || !shared_buffer.empty())
    {
        std::unique_lock<std::mutex> lock(shared_buffer_mutex);
        cv.wait(lock, []{
            return !shared_buffer.empty() || done.load();
        });
        if(!shared_buffer.empty()){
            const auto line  = shared_buffer.front();
            shared_buffer.pop();
            cv.notify_all();
            lock.unlock();
            fout << line << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    std::string input_file;
    std::string output_file;
    int producer_threads;
    input_file = std::string(argv[1]).substr(5);
    producer_threads = std::stoi(std::string(argv[2]).substr(5));
    lines_per_thread = std::stoi(std::string(argv[3]).substr(5));
    bufferSize = std::stoi(std::string(argv[4]).substr(5));
    output_file = std::string(argv[5]).substr(5);
    inputstream = std::ifstream(input_file);
    if(!inputstream){
        std::cerr << "Unable to open input file\n";
        exit(EXIT_FAILURE);
    }
    // std::cout << input_file << std::endl;

    std::vector<std::thread> threads;
    for(int tid=1; tid <= producer_threads; tid++){
        threads.emplace_back(producer_function, tid);
    }
    std::thread consumer_thread(consumer_function, output_file);
    for(auto &th:threads){
        th.join();
    }
    done.store(true);
    cv.notify_all();
    consumer_thread.join();
    exit(EXIT_SUCCESS);
}