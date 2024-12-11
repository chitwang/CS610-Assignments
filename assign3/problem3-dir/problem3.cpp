#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <pthread.h>
#include <string>
#include <omp.h>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <unistd.h>

int main(int argc, char** argv) { 
    const std::string input_file = std::string(argv[1]).substr(5);
    const int producer_threads = std::stoi(std::string(argv[2]).substr(5));
    const int lines_per_thread = std::stoi(std::string(argv[3]).substr(5));
    const int bufferSize = std::stoi(std::string(argv[4]).substr(5));
    const std::string output_file = std::string(argv[5]).substr(5);
    std::ifstream inputstream = std::ifstream(input_file);
    
    std::mutex shared_buffer_mutex;
    std::queue<std::string> shared_buffer;
    std::condition_variable cv;

   	bool done = false;	

    omp_set_nested(true);   // turn on nested parallelism 
   #pragma omp parallel num_threads(2)
    {
	    /// one consumer and one producer threads
	    const int tid = omp_get_thread_num();
	    if(tid == 0){
	    	// consumer thread
			std::ofstream fout(output_file);
            if(!fout){
            	std::cerr << "Unable to open output file\n";
                	// return;
           	}
            while (!done or !shared_buffer.empty()){
				// sleep(rand()%2);
                std::unique_lock<std::mutex> lock(shared_buffer_mutex);
            	cv.wait(lock, [&]{
            		return !shared_buffer.empty() || done;
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
	    else{
		   #pragma omp parallel num_threads(producer_threads)
			{
				std::vector<std::string> local_buffer;
				while(true){
				   #pragma omp critical (read_file)
					{
						std::string line;
						for(int i=0; i < lines_per_thread && std::getline(inputstream, line); i++){
							local_buffer.push_back(line);
						}
					}
					if(local_buffer.empty()){
							break;
					}
				   #pragma omp critical (producer_work)
					{
						for(const auto &line: local_buffer){
							std::unique_lock<std::mutex> consumer_lock(shared_buffer_mutex);
							cv.wait(consumer_lock, [&]{
									return shared_buffer.size() < bufferSize;
							});
							shared_buffer.push(line);
							cv.notify_all();
							// sleep(rand()%5);    //used to test various contention scenarios   
						}
					}
					local_buffer.clear();
            		}
            	   #pragma omp barrier
            	   #pragma omp single nowait
            		{
						done = true;
                		cv.notify_all();
            		}
			// std::cout << "Producer " << std::endl;
			}

	    }
    }
    return EXIT_SUCCESS; 
}
