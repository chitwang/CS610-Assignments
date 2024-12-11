#if !defined(HASHSET)
#define HASHSET

#include <iostream>
#include <pthread.h>
#include <vector>
#include <atomic>
#include <cmath>
#include <cassert>
#include <functional>

constexpr float LOAD_FACTOR = 0.8f;
constexpr uint32_t EMPTY_KEY = 0;
constexpr uint32_t TOMBSTONE_KEY = UINT32_MAX;
constexpr uint32_t INITIAL_SIZE = 1000;
static const uint32_t PROBING_RETRIES = (1 << 20);



using std::cout;
using std::endl;
using HashFunction = std::function<size_t(uint32_t, size_t)>;

// Simple modulo-based hash function
size_t simple_hash(uint32_t key, size_t table_size) {
    return key % table_size;
}

// Simple modulo-based hash function
size_t simple_probe_hash(uint32_t key, size_t table_size) {
    return 1 + (key % (table_size - 1));
}

// Multiplicative hash function
size_t multiplicative_hash(uint32_t key, size_t table_size) {
    const double A = 0.6180339887;
    double frac = std::fmod(key * A, 1.0);
    return static_cast<size_t>(std::floor(table_size * frac));
}

// XOR Shift hash function
size_t xor_shift_hash(uint32_t key, size_t table_size) {
    key ^= key << 13;
    key ^= key >> 17;
    key ^= key << 5;
    return key % table_size;
}

// fnv1a hash function
size_t fnv1a_hash(uint32_t key, size_t table_size) {
    const uint32_t fnv_prime = 16777619u;
    const uint32_t offset_basis = 2166136261u;

    uint32_t hash = offset_basis;
    for (int i = 0; i < 4; ++i) {
        uint8_t byte = (key >> (i * 8)) & 0xFF;  
        hash ^= byte;
        hash *= fnv_prime;
    }
    return hash % table_size;
}

// jenkins hash function
size_t jenkins_hash(uint32_t key, size_t table_size) {
    key += (key << 12);
    key ^= (key >> 22);
    key += (key << 4);
    key ^= (key >> 9);
    key += (key << 10);
    key ^= (key >> 2);
    key += (key << 7);
    key ^= (key >> 12);
    return key % table_size;
}

std::vector<HashFunction> functions = {simple_hash, simple_probe_hash, multiplicative_hash, xor_shift_hash, jenkins_hash, fnv1a_hash};
std::vector<std::string> function_name = {"Modulo Hash", "Modulo Hash 2", "Multiplicative Hash", "XOR Shift Hash", "Jenkins Hash", "FNV1A Hash"};

struct HashTableEntry {
    uint32_t key;
    uint32_t value;
    HashTableEntry() : key(EMPTY_KEY), value(0) {}
};

class HashTable {
    std::vector<HashTableEntry> table;
    pthread_mutex_t *mutexes;
    size_t table_size;
    std::atomic<int> count;
    size_t probing_retries;
    pthread_rwlock_t table_lock; 
    HashFunction hash1;
    HashFunction hash2;

public:
    HashTable(HashFunction h1, HashFunction h2) : table_size(INITIAL_SIZE), probing_retries(PROBING_RETRIES), count(0), hash1(h1), hash2(h2) {
        table = std::vector<HashTableEntry> (table_size);
        mutexes = new pthread_mutex_t[table_size];
        for (size_t i = 0; i < table_size; ++i) {
            pthread_mutex_init(&mutexes[i], nullptr);
        }
        pthread_rwlock_init(&table_lock, nullptr);
    }

    ~HashTable() {
        for (size_t i = 0; i < table_size; ++i) {
            pthread_mutex_destroy(&mutexes[i]);
        }
        delete[] mutexes;
        pthread_rwlock_destroy(&table_lock);
    }


    bool insert(uint32_t key, uint32_t value) {
        while (true) {
            // Acquire read lock for regular operations
            if (pthread_rwlock_rdlock(&table_lock) != 0) {
                std::cerr << "Failed to acquire read lock in insert.\n";
                return false;
            }

            size_t idx = hash1(key, table_size);
            size_t step = hash2(key, table_size);

            bool inserted = false;

            for (size_t i = 0; i < PROBING_RETRIES; ++i) {
                size_t probe_idx = (idx + i * step) % table_size;

                // Safety Check
                assert(probe_idx < table_size && "probe_idx out of bounds!");

                if (pthread_mutex_lock(&mutexes[probe_idx]) != 0) {
                    std::cerr << "Failed to acquire mutex for index " << probe_idx << " in insert.\n";
                    pthread_rwlock_unlock(&table_lock);
                    return false;
                }

                uint32_t current_key = table[probe_idx].key;
                if (current_key == EMPTY_KEY || current_key == TOMBSTONE_KEY) {
                    table[probe_idx].key = key;
                    table[probe_idx].value = value;
                    count.fetch_add(1);
                    pthread_mutex_unlock(&mutexes[probe_idx]);
                    pthread_rwlock_unlock(&table_lock);

                    bool should_resize = false;

                    if (pthread_rwlock_rdlock(&table_lock) != 0) {
                        std::cerr << "Failed to re-acquire read lock in insert.\n";
                        return false;
                    }

                    if (static_cast<float>(count.load()) / table_size > LOAD_FACTOR) {
                        should_resize = true;
                    }
                    pthread_rwlock_unlock(&table_lock);

                    if (should_resize) {
                        resize();
                    }

                    return true;
                } 
                else if (current_key == key) {
                    pthread_mutex_unlock(&mutexes[probe_idx]);
                    pthread_rwlock_unlock(&table_lock);
                    return false;  // Duplicate key
                }

                pthread_mutex_unlock(&mutexes[probe_idx]);
            }

            pthread_rwlock_unlock(&table_lock);

            // If probing retries exceeded, resize and retry
            if (static_cast<float>(count.load()) / table_size > LOAD_FACTOR) {
                resize();
            }
            else{
                return false;
            }
        }
    }


    bool remove_key(uint32_t key) {  
        if (pthread_rwlock_rdlock(&table_lock) != 0) {
            std::cerr << "Failed to acquire read lock in remove_key.\n";
            return false;
        }

        size_t idx = hash1(key, table_size);
        size_t step = hash2(key, table_size);

        for (size_t i = 0; i < PROBING_RETRIES; ++i) {
            size_t probe_idx = (idx + i * step) % table_size;
            // Safety Check
            assert(probe_idx < table_size && "probe_idx out of bounds!");

            if (pthread_mutex_lock(&mutexes[probe_idx]) != 0) {
                std::cerr << "Failed to acquire mutex for index " << probe_idx << " in remove_key.\n";
                pthread_rwlock_unlock(&table_lock);
                return false;
            }

            uint32_t current_key = table[probe_idx].key;
            if (current_key == key) {
                // cout << key << " " << probe_idx << endl;
                table[probe_idx].key = TOMBSTONE_KEY;
                count.fetch_sub(1);
                pthread_mutex_unlock(&mutexes[probe_idx]);
                pthread_rwlock_unlock(&table_lock);
                return true;
            } 
            else if (current_key == EMPTY_KEY) {
                pthread_mutex_unlock(&mutexes[probe_idx]);
                pthread_rwlock_unlock(&table_lock);
                return false;
            }

            pthread_mutex_unlock(&mutexes[probe_idx]);
        }

        pthread_rwlock_unlock(&table_lock);
        return false;
    }

    uint32_t lookup(uint32_t key) {

        size_t idx = hash1(key, table_size);
        size_t step = hash2(key, table_size);

        for (size_t i = 0; i < PROBING_RETRIES; ++i) {
            size_t probe_idx = (idx + i * step) % table_size;

            // Safety Check
            assert(probe_idx < table_size && "probe_idx out of bounds!");

            uint32_t current_key = table[probe_idx].key;
            // cout << "Searching for key " << current_key << endl; 
            if (current_key == key) {
                int32_t value = table[probe_idx].value;
                return value;
            } 
            else if (current_key == EMPTY_KEY) {
                return -1;
            }
        }
        return -1;
    }

private:
    void resize() {
        // Acquire write lock to block other operations during resize
        // std::cout << "Entered resize\n";
        if (pthread_rwlock_wrlock(&table_lock) != 0) {
            std::cerr << "Failed to acquire write lock in resize.\n";
            return;
        }
        if (static_cast<float>(count.load()) / table_size <= LOAD_FACTOR) {
            // cout << "Policy\n";
            pthread_rwlock_unlock(&table_lock);
            return;
        }
        size_t new_table_size = table_size * 2;
        std::vector<HashTableEntry> new_table(new_table_size);
        pthread_mutex_t *new_mutexes = new pthread_mutex_t[new_table_size];
        for (size_t i = 0; i < new_table_size; ++i) {
            if (pthread_mutex_init(&new_mutexes[i], nullptr) != 0) {
                std::cerr << "Mutex initialization failed for new index " << i << " in resize.\n";
                // Clean up already initialized mutexes
                for (size_t j = 0; j < i; ++j) {
                    pthread_mutex_destroy(&new_mutexes[j]);
                }
                delete[] new_mutexes;
                pthread_rwlock_unlock(&table_lock);
                exit(EXIT_FAILURE);
            }
        }

        for (size_t i = 0; i < table_size; ++i) {
            uint32_t current_key = table[i].key;
            if (current_key != EMPTY_KEY && current_key != TOMBSTONE_KEY) {
                uint32_t value = table[i].value;
                size_t idx = hash1(current_key, new_table_size);
                size_t step = hash2(current_key, new_table_size);
                bool inserted = false;
                for (size_t j = 0; j < new_table_size; ++j) {
                    size_t probe_idx = (idx + j * step) % new_table_size;

                    // Safety Check
                    assert(probe_idx < new_table_size && "probe_idx out of bounds during resize.");

                    if (pthread_mutex_lock(&new_mutexes[probe_idx]) != 0) {
                        std::cerr << "Failed to acquire mutex for new index " << probe_idx << " in resize.\n";
                        pthread_rwlock_unlock(&table_lock);
                        return;
                    }

                    uint32_t target_key = new_table[probe_idx].key;
                    if (target_key == EMPTY_KEY || target_key == TOMBSTONE_KEY) {
                        new_table[probe_idx].key = current_key;
                        new_table[probe_idx].value = value;
                        pthread_mutex_unlock(&new_mutexes[probe_idx]);
                        inserted = true;
                        break;
                    }

                    pthread_mutex_unlock(&new_mutexes[probe_idx]);
                }

            }
        }

        for (size_t i = 0; i < table_size; ++i) {
            pthread_mutex_destroy(&mutexes[i]);
        }
        delete[] mutexes;

        table = std::move(new_table);
        mutexes = new_mutexes;
        table_size = new_table_size;

        // std::cout << "Resized table to new size: " << table_size << "\n";

        pthread_rwlock_unlock(&table_lock);
    }
};


#endif
