#if !defined(CSTACK)
#define CSTACK

#include <atomic>
#include <iostream>
#include <cstdint>

struct alignas(16) Node {
    int data;
    Node* next;
};


class LFStack {
private:
    std::atomic<std::uintptr_t> head; 

    static constexpr std::uintptr_t TAG_MASK = 0xF;
    static constexpr std::uintptr_t PTR_MASK = ~TAG_MASK;

public:
    LFStack() : head(0) {}

    void push(int value) {
        Node* node = new (std::align_val_t(16)) Node;
        node->data = value;
        std::uintptr_t old_head = head.load();
        while (true) {
            node->next = reinterpret_cast<Node*>(old_head & PTR_MASK);
            std::uintptr_t new_head = reinterpret_cast<std::uintptr_t>(node) | ((old_head + 1) & TAG_MASK);
            if (head.compare_exchange_weak(old_head, new_head)) {
                break;
            }
        }
    }

    int pop() {
        std::uintptr_t old_head = head.load();
        while (true) {
            Node* node = reinterpret_cast<Node*>(old_head & PTR_MASK);
            if (!node) return -1;
            std::uintptr_t new_head = reinterpret_cast<std::uintptr_t>(node->next) | ((old_head + 1) & TAG_MASK);
            if (head.compare_exchange_weak(old_head, new_head)) {
                int value = node->data;
                return value;
            }
        }
    }

    void print_stack() const {
        std::uintptr_t current_head = head.load();
        Node* node = reinterpret_cast<Node*>(current_head & PTR_MASK);
        std::cout << "Stack contents (top to bottom): ";
        while (node != nullptr) {
            std::cout << node->data << " ";
            node = node->next;
        }
        std::cout << std::endl;
    }
};


#endif 


