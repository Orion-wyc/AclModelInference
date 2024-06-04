#ifndef SRC_TIMER_H_
#define SRC_TIMER_H_
#include <chrono>

class Timer {
 public:
  Timer() : start_time_point(std::chrono::high_resolution_clock::now()) {}

  void Reset() {
    start_time_point = std::chrono::high_resolution_clock::now();
  }

  double Elapsed() const {
    auto end_time_point = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time_point - start_time_point).count();
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
};

#endif  // SRC_TIMER_H_