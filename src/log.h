#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "acl/acl_base.h"

using Status = uint32_t;
constexpr const int32_t SUCCESS = ACL_SUCCESS;
constexpr const int32_t FAILED = ACL_ERROR_FAILURE;

// 定义日志级别
enum LogLevel {
  LOG_LEVEL_DEBUG,
  LOG_LEVEL_INFO,
  LOG_LEVEL_WARN,
  LOG_LEVEL_ERROR,
};

// 获取当前时间字符串
inline std::string currentDateTime() {
  std::time_t now = std::time(nullptr);
  std::tm *tm_now = std::localtime(&now);
  std::ostringstream oss;
  oss << std::put_time(tm_now, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

// 基础日志宏实现
#define LOG(level, format, ...)                                                 \
  do {                                                                          \
    if (level >= LOG_LEVEL) {                                                   \
      std::ostringstream oss;                                                   \
      oss << "[" << logLevelToString(level) << "] ";                            \
      oss << "[" << currentDateTime() << "] ";                                  \
      oss << "[" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << "] "; \
      oss << formatString(format, ##__VA_ARGS__) << std::endl;                  \
      std::cout << oss.str();                                                   \
    }                                                                           \
  } while (0)

// 日志级别转换为字符串
inline const char *logLevelToString(LogLevel level) {
  switch (level) {
    case LOG_LEVEL_DEBUG:
      return "DEBUG";
    case LOG_LEVEL_INFO:
      return "INFO";
    case LOG_LEVEL_WARN:
      return "WARN";
    case LOG_LEVEL_ERROR:
      return "ERROR";
    default:
      return "UNKNOWN";
  }
}

// 格式化字符串
template <typename... Args>
std::string formatString(const std::string &format, Args... args) {
  // todo fix compile warning: format not a string literal and no format arguments [-Wformat-security]
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
  if (size_s <= 0) {
    return "Error during formatting log.";
  }
  auto size = static_cast<size_t>(size_s);
  auto buf = std::unique_ptr<char[]>(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size - 1);
}

// 设置默认的日志级别
constexpr LogLevel LOG_LEVEL = LOG_LEVEL_DEBUG;

#define LOG_DEBUG(format, ...) LOG(LOG_LEVEL_DEBUG, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) LOG(LOG_LEVEL_INFO, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) LOG(LOG_LEVEL_WARN, format, ##__VA_ARGS__)
#define LOG_ERROR(error_code, format, ...) LOG(LOG_LEVEL_ERROR, "error_code=%d," format, (error_code), ##__VA_ARGS__)

#define APP_CHK_STATUS(expr, ...)          \
  do {                                     \
    const Status _chk_status = (expr);     \
    if (_chk_status != SUCCESS) {          \
      LOG_ERROR(_chk_status, __VA_ARGS__); \
      return _chk_status;                  \
    }                                      \
  } while (false)

#define APP_CHK_NOTNULL(ptr, ...)                                   \
  do {                                                              \
    if ((ptr) == nullptr) {                                         \
      LOG_ERROR(FAILED, "Pointer " #ptr " is NULL, please check."); \
      return FAILED;                                                \
    }                                                               \
  } while (false)
