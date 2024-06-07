#ifndef ACL_RESNET50_ALLOCATOR_H
#define ACL_RESNET50_ALLOCATOR_H
#include "acl/acl.h"
#include "ge/ge_allocator.h"

namespace mds {
class Allocator {
 public:
  Allocator() = default;
  ~Allocator() = default;
  Allocator(const Allocator &) = delete;
  Allocator &operator=(const Allocator &) = delete;

  void *Malloc(size_t size) {
    void *block = nullptr;
    (void)aclrtMalloc(&block, size, ACL_MEM_MALLOC_HUGE_FIRST);
    return block;
  }

  void Free(void *block) {
    (void)aclrtFree(block);
  }

  void *MallocAdvise(size_t size, void *addr) {
    (void)addr;
    return Malloc(size);
  }
};
}  // namespace mds

#endif  // ACL_RESNET50_ALLOCATOR_H
