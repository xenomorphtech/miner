#pragma once
#include <cstddef>
#include <cstdlib>

namespace thrust { namespace system { namespace cuda { namespace experimental {

template<class T> struct pinned_allocator {
  using value_type = T;
  __host__ pinned_allocator() noexcept = default;
  template<class U> __host__ pinned_allocator(const pinned_allocator<U>&) noexcept {}
  __host__ T* allocate(std::size_t n) { return static_cast<T*>(::malloc(n*sizeof(T))); }
  __host__ void deallocate(T* p, std::size_t) noexcept { ::free(p); }
};

}}}}  // namespaces
