#include <initializer_list>
#include <vector>
#include <array>
#include <numeric>
#include <iostream>

template <typename Derived>
class DenseVector {
  protected:
    ~DenseVector() = default;
  public:
    Derived& derived() { return static_cast<Derived&>(*this); }
    Derived const& derived() const { return static_cast<Derived const&>(*this); }
    decltype(auto) begin() { return derived().begin(); }
    decltype(auto) end() { return derived().end(); }
    decltype(auto) begin() const { return derived().begin(); }
    decltype(auto) end() const { return derived().end(); }
    size_t size() { return derived().size(); }
    auto sum() const { return std::accumulate(begin(), end(), typename Derived::value_type{}); }
    auto sqnorm() const {return std::accumulate(begin(), end(), typename Derived::value_type{}, [](auto acc, auto x) { return acc + x * x; }); }
    // and more...
};

template <typename T>
class DynamicVector : public DenseVector<DynamicVector<T>>
{
  public:
    using value_type = T; 
    DynamicVector(size_t size) : data_(size) {}
    DynamicVector(std::initializer_list<T> list) : data_(list) {}
    size_t size() { return this->data_.size(); }
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
  private:
    std::vector<T> data_;
};

template <typename T, size_t N>
class StaticVector : public DenseVector<StaticVector<T, N>>
{
  public:
    using value_type = T; 
    StaticVector(std::initializer_list<T> list) {
      std::copy(list.begin(), list.end(), data_.begin());
    }
    size_t size() { return N; }
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
  private:
    std::array<T, N> data_;
};

int main() {
  auto dyn = DynamicVector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto stat = StaticVector<int, 10>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  dyn.size();
  stat.size();
  std::cout << stat.sum() << std::endl;
  std::cout << dyn.sum() << std::endl;
}
