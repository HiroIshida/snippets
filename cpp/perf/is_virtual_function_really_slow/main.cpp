#include <iostream>
#include <memory>
#include <vector>
#include <array>
#include <chrono>

class ShapeInterface {
public:
    virtual bool is_inside(double x, double y) const = 0;
    virtual ~ShapeInterface() = default; // Virtual destructor for interface
};

class Rectangle : public ShapeInterface {
public:
    Rectangle(double x, double y, double w, double h) : x(x), y(y), w(w), h(h) {}

    bool is_inside(double x, double y) const override {
        return x >= this->x && x <= this->x + this->w && y >= this->y && y <= this->y + this->h;
    }

private:
    double x, y, w, h;
};

class Sphere : public ShapeInterface {
public:
    Sphere(double x, double y, double r) : x(x), y(y), r(r) {}

    bool is_inside(double x, double y) const override {
        return (x - this->x) * (x - this->x) + (y - this->y) * (y - this->y) <= r * r;
    }
private:
    double x, y, r;
};


void benchmark_polymorphic(const std::vector<std::unique_ptr<ShapeInterface>>& shapes, double x, double y, size_t n_bench) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t sum = 0;
    for (size_t i = 0; i < n_bench; i++) {
        for (const auto& shape : shapes) {
            sum += shape->is_inside(x, y);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Polymorphic (unique_ptr) elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::cout << "sum: " << sum << std::endl; // to avoid optimization
}

void benchmark_casted(const std::vector<std::unique_ptr<ShapeInterface>>& shapes, double x, double y, size_t n_bench) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t sum = 0;
    for (size_t i = 0; i < n_bench; i++) {
        for (const auto& shape : shapes) {
            sum += static_cast<Rectangle*>(shape.get())->is_inside(x, y);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Casted (static_cast) elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::cout << "sum: " << sum << std::endl; // to avoid optimization
}

int main() {
    const size_t n_bench = 100000000;
    const double x = 3, y = 3;

    // Setup shapes with unique_ptr
    std::vector<std::unique_ptr<ShapeInterface>> shapes_unique_ptr;
    for(int i = 0; i < 5; i++) {
        shapes_unique_ptr.push_back(std::make_unique<Rectangle>(i, i, 1, 1));
    }
    for(int i = 0; i < 5; i++) {
        shapes_unique_ptr.push_back(std::make_unique<Sphere>(i, i, 1));
    }
    benchmark_polymorphic(shapes_unique_ptr, x, y, n_bench);
    benchmark_casted(shapes_unique_ptr, x, y, n_bench);
    return 0;
}
