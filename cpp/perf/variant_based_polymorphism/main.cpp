#include <iostream>
#include <memory>
#include <vector>
#include <array>
#include <chrono>
#include <variant>

class Rectangle {
public:
    Rectangle(double x, double y, double w, double h) : x(x), y(y), w(w), h(h) {}
    bool is_inside(double x, double y) const {
        return x >= this->x && x <= this->x + this->w && y >= this->y && y <= this->y + this->h;
    }
private:
    double x, y, w, h;
};

class Sphere {
public:
    Sphere(double x, double y, double r) : x(x), y(y), r(r) {}
    bool is_inside(double x, double y) const {
        return (x - this->x) * (x - this->x) + (y - this->y) * (y - this->y) <= r * r;
    }
private:
    double x, y, r;
};

using Shape = std::variant<Rectangle, Sphere>;

class ShapeInterface {
public:
    virtual bool is_inside(double x, double y) const = 0;
    virtual ~ShapeInterface() = default;
};

class RectanglePolymorphic : public ShapeInterface {
public:
    RectanglePolymorphic(double x, double y, double w, double h) : x(x), y(y), w(w), h(h) {}
    bool is_inside(double x, double y) const override {
        return x >= this->x && x <= this->x + this->w && y >= this->y && y <= this->y + this->h;
    }
private:
    double x, y, w, h;
};

class SpherePolymorphic : public ShapeInterface {
public:
    SpherePolymorphic(double x, double y, double r) : x(x), y(y), r(r) {}
    bool is_inside(double x, double y) const override {
        return (x - this->x) * (x - this->x) + (y - this->y) * (y - this->y) <= r * r;
    }
private:
    double x, y, r;
};

using ReuseShape = std::variant<RectanglePolymorphic, SpherePolymorphic>;

enum class ShapeType {
    Rectangle,
    Sphere
};

struct EnumShape {
    ShapeType type;
    double x, y, w_or_r, h;

    bool is_inside(double x, double y) const {
        switch (type) {
        case ShapeType::Rectangle:
            return x >= this->x && x <= this->x + this->w_or_r && y >= this->y && y <= this->y + this->h;
        case ShapeType::Sphere:
            return (x - this->x) * (x - this->x) + (y - this->y) * (y - this->y) <= w_or_r * w_or_r;
        }
        return false; // デフォルトケース
    }
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
    std::cout << "sum: " << sum << std::endl;
}

void benchmark_casted(const std::vector<std::unique_ptr<ShapeInterface>>& shapes, double x, double y, size_t n_bench) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t sum = 0;
    for (size_t i = 0; i < n_bench; i++) {
        for (const auto& shape : shapes) {
            sum += static_cast<RectanglePolymorphic*>(shape.get())->is_inside(x, y);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Casted (static_cast) elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::cout << "sum: " << sum << std::endl;
}

void benchmark_variant(const std::vector<Shape>& shapes, double x, double y, size_t n_bench) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t sum = 0;
    for (size_t i = 0; i < n_bench; i++) {
        for (const auto& shape : shapes) {
            sum += std::visit([x, y](const auto& s) { return s.is_inside(x, y); }, shape);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Variant (std::visit) elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::cout << "sum: " << sum << std::endl;
}

void benchmark_variant_get_if(const std::vector<Shape>& shapes, double x, double y, size_t n_bench) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t sum = 0;
    for (size_t i = 0; i < n_bench; i++) {
        for (const auto& shape : shapes) {
            if (auto rect = std::get_if<Rectangle>(&shape)) {
                sum += rect->is_inside(x, y);
            } else if (auto sphere = std::get_if<Sphere>(&shape)) {
                sum += sphere->is_inside(x, y);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Variant (get_if) elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::cout << "sum: " << sum << std::endl;
}

void benchmark_reuse_variant_get_if(const std::vector<ReuseShape>& shapes, double x, double y, size_t n_bench) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t sum = 0;
    for (size_t i = 0; i < n_bench; i++) {
        for (const auto& shape : shapes) {
            if (auto rect = std::get_if<RectanglePolymorphic>(&shape)) {
                sum += rect->is_inside(x, y);
            } else if (auto sphere = std::get_if<SpherePolymorphic>(&shape)) {
                sum += sphere->is_inside(x, y);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Reuse Variant (get_if) elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::cout << "sum: " << sum << std::endl;
}


void benchmark_enum_switch(const std::vector<EnumShape>& shapes, double x, double y, size_t n_bench) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t sum = 0;
    for (size_t i = 0; i < n_bench; i++) {
        for (const auto& shape : shapes) {
            sum += shape.is_inside(x, y);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Enum + Switch elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::cout << "sum: " << sum << std::endl;
}

int main() {
    const size_t n_bench = 100000000;
    const double x = 3, y = 3;

    std::vector<std::unique_ptr<ShapeInterface>> shapes_polymorphic;
    std::vector<Shape> shapes_variant;
    std::vector<ReuseShape> shapes_reuse;
    std::vector<EnumShape> shapes_enum;

    for(int i = 0; i < 5; i++) {
        shapes_polymorphic.push_back(std::make_unique<RectanglePolymorphic>(i, i, 1, 1));
        shapes_variant.emplace_back(Rectangle(i, i, 1, 1));
        shapes_reuse.emplace_back(RectanglePolymorphic(i, i, 1, 1));
        shapes_enum.push_back({ShapeType::Rectangle, static_cast<double>(i), static_cast<double>(i), 1, 1});
    }
    for(int i = 0; i < 5; i++) {
        shapes_polymorphic.push_back(std::make_unique<SpherePolymorphic>(i, i, 1));
        shapes_variant.emplace_back(Sphere(i, i, 1));
        shapes_reuse.emplace_back(SpherePolymorphic(i, i, 1));
        shapes_enum.push_back({ShapeType::Sphere, static_cast<double>(i), static_cast<double>(i), 1, 1});
    }

    benchmark_polymorphic(shapes_polymorphic, x, y, n_bench);
    benchmark_casted(shapes_polymorphic, x, y, n_bench);
    benchmark_variant(shapes_variant, x, y, n_bench);
    benchmark_variant_get_if(shapes_variant, x, y, n_bench);
    benchmark_reuse_variant_get_if(shapes_reuse, x, y, n_bench);
    benchmark_enum_switch(shapes_enum, x, y, n_bench);
    return 0;
}
