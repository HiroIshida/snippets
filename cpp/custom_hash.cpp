// https://qiita.com/izmktr/items/8e0fd1b6e37de59a9bd0
#include<tuple>
#include<unordered_map>

template<typename T>
inline void hash_combine(std::size_t& seed, const T& val)
{
    std::hash<T> hasher;
    seed ^= hasher(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

class Hoge
{
    public:
        int aho;
        std::string unko;
};

namespace std{
    // example for std::tuple 
    // hash is std's type. By usign template specilization, we can add a new hashable type here!
    template<typename S, typename T>
    struct hash<std::pair<S, T>>
    {
        inline size_t operator()(const std::pair<S, T>& val) const
        {
            size_t seed = 0;
            hash_combine(seed, val.first);
            hash_combine(seed, val.second);
            return seed;
        }
    };

    template<>
    struct hash<Hoge>
    {
        inline size_t operator()(const Hoge& val) const
        {
            size_t seed = 0;
            std::hash<int> int_hasher;
            std::hash<std::string> str_hasher;
            seed ^= int_hasher(val.aho) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= str_hasher(val.unko) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
}


int main(){
    std::unordered_map<std::pair<int, int>, int> hoge;
    std::unordered_map<Hoge, int> heck;
}
