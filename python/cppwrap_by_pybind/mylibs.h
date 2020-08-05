using namespace::std;

int add(int x, int y);

vector<int> vec_double(vector<int> &v);

vector<vector<int>> vec_add(vector<vector<int>> &vec);

class POINT {
private:
    int x;
    int y;
public:
    int sum;

    POINT(int x, int y) { this->x = x; this->y = y; this->sum = x+y; }
    int X() { return x; }
    int Y() { return y; }
};

POINT move_p(POINT p, int d);
