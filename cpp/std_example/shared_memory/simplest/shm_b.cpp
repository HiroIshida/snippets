#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
using namespace std;


int main(){
    // 作成済みの共有メモリのIDを取得する
    const string file_path = "./key.dat";
    const int id = 51;

    const key_t key = ftok(file_path.c_str(), id);

    const int seg_id = shmget(key, 0, 0);
    if(seg_id == -1){
        cerr << "Failed to acquire segment" << endl;
        return EXIT_FAILURE;
    }

    // 共有メモリをプロセスにアタッチする
    char* const shared_memory = reinterpret_cast<char*>(shmat(seg_id, 0, 0));

    // 共有メモリの文字を読み取る
    int flag = 0;
    char c;
    cout << "If you want to close, please type 'q'" << endl;
    cout << "If you want to read the shared memory, push enter button." << endl;
    while(flag == 0){
        cin.get(c);
        if(c == 'q') flag = 1;
        else printf("%s\n", shared_memory);
    }

    // 共有メモリをプロセスから切り離す
    shmdt(shared_memory);

    return 0;
}
