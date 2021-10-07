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
    // 空のファイル作成
    FILE *fp;
    const string file_path = "./key.dat";
    fp = fopen(file_path.c_str(), "w");
    fclose(fp);

    // IPC keyの取得
    const int id = 51;
    const key_t key = ftok(file_path.c_str(), id);
    if(key == -1){
        cerr << "Failed to acquire key" << endl; 
        return EXIT_FAILURE;  
    }

    // 共有メモリIDの取得
    const int size = 0x6400;
    const int seg_id = shmget(key, size, 
                              IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR);
    if(seg_id == -1){
        cerr << "Failed to acquire segment" << endl;
        return EXIT_FAILURE;
    }

    // 共有メモリをプロセスにアタッチする
    char* const shared_memory = reinterpret_cast<char*>(shmat(seg_id, 0, 0));

    // 共有メモリに書き込む
    string s;
    int flag = 0;
    cout << "if you want to close, please type 'q'" << endl;
    while(flag == 0){
        cout << "word: ";
        cin >> s;
        if(s == "q") flag = 1;
        else sprintf(shared_memory, s.c_str());
    }

    // 共有メモリをプロセスから切り離す
    shmdt(shared_memory);

    // 共有メモリを解放する
    shmctl(seg_id, IPC_RMID, NULL);    

    return 0;
}
