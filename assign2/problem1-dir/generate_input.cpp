#include<bits/stdc++.h>
using namespace std;

string word(){
    int sz = rand()%8 + 1;
    string w = "";
    for(int i=0;i<sz;i++){
        w += (rand()%26 + 97);
    }
    return w;
}

int main(int argc, char *argv[])
{
    ofstream fout("./test1/" + string(argv[1]));
    int lines = rand()%100 + 5000;
    for(int l =0 ;l<lines;l++){
        int words = rand()%5 + 8;
        for(int i=0;i<words;i++){
            fout << word();
            if(i == words-1)
                fout << '\n';
            else   
                fout << ' ';
        }
    }
    fout.close();
}