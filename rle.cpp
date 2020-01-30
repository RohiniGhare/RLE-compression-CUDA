#include <iostream>
#include <string>

std::string RLE_CPU(std::string input) {

    if(input.length() == 0) return "";

    std::string result;

    int counter = 1;

    for(int i = 1; i < input.length(); i++) {
        if(input[i] != input[i+1]) {
            result += std::to_string(counter);
            result += input[i];
            counter = 1;
        } else {
            counter++;
        }
    }

    return result;
}

int main(int argc, char ** argv) {

    if(argc != 2) return -1;

    std::string s(argv[1]);

    std::string result = RLE_CPU(s);

    std::cout << result << std::endl;
    
    return 0;
}