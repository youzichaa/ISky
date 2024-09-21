#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <numeric>
#include <set>

namespace fs = std::filesystem;

std::string data_folderT = "data/data";
const std::vector<std::string> input_size = {"1000/", "3000/", "5000/", "7000/", "9000/", "11000/"};
const std::vector<std::string> input_files = {"data_anti_corr.txt", "data_correlate.txt", "data_equally.txt"};
const std::vector<std::string> output_files = {"small_anti_corr.txt", "small_correlate.txt", "small_uniformly.txt"};
// const std::vector<std::string> output_files = {"hotel.txt"};
// const std::vector<std::string> output_files = {"nba.txt"};
// const std::vector<std::string> output_files = {"small_anti_corr.txt", "small_correlate.txt", "small_uniformly.txt", "nba.txt", "hotel.txt"};

int domain = 10000;

void data_int()
{
    for (int lk = 2; lk < 11; ++lk)
    {
        std::string data_folderTT = data_folderT + std::to_string(lk);
        for (size_t ln = 0; ln < input_size.size(); ++ln)
        {
            std::string data_folder = data_folderTT + "/size=" + input_size[ln];
            for (size_t i = 0; i < input_files.size(); ++i)
            {
                std::ifstream infile(data_folder + input_files[i]);
                std::cout << data_folder + input_files[i] << std::endl;
                std::ofstream outfile(data_folder + output_files[i]);

                int n, m;
                infile >> n >> m;

                std::vector<std::vector<int>> data(n, std::vector<int>(m));

                for (int i = 0; i < n; ++i)
                {
                    for (int j = 0; j < m; ++j)
                    {
                        float value;
                        infile >> value;
                        data[i][j] = static_cast<int>(value * domain);
                        if (data[i][j] < 1)
                        {
                            data[i][j] = 1;
                        }
                    }
                }
                // Check for duplicate rows and offset them
                std::set<std::vector<int>> uniqueRows;
                for (int i = 0; i < n; ++i)
                {
                    if (uniqueRows.find(data[i]) != uniqueRows.end())
                    {
                        int offset = (std::rand() % 2 > 0) ? 1 : -1;
                        // Function to offset a row of data
                        int dimension = std::rand() % m;
                        data[i][dimension] += offset;
                        if ((data[i][dimension] >= domain) || (data[i][dimension] <= 0))
                        {
                            data[i][dimension] -= 2 * offset;
                        }
                        while (uniqueRows.find(data[i]) != uniqueRows.end())
                        {
                            offset = (std::rand() % 2 > 0) ? 1 : -1;
                            dimension = std::rand() % m;
                            data[i][dimension] += offset;
                            if ((data[i][dimension] >= domain) || (data[i][dimension] <= 0))
                            {
                                data[i][dimension] -= 2 * offset;
                            }
                        }
                    }
                    uniqueRows.insert(data[i]);
                }

                // Write the processed data to the output file
                outfile << n << " " << m << std::endl;
                for (const auto &row : data)
                {
                    for (size_t j = 0; j < row.size(); ++j)
                    {
                        outfile << row[j];
                        if (j < row.size() - 1)
                        {
                            outfile << ",";
                        }
                    }
                    outfile << std::endl;
                }

                infile.close();
                outfile.close();
            }
        }
    }
}

void data_incomplete()
{
    const std::vector<int> rate = {10, 20, 30, 40, 50};

    for (int lk = 6; lk < 7; ++lk)
    {
        std::string data_folderTT = data_folderT + std::to_string(lk);
        // if data is hotel, size is 1000
        for (size_t ln = 0; ln < input_size.size(); ++ln)
        {
            std::string data_folder = data_folderTT + "/size=" + input_size[ln];
            for (size_t i = 0; i < output_files.size(); ++i)
            {
                if (i == 4)
                {
                    ln += input_size.size();
                }
                std::ifstream infile(data_folder + output_files[i]);
                std::cout << data_folder + output_files[i] << std::endl;
                int n, m;
                infile >> n >> m;
                std::vector<std::vector<int>> data(n, std::vector<int>(m));
                std::string line;
                std::getline(infile, line);
                int row = 0;
                while (std::getline(infile, line) && row < n)
                {
                    std::istringstream iss(line);
                    std::string cell;
                    int col = 0;
                    while (std::getline(iss, cell, ',') && col < m)
                    {
                        int value;
                        std::istringstream(cell) >> value;
                        data[row][col] = value;
                        ++col;
                    }
                    ++row;
                }
                for (int ll = 0; ll < rate.size(); ll++)
                {
                    int selectNum = n * m * rate[ll] / 100;
                    std::vector<std::vector<int>> zeromatrix(n, std::vector<int>(m, 0));
                    while (selectNum > 0)
                    {
                        int i = rand() % n;
                        int j = rand() % m;
                        if (zeromatrix[i][j] == 1)
                        {
                            continue;
                        }
                        zeromatrix[i][j] = 1;
                        if (std::accumulate(zeromatrix[i].begin(), zeromatrix[i].end(), 0) == m)
                        {
                            zeromatrix[i][j] = 0;
                            continue;
                        }
                        selectNum--;
                    }
                    // for (const auto &row : data)
                    // {
                    //     for (int val : row)
                    //     {
                    //         std::cout << val << " ";
                    //     }
                    //     std::cout << std::endl;
                    // }
                    fs::path directoryPath = data_folder + "rate=" + std::to_string(rate[ll]);
                    fs::path filePath = directoryPath / output_files[i];
                    std::cout << filePath << std::endl;
                    // Write the processed data to the output file
                    if (!fs::exists(directoryPath))
                    {
                        fs::create_directories(directoryPath);
                    }
                    std::ofstream outfile(filePath);
                    outfile << n << " " << m << std::endl;
                    int index = 0;
                    for (int i = 0; i < n; ++i)
                    {
                        for (int j = 0; j < m; ++j)
                        {
                            if (zeromatrix[i][j] == 1)
                            {
                                outfile << 0;
                            }
                            else
                            {
                                outfile << data[i][j];
                            }
                            if (j < m - 1)
                            {
                                outfile << ",";
                            }
                            index++;
                        }
                        outfile << std::endl;
                    }
                    infile.close();
                    outfile.close();
                }
            }
        }
    }
}

// Main function to read, process, and write files
int main()
{
    std::srand(std::time(0)); // Seed the random number generator
    // data_int();
    data_incomplete();

    return 0;
}