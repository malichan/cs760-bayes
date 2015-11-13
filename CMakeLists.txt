cmake_minimum_required(VERSION 2.8)
project(bayes)

set(CMAKE_CXX_FLAGS "-std=c++11")

add_executable(bayes bayes.cpp Feature.cpp Instance.cpp Dataset.cpp BayesNet.cpp)