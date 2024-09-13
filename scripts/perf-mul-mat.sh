#!/bin/bash

TEST_BACKEND_OPS_N_THREADS=1 ./bin/test-backend-ops perf -o MUL_MAT -b CPU > llamacpp_jetson_t1.log
TEST_BACKEND_OPS_N_THREADS=2 ./bin/test-backend-ops perf -o MUL_MAT -b CPU > llamacpp_jetson_t2.log
TEST_BACKEND_OPS_N_THREADS=4 ./bin/test-backend-ops perf -o MUL_MAT -b CPU > llamacpp_jetson_t4.log
TEST_BACKEND_OPS_N_THREADS=8 ./bin/test-backend-ops perf -o MUL_MAT -b CPU > llamacpp_jetson_t8.log
