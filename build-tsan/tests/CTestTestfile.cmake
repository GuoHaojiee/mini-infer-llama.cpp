# CMake generated Testfile for 
# Source directory: /home/guohaojie/Guo/mini-infer/tests
# Build directory: /home/guohaojie/Guo/mini-infer/build-tsan/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tensor "/usr/bin/setarch" "x86_64" "-R" "/home/guohaojie/Guo/mini-infer/build-tsan/tests/test_tensor")
set_tests_properties(tensor PROPERTIES  _BACKTRACE_TRIPLES "/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;32;add_test;/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;0;")
add_test(context "/usr/bin/setarch" "x86_64" "-R" "/home/guohaojie/Guo/mini-infer/build-tsan/tests/test_context")
set_tests_properties(context PROPERTIES  _BACKTRACE_TRIPLES "/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;32;add_test;/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;0;")
add_test(gguf "/usr/bin/setarch" "x86_64" "-R" "/home/guohaojie/Guo/mini-infer/build-tsan/tests/test_gguf")
set_tests_properties(gguf PROPERTIES  _BACKTRACE_TRIPLES "/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;32;add_test;/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;0;")
add_test(graph "/usr/bin/setarch" "x86_64" "-R" "/home/guohaojie/Guo/mini-infer/build-tsan/tests/test_graph")
set_tests_properties(graph PROPERTIES  _BACKTRACE_TRIPLES "/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;32;add_test;/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;0;")
add_test(gallocr "/usr/bin/setarch" "x86_64" "-R" "/home/guohaojie/Guo/mini-infer/build-tsan/tests/test_gallocr")
set_tests_properties(gallocr PROPERTIES  _BACKTRACE_TRIPLES "/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;32;add_test;/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;0;")
add_test(ops_cpu "/usr/bin/setarch" "x86_64" "-R" "/home/guohaojie/Guo/mini-infer/build-tsan/tests/test_ops_cpu")
set_tests_properties(ops_cpu PROPERTIES  _BACKTRACE_TRIPLES "/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;32;add_test;/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;0;")
add_test(threadpool "/usr/bin/setarch" "x86_64" "-R" "/home/guohaojie/Guo/mini-infer/build-tsan/tests/test_threadpool")
set_tests_properties(threadpool PROPERTIES  _BACKTRACE_TRIPLES "/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;32;add_test;/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;0;")
add_test(quant "/usr/bin/setarch" "x86_64" "-R" "/home/guohaojie/Guo/mini-infer/build-tsan/tests/test_quant")
set_tests_properties(quant PROPERTIES  _BACKTRACE_TRIPLES "/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;32;add_test;/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;0;")
add_test(model_qwen2 "/usr/bin/setarch" "x86_64" "-R" "/home/guohaojie/Guo/mini-infer/build-tsan/tests/test_model_qwen2")
set_tests_properties(model_qwen2 PROPERTIES  _BACKTRACE_TRIPLES "/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;32;add_test;/home/guohaojie/Guo/mini-infer/tests/CMakeLists.txt;0;")
