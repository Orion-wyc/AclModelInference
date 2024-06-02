export APP_SOURCE_PATH=<SAMPLE_DIR>/MyFirstApp_ONNX
export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=$HOME/Ascend/ascend-toolkit/latest/<arch-os>/devlib

./sample_build.sh
./sample_run.sh