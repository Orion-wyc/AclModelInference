#include "model_handle.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <map>
#include <math.h>

#include "log.h"

namespace aclapp {
aclError ModelHandle::InitResource() {
  return ACL_SUCCESS;
}

aclError ModelHandle::LoadModel(const char *modelPath) {
  return aclmdlLoadFromFile(modelPath, &modelId_);
}

aclError ModelHandle::LoadPicture(const char *picturePath) {
  APP_CHK_STATUS(ReadPictureTotHost(picturePath));
  return CopyDataFromHostToDevice();
}

// 以同步接口执行静态模型
aclError ModelHandle::Inference() {
  APP_CHK_STATUS(CreateModelInput());
  APP_CHK_STATUS(CreateModelOutput());
  APP_CHK_STATUS(aclmdlExecute(modelId_, inputDataSet_, outputDataSet_));
  return ACL_SUCCESS;
}

// 以一步接口执行动态Shape模型
aclError ModelHandle::InferenceDynamic() {
  APP_CHK_STATUS(CreateModelInput());
  APP_CHK_STATUS(CreateModelOutput(true));

  // 设置输入 shape
  int64_t inputShape[4] = {1, 3, 224, 224};
  aclTensorDesc *inputDesc = aclCreateTensorDesc(ACL_FLOAT, 4, inputShape, ACL_FORMAT_NCHW);
  APP_CHK_NOTNULL(inputDesc);
  APP_CHK_STATUS(aclmdlSetDatasetTensorDesc(inputDataSet_, inputDesc, 0));

  // 执行动态shape推理
  APP_CHK_STATUS(aclmdlExecuteAsync(modelId_, inputDataSet_, outputDataSet_, stream_));
  APP_CHK_STATUS(aclrtSynchronizeStreamWithTimeout(stream_, 3000));

  // 获取输出实际大小，此处刷新后会按照实际大小进行拷贝
  aclTensorDesc *outputDesc = aclmdlGetDatasetTensorDesc(outputDataSet_, 0);
  APP_CHK_NOTNULL(outputDesc);
  outputDataSize_ = aclGetTensorDescSize(outputDesc);
  LOG_INFO("output data size of outputDesc is %lu", outputDataSize_);

  return ACL_SUCCESS;
}

aclError ModelHandle::PrintResult() {
  // 获取推理结果数据
  APP_CHK_STATUS(aclrtMallocHost(&outputHostData_, outputDataSize_));
  APP_CHK_STATUS(
      aclrtMemcpy(outputHostData_, outputDataSize_, outputDeviceData_, outputDataSize_, ACL_MEMCPY_DEVICE_TO_HOST));
  // 将内存中的数据转换为float类型
  float *outFloatData = reinterpret_cast<float *>(outputHostData_);

  // 屏显测试图片的top5置信度的类别编号
  std::map<float, unsigned int, std::greater<float>> resultMap;
  for (unsigned int j = 0; j < outputDataSize_ / sizeof(float); ++j) {
    resultMap[*outFloatData] = j;
    outFloatData++;
  }

  // do data processing with softmax and print top 5 classes
  double totalValue = 0.0;
  for (auto it = resultMap.begin(); it != resultMap.end(); ++it) {
    totalValue += exp(it->first);
  }

  int cnt = 0;
  for (auto it = resultMap.begin(); it != resultMap.end(); ++it) {
    if (++cnt > 5) {
      break;
    }
    printf("top %d: index[%d] value[%lf] \n", cnt, it->second, exp(it->first) / totalValue);
  }
  return ACL_SUCCESS;
}

aclError ModelHandle::UnloadModel() {
  // 释放模型描述信息
  APP_CHK_STATUS(aclmdlDestroyDesc(modelDesc_));
  // 卸载模型
  APP_CHK_STATUS(aclmdlUnload(modelId_));
  return ACL_SUCCESS;
}

aclError ModelHandle::UnloadPicture() {
  APP_CHK_STATUS(aclrtFreeHost(pictureHostData_));
  pictureHostData_ = nullptr;
  APP_CHK_STATUS(aclrtFree(pictureDeviceData_));
  pictureDeviceData_ = nullptr;
  APP_CHK_STATUS(aclDestroyDataBuffer(inputDataBuffer_));
  inputDataBuffer_ = nullptr;
  APP_CHK_STATUS(aclmdlDestroyDataset(inputDataSet_));
  inputDataSet_ = nullptr;

  APP_CHK_STATUS(aclrtFreeHost(outputHostData_));
  outputHostData_ = nullptr;
  APP_CHK_STATUS(aclrtFree(outputDeviceData_));
  outputDeviceData_ = nullptr;
  APP_CHK_STATUS(aclDestroyDataBuffer(outputDataBuffer_));
  outputDataBuffer_ = nullptr;
  APP_CHK_STATUS(aclmdlDestroyDataset(outputDataSet_));
  outputDataSet_ = nullptr;
  return ACL_SUCCESS;
}

aclError ModelHandle::DestroyResource() {
  return ACL_SUCCESS;
}

// private member functions

//申请内存，使用C/C++标准库的函数将测试图片读入内存
aclError ModelHandle::ReadPictureTotHost(const char *picturePath) {
  std::string fileName = picturePath;
  std::ifstream binFile(fileName, std::ifstream::binary);
  binFile.seekg(0, binFile.end);
  pictureDataSize_ = binFile.tellg();
  binFile.seekg(0, binFile.beg);
  APP_CHK_STATUS(aclrtMallocHost(&pictureHostData_, pictureDataSize_));
  binFile.read((char *)pictureHostData_, pictureDataSize_);
  binFile.close();
  return ACL_SUCCESS;
}

//申请Device侧的内存，再以内存复制的方式将内存中的图片数据传输到Device
aclError ModelHandle::CopyDataFromHostToDevice() {
  APP_CHK_STATUS(aclrtMalloc(&pictureDeviceData_, pictureDataSize_, ACL_MEM_MALLOC_HUGE_FIRST));
  APP_CHK_STATUS(
      aclrtMemcpy(pictureDeviceData_, pictureDataSize_, pictureHostData_, pictureDataSize_, ACL_MEMCPY_HOST_TO_DEVICE));
  return ACL_SUCCESS;
}

// 准备模型推理的输入数据结构
aclError ModelHandle::CreateModelInput() {
  // 创建aclmdlDataset类型的数据，描述模型推理的输入
  inputDataSet_ = aclmdlCreateDataset();
  APP_CHK_NOTNULL(inputDataSet_);
  inputDataBuffer_ = aclCreateDataBuffer(pictureDeviceData_, pictureDataSize_);
  APP_CHK_NOTNULL(inputDataBuffer_);
  APP_CHK_STATUS(aclmdlAddDatasetBuffer(inputDataSet_, inputDataBuffer_));

  return ACL_SUCCESS;
}

// 准备模型推理的输出数据结构
aclError ModelHandle::CreateModelOutput(bool isDynamic) {
  // 创建模型描述信息
  modelDesc_ = aclmdlCreateDesc();
  APP_CHK_NOTNULL(modelDesc_);
  APP_CHK_STATUS(aclmdlGetDesc(modelDesc_, modelId_));
  // 创建aclmdlDataset类型的数据，描述模型推理的输出
  outputDataSet_ = aclmdlCreateDataset();
  APP_CHK_NOTNULL(outputDataSet_);
  // 获取模型输出数据需占用的内存大小，单位为Byte
  if (isDynamic) {
    outputDataSize_ = dynamicOutputSize_;
  } else {
    outputDataSize_ = aclmdlGetOutputSizeByIndex(modelDesc_, 0U);
  }
  // 申请输出内存
  APP_CHK_STATUS(aclrtMalloc(&outputDeviceData_, outputDataSize_, ACL_MEM_MALLOC_HUGE_FIRST));
  outputDataBuffer_ = aclCreateDataBuffer(outputDeviceData_, outputDataSize_);
  APP_CHK_NOTNULL(outputDataBuffer_);
  APP_CHK_STATUS(aclmdlAddDatasetBuffer(outputDataSet_, outputDataBuffer_));

  return ACL_SUCCESS;
}

}  // namespace aclapp