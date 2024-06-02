#ifndef SRC_MODEL_HANDLE_H_
#define SRC_MODEL_HANDLE_H_
#include "acl/acl.h"

namespace aclapp {
class ModelHandle {
 public:
  ModelHandle(aclrtStream stream = nullptr) : stream_(stream) {}
  ~ModelHandle() = default;

  aclError InitResource();
  aclError DestroyResource();

  aclError LoadPicture(const char *picturePath);
  aclError UnloadPicture();

  aclError LoadModel(const char *modelPath);
  aclError UnloadModel();

  aclError Inference();
  aclError InferenceDynamic();

  void SetDynamicOutputSize(const size_t outSize) {
    dynamicOutputSize_ = outSize;
  }
  size_t GetDynamicOutputSize(const size_t outSize) const {
    return dynamicOutputSize_;
  }

  aclError PrintResult();

 private:
  aclError ReadPictureTotHost(const char *picturePath);
  aclError CopyDataFromHostToDevice();
  aclError CreateModelInput();
  aclError CreateModelOutput(bool isDynamic = false);

 private:
  aclrtStream stream_;

  uint32_t modelId_{0};
  int32_t deviceId_{0};
  size_t pictureDataSize_{0};
  // 静态Shape推理，输出内存在执行前是确定的
  size_t outputDataSize_{0};
  // 动态Shape推理，需要预先分配一块足够大的输出内存
  size_t dynamicOutputSize_{1024};

  void *pictureHostData_{nullptr};
  void *pictureDeviceData_{nullptr};
  void *outputDeviceData_{nullptr};
  void *outputHostData_{nullptr};
  aclmdlDataset *inputDataSet_{nullptr};
  aclDataBuffer *inputDataBuffer_{nullptr};
  aclmdlDataset *outputDataSet_{nullptr};
  aclDataBuffer *outputDataBuffer_{nullptr};
  aclmdlDesc *modelDesc_{nullptr};
};
}  // namespace aclapp

#endif  // SRC_MODEL_HANDLE_H_