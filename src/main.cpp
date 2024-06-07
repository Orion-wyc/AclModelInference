#include <cstdlib>

#include "acl/acl.h"
#include "acl/acl_rt_allocator.h"
#include "model_handle.h"
#include "log.h"
#include "allocator.h"

// 1.定义一个资源初始化的函数，用于AscendCL初始化、运行管理资源申请（指定计算设备）
aclError InitResource(int32_t deviceId) {
  APP_CHK_STATUS(aclInit(nullptr));
  APP_CHK_STATUS(aclrtSetDevice(deviceId));
  return ACL_SUCCESS;
}

// 8.定义一个资源去初始化的函数，用于AscendCL去初始化、运行管理资源释放（释放计算设备）
aclError DestroyResource(int32_t deviceId) {
  APP_CHK_STATUS(aclrtResetDevice(deviceId));
  APP_CHK_STATUS(aclFinalize());
  return ACL_SUCCESS;
}

aclError TestStaticModel(aclapp::ModelHandle &modelHandle) {
  // 2.定义一个模型加载的函数，加载图片分类的模型，用于后续推理使用
  const char *modelPath = "../model/resnet50.om";
  APP_CHK_STATUS(modelHandle.LoadModel(modelPath));

  // 3.定义一个读图片数据的函数，将测试图片数据读入内存，并传输到Device侧，用于后续推理使用
  const char *picturePath = "../data/dog1_1024_683.bin";
  APP_CHK_STATUS(modelHandle.LoadPicture(picturePath));

  // 4.定义一个推理的函数，用于执行推理
  APP_CHK_STATUS(modelHandle.Inference());

  // 5.定义一个推理结果数据处理的函数，用于在终端上屏显测试图片的top5置信度的类别编号
  APP_CHK_STATUS(modelHandle.PrintResult());

  return ACL_SUCCESS;
}

aclError ReleaseModelResource(aclapp::ModelHandle &modelHandle) {
  // 6.定义一个模型卸载的函数，卸载图片分类的模型
  APP_CHK_STATUS(modelHandle.UnloadModel());

  // 7.定义一个函数，用于释放内存、销毁推理相关的数据类型，防止内存泄露
  APP_CHK_STATUS(modelHandle.UnloadPicture());

  return ACL_SUCCESS;
}

aclError TestDynamicModel(aclapp::ModelHandle &modelHandle) {
  APP_CHK_STATUS(modelHandle.InitResource());
  const char *modelPath = "../model/resnet50_dynamic.om";
  APP_CHK_STATUS(modelHandle.LoadModel(modelPath));
  const char *picturePath = "../data/dog1_1024_683.bin";
  APP_CHK_STATUS(modelHandle.LoadPicture(picturePath));
  // 动态Shape模型在执行前手动分配一块较大的内存
  modelHandle.SetDynamicOutputSize(4096);
  APP_CHK_STATUS(modelHandle.InferenceDynamic());
  APP_CHK_STATUS(modelHandle.PrintResult());
  return ACL_SUCCESS;
}

void *RawMalloc(aclrtAllocator allocator, size_t size) {
  return static_cast<mds::Allocator *>(allocator)->Malloc(size);
}

void RawFree(aclrtAllocator allocator, void *block) {
  return static_cast<mds::Allocator *>(allocator)->Free(block);
}

void *RawMallocAdvise(aclrtAllocator allocator, size_t size, aclrtAllocatorAddr addr) {
  return static_cast<mds::Allocator *>(allocator)->MallocAdvise(size, addr);
}

void *RawGetBlockAddr(aclrtAllocatorBlock block) {
  return block;
}

aclError RegisterCustomAllocator(aclrtAllocator allocator, aclrtStream stream) {
  // 创建AllocatorDesc
  aclrtAllocatorDesc allocatorDesc = aclrtAllocatorCreateDesc();

  // 初始化
  APP_CHK_STATUS(aclrtAllocatorSetObjToDesc(allocatorDesc, allocator));
  APP_CHK_STATUS(aclrtAllocatorSetAllocFuncToDesc(allocator, RawMalloc));
  APP_CHK_STATUS(aclrtAllocatorSetFreeFuncToDesc(allocator, RawFree));
  APP_CHK_STATUS(aclrtAllocatorSetAllocAdviseFuncToDesc(allocator, RawMallocAdvise));
  APP_CHK_STATUS(aclrtAllocatorSetGetAddrFromBlockFuncToDesc(allocator, RawGetBlockAddr));

  // 注册
  APP_CHK_STATUS(aclrtAllocatorRegister(stream, allocatorDesc));

  // 销毁AllocatorDesc
  APP_CHK_STATUS(aclrtAllocatorDestroyDesc(allocatorDesc));
  return ACL_SUCCESS;
}

aclError TestMultipleModelsShareStream(bool isDynamic) {
  if (!isDynamic) {
    LOG_INFO("Begin to test static model.");
    aclapp::ModelHandle modelHandle1(nullptr);
    TestStaticModel(modelHandle1);
    ReleaseModelResource(modelHandle1);
  } else {
    LOG_INFO("Begin to test dynamic model.");
    aclrtStream stream = nullptr;
    (void)aclrtCreateStream(&stream);
    APP_CHK_NOTNULL(stream);

    mds::Allocator allocator;
    RegisterCustomAllocator(&allocator, stream);

    aclapp::ModelHandle modelHandle1;
    modelHandle1.SetStream(stream);
    (void)TestDynamicModel(modelHandle1);

    APP_CHK_STATUS(aclrtAllocatorUnregister(stream));
    APP_CHK_STATUS(aclrtDestroyStream(stream));
    ReleaseModelResource(modelHandle1);
  }
  return ACL_SUCCESS;
}

bool EnableDynamicShape(const std::string &envVarName) {
  const char *envVarValue = std::getenv(envVarName.c_str());
  if (envVarValue == nullptr) {
    return false;
  }
  std::string value(envVarValue);

  return (value != "0");
}

int main() {
  int32_t deviceId = 0;
  InitResource(deviceId);

  if (EnableDynamicShape("ENABLE_RUNTIME_V2")) {
    // 测试动态Shape模型推理
    (void)TestMultipleModelsShareStream(true);
  } else {
    // 测试静态Shape模型推理
    (void)TestMultipleModelsShareStream(false);
  }

  DestroyResource(deviceId);

  return 0;
}