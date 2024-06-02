#include <cstdlib>

#include "acl/acl.h"
#include "model_handle.h"
#include "log.h"

aclError TestStaticModel(aclapp::ModelHandle &modelHandle) {
  // 1.定义一个资源初始化的函数，用于AscendCL初始化、运行管理资源申请（指定计算设备）
  APP_CHK_STATUS(modelHandle.InitResource());

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

aclError ReleaseResource(aclapp::ModelHandle &modelHandle) {
  // 6.定义一个模型卸载的函数，卸载图片分类的模型
  APP_CHK_STATUS(modelHandle.UnloadModel());

  // 7.定义一个函数，用于释放内存、销毁推理相关的数据类型，防止内存泄露
  APP_CHK_STATUS(modelHandle.UnloadPicture());

  // 8.定义一个资源去初始化的函数，用于AscendCL去初始化、运行管理资源释放（释放计算设备）
  APP_CHK_STATUS(modelHandle.DestroyResource());
  return ACL_SUCCESS;
}

aclError TestDynamicModel(aclapp::ModelHandle &modelHandle) {
  APP_CHK_STATUS(modelHandle.InitResource());
  const char *modelPath = "../model/resnet50_dynamic.om";
  APP_CHK_STATUS(modelHandle.LoadModel(modelPath));
  const char *picturePath = "../data/dog1_1024_683.bin";
  APP_CHK_STATUS(modelHandle.LoadPicture(picturePath));
  // 动态Shape模型在执行前手动分配一块较大的内存
  modelHandle.SetDynamicOutputSize(1024);
  APP_CHK_STATUS(modelHandle.InferenceDynamic());
  APP_CHK_STATUS(modelHandle.PrintResult());
  return ACL_SUCCESS;
}

void TestMultipleModelsShareStream(bool isDynamic) {
  if (!isDynamic) {
    LOG_INFO("Begin to test static model.");
    aclapp::ModelHandle modelHandle1(nullptr);
    TestStaticModel(modelHandle1);
    ReleaseResource(modelHandle1);
  } else {
    LOG_INFO("Begin to test dynamic model.");
    aclrtStream stream = nullptr;
    (void)aclrtCreateStream(&stream);
    aclapp::ModelHandle modelHandle1(stream);
    (void)TestDynamicModel(modelHandle1);
    ReleaseResource(modelHandle1);
  }
}

bool TestDynamicShape(const std::string &envVarName) {
  const char *envVarValue = std::getenv(envVarName.c_str());
  if (envVarValue == nullptr) {
    return false;
  }
  std::string value(envVarValue);

  return (value != "0");
}

int main() {
  if (TestDynamicShape("ENABLE_RUNTIME_V2")) {
    // 测试动态Shape模型推理
    TestMultipleModelsShareStream(true);
  } else {
    // 测试静态Shape模型推理
    TestMultipleModelsShareStream(false);
  }

  return 0;
}