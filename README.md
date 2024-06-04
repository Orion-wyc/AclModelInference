## 准备数据

### 下载或导出 ONNX 模型

- 下载[Download From huggingface](https://huggingface.co/OWG/resnet-50/blob/main/onnx/model.onnx)

- 手动导出 
  - 导出脚本 [export_onnx.py](script/export_onnx.py)
  - [权重文件](https://download.pytorch.org/models/resnet50-0676ba61.pth)

### ATC 工具转 om 模型

```bash
atc --model=./model/resnet50_XXX.onnx --framework=5 --output=./model/resnet50_dynamic --input_shape="input:1~128,3,224,224" --soc_version=<soc_version>  
```