# 关键点检测 trt模型部署
- tensorRT不支持eqal(=)操作
- 在关键点检测的时候，网络输出```1x1x112x112```。需要使用maxpool进行nms
```
x = torch.sigmoid(x)
h = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(x)
keep = 1. - torch.ceil(h - x)
# keep = (h == x).float()  # 保留下极大值点
x = h * keep
```
但是```keep = (h == x).float() 中 = ```无法转到为tensorRT算子。因此需要用```keep = 1. - torch.ceil(h - x)```代替。

后续一些后处理也可以作为一个算子写入模型中。


## pytorch转ONNX
```
python torch2onnx.py --weight model.pth --f model.onnx --opset_version 12 --simplify --dynamic
```

## ONNX转trt
- 法1：
```
"""
#生成静态batchsize的engine
./trtexec 	
            --onnx=<onnx_file> \ 						
            --explicitBatch \ 						
            --saveEngine=<tensorRT_engine_file> \ 		
            --workspace=<size_in_megabytes> \ 		
            --fp16 		
                                            
#生成动态batchsize的engine
./trtexec 	
            --onnx=<onnx_file> \					
            --minShapes=input:<shape_of_min_batch> \ 
            --optShapes=input:<shape_of_opt_batch> \  	
            --maxShapes=input:<shape_of_max_batch> \ 	
            --workspace=<size_in_megabytes> \ 			
            --saveEngine=<engine_file> \   				
            --fp16   	
            
trtexec --onnx=xx.onnx --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 --maxShapes=input:8x3x224x224 --saveEngine=xx.engine --fp16
"""
```

- 法2
```
python  onnx2trt.py 
```

## 模型推理
- pytorch模型推理
```
python inference_pytorch.py --weight model.pth --data_source 000000.jpg --cal_fps
```

- onnx模型推理
```
python inference_onnx.py --onnx_file model.onnx --data_source 000000.jpg --cal_fps
```

- trt python api模型推理
```
python inference_trt.py --engine model.engine --data_source 000000.jpg --input_names input --output_names output cal_fps
```