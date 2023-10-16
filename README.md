# IDP-system

`IDP-system`是一套基于人工智能技术对文档进行自动分析、理解的智能文档处理（Intelligent Document Processing, IDP）系统，通过结合计算机视觉、自然语言处理以及文档指纹等技术，`IDP-system`可以实现版面分析、文字提取、标题提取、表格识别、信息抽取、文档分级分类和文档相似度计算等多种功能，从而可以快速、准确地处理大量的文档数据，并进一步实现关键数据识别、防泄漏等工作。

目前支持pdf、docx以及各类图片格式的文档类型。

## 目录

- [文件目录](#文件目录)
- [环境安装](#环境安装)
- [使用的模型](#使用的模型)
- [接口说明](#接口说明)
  - [文件读取接口](#文件读取接口)
  	- [read_file](#read_file)
  - [领域分类接口](#领域分类接口)
  	- [get_domain_type](#get_domain_type)
  	- [creat_index](#creat_index)
  - [信息抽取接口](#信息抽取接口)
	  - [Info_Extraction](#info_extraction)
	  - [sensitive_word](#sensitive_word)
	  - [extraction_classify](#extraction_classify)
  - [文档指纹接口](#文档指纹接口)
	  - [get_file_finger](#get_file_finger)
	  - [file_file_check](#file_file_check)
	  - [check_file_finger](#check_file_finger)
	  - [write_pickle](#write_pickle)
	  - [read_pickle](#read_pickle)

## 文件目录

```
│  api.py  # 接口代码
│  utils.py  # 基础函数库
│  update_keywords_table.py  # 操作领域词库的基础函数库
│  requirements.txt  # 环境包需求
│
│  integration_optimize.py  # 文档敏感识别全流程的整合代码（不包含指纹抽取）
│  Finger_integration_optimize.py  # 面向文档指纹抽取的多线程提取接口
│
│  doc_classification_exp.py  # 文档分类效果实验代码
│  uie_test.py  # uie模型效果实验代码
│
├─configuration  # 配置文件
│  │  100000-small-modi.txt  # 腾讯词向量，10万词（删除了空格的向量）
│  │  base.log  # 领域词基本信息
│  │  domain_keywords.txt  # 各领域关键词表
│  │  intersection.txt  # 领域词交集表
│  └─ocr  # PaddleOCR和PPStructure模型文件
│                  
└─label # uie模型微调代码
```

## 环境安装

**系统要求：**
- Windows/Linux
- python=3.8
- CUDA=10.2

**环境安装：**

```python
pip install -r requirements.txt
```

注：尽量优先安装如下的包，再运行上述requirements：

1. gpu环境下安装的包（不兼容cpu）：`pip install paddlepaddle-gpu==2.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple`
   
	非AVX指令集安装：
	```
	手动下载
	paddlepaddle_gpu-2.4.2-cp38-cp38-win_amd64.whl：https://www.paddlepaddle.org.cn/whl/windows/mkl/noavx/stable.html
	pip install paddlepaddle_gpu-2.4.2-cp38-cp38-win_amd64.whl.whl
	```	
2. cpu环境下安装的包：`pip install paddlepaddle==2.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple`
3. `pip install paddleocr==2.7.0.2`
4. `pip install paddlenlp==2.5.2`

## 使用的模型

- ocr：PaddleOCR-v4
- 版面识别：PPStructure -> picodet_lcnet_x1_0_fgd_layout_cdla_infer
- 表格识别：PPStructure -> ch_ppstructure_mobile_v2.0_SLANet_infer
- 信息抽取：UIE -> uie-m-base
- 文档哈希指纹：N-gram + LSH
- 文档指纹相似度：汉明距离

## 接口说明

### 文件读取接口

 #### `read_file`
 
读取pdf和docx文件的信息，提取出文件的文本信息和标题

***接口输入参数***

|名称|类型|可选/必选|示例值|描述|
| :-: | :-: | :-: | :-: | :-: |
|file\_path|String|必选|"C:\test\1.pdf"|要提取文件的绝对路径或相对路径|
|ocr||必选||传入已经加载好的paddleOcr模型。|
|table\_engine||必选||传入已经加载好的PPStructure模型|
|table\_extract|Bool|可选|True|是否提取文件中的表格到excel中，默认为False|
|table\_dir|String|可选|"C:\test\excel"|如果要提取文件的excel，给出路径（显贵路径或绝对路径）|

***接口输出格式***

|名称|类型|示例值|描述|
| :-: | :-: | :-: | :-: |
|ori\_filename|String|"123.pdf"|文件本身文件名|
|filename|String|"保险单"|提取出的文件名|
|content|String|"这是一张保险单xxxxx"|从文件中提取出的文本信息|

***接口调用示例***

```python
## 模型加载(模型已经提前下载到项目中，可以离线运行)
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch",
						  det_model_dir="../configuration/ocr/ch_PP-OCRv4_det_infer",
						  rec_model_dir="../configuration/ocr/ch_PP-OCRv4_rec_infer",
						  cls_model_dir="../configuration/ocr/ch_ppocr_mobile_v2.0_cls_infer")  # 初始化ocr模型
table_engine = PPStructure(table=False, ocr=False, show_log=True,
						  layout_model_dir="../configuration/ocr/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer",
						  table_model_dir="../configuration/ocr/table/ch_ppstructure_mobile_v2.0_SLANet_infer",
						  det_model_dir="../configuration/ocr/ch_PP-OCRv4_det_infer",
						  rec_model_dir="../configuration/ocr/ch_PP-OCRv4_rec_infer")  # 初始化版面识别模型
read_file("./test/1.pdf",ocr,table_engine,table_extract=True,table_dir="./test/excel")  # 文件读入
```

### 领域分类接口

#### `get_domain_type`

获取某标题的领域类型

***接口输入参数***

|名称|类型|可选/必选|示例值|描述|
| :-: | :-: | :-: | :-: | :-: |
|text|String|必选|"2023年度报告"|文件名/标题|
|embeddings||必选||词嵌入表|
|domain\_keywords||必选||领域关键词表|
|intersection||必选||词交集表|
|index||必选||faiss创建的索引|
|key\_d|dict|必选|{‘A’:[1,2,3]}|用于确定领域词id位置的字典|

***接口输出格式***

|名称|类型|示例值|描述|
| :-: | :-: | :-: | :-: |
|domain\_pred|String|'A'|领域预测结果|
|siminum/match\_num|Float||平均预测相似度|
|words|set||text 分词结果|

***接口调用示例***

```python
keyword = get_domain_keywords(keywords_path)  # 读入领域词表
index, key_d = creat_index(keyword)  # 创建领域词表索引
domain_pred, simi, words = get_domain_type(text,embeddings,keywords,intersection,index,key_d)  # 预测领域分类结果
```

#### `creat_index`

基于领域关键词表生成faiss索引

***接口输入参数***

|名称|类型|可选/必选|示例值|描述|
| :-: | :-: | :-: | :-: | :-: |
|domain\_keywords||必选||领域关键词表|

***接口输出格式***

|名称|类型|示例值|描述|
| :-: | :-: | :-: | :-: |
|index|||faiss创建的索引|
|key\_d|dict|{‘A’:[1,2,3]}|用于确定领域词id位置的字典|

***接口调用示例***
```python
keyword = get_domain_keywords(keywords_path)  # 读入领域词表
index, key_d = creat_index(keyword)  #初始化faiss索引
```

### 信息抽取接口

#### `Info_Extraction`

类初始化加载模型，直接调用进行抽取对应领域的schema

***类初始化***

|名称|类型|可选/必选|示例值|描述|
| :-: | :-: | :-: | :-: | :-: |
|schemas\_dict|dict|可选（默认值来自utils.py中schemas\_dict\_education）|{'人员域': {'学生-基本信息': ['姓名'] },'财务域': {'一卡通消费信息': ['钱包交易金额','钱包余额'],    },}|设定要抽取的信息对应schema，来自于文件类型表，整理成字典格式。|
|model|String|可选（默认值'uie-m-base'）|'uie-base'|Paddlenlp官方模型名称|
|task\_path|String|可选（默认值None）|'./checkpoint/model\_best'|微调过的模型路径，使用此项则model参数忽略|
|use\_fast|bool|可选（默认值True）|True|使用FastTokenizer库加速(需要安装fast-tokenizer-python包)|

***接口输入参数***

|名称|类型|可选/必选|示例值|描述|
| :-: | :-: | :-: | :-: | :-: |
|text\_str|String|必选|'我叫小明，学号12345'|文档识别出来的纯本文字符串|
|field|String|必选|'人员域'|文档所属领域，来自文件分类表格，手动指定（领域识别）|
|print\_info|bool|可选（默认值False）|False|运行时是否打印信息，不影响返回结果|

***接口输出格式***

|名称|类型|示例值|描述|
| :-: | :-: | :-: | :-: |
|extract\_result|dict|{'姓名':[ '小明'], '学号':[ '12345']}|抽取信息后的schema键值对结果|
|mg|bool|False|文档是否是敏感内容|

***接口调用示例***

```python
# 类初始化，加载模型
uie = Info_Extraction(schemas_dict=schemas_dict_education, model='uie-m-base')

# 调用实例（__call__方法）
field1 = '人员域'
text1 = '我叫小明，学号12345'
extract_result1, mg1 = uie(text1, field1)

field2 = '财务域'
text2 = '发票金额1000元, 共1张'
extract_result2, mg2 = uie(text2, field2)

# 类初始化，加载其他模型为另一个实例
uie2 = Info_Extraction(schemas_dict=schemas_dict_education, model='uie-base', task_path='./checkpoint/model_best')

# 调用新的模型实例进行抽取
extract_result3, mg3 = uie2(text1, field1)
extract_result4, mg4 = uie2(text2, field2)
```

#### `sensitive_word`

关键词匹配，通过关键词判断是否敏感

***接口输入参数***

|名称|类型|可选/必选|示例值|描述|
| :-: | :-: | :-: | :-: | :-: |
|file\_txt|String|必选|"此件不公开"|文档识别出来的纯本文字符串|
|print\_info|bool|可选（默认值False）|True|运行时是否打印信息，不影响返回结果|

***接口输出格式***

|名称|类型|示例值|描述|
| :-: | :-: | :-: | :-: |
|have\_sensitive\_words|bool|True|是否包含敏感关键词（来自于utils.py），即是否敏感|

***接口调用示例***

```python
text = '我叫小明，学号12345'
have_sensitive_words = sensitive_word(text)
```

#### `extraction_classify`

根据抽取结果进行文件分类，分类到文件分类表中的D列

***接口输入参数***

|名称|类型|可选/必选|示例值|描述|
| :-: | :-: | :-: | :-: | :-: |
|field|String|必选|'人员域' |文档所属领域，来自文件分类表格，手动指定（领域识别）|
|extract\_result|dict|必选|{'姓名':[ '小明'], '学号':[ '12345']}|抽取信息的schema键值对，来自Info\_Extraction调用结果|
|schema\_dict|dict|可选（默认值来自于utils.py中schemas\_dict\_education\_D）|{'人员域': {'学生身份信息': ['姓名','学号'],}}|分类所用字典，需要根据文件分类表中领域词schema设置，将schema集成到{领域:{D列:[schema1, schema2]}}，需要手动根据分类表设置|
|print\_info|bool|可选（默认值False）|False|运行时是否打印信息，不影响返回结果|

***接口输出格式***

|名称|类型|示例值|描述|
| :-: | :-: | :-: | :-: |
|classify|list|['学生身份信息','学生校内信息']|根据抽取信息键值对，判断属于文件分类表中D列哪一项|

***接口调用示例***

```python
# 首先调用Info_Extraction进行抽取
text = '我叫小明，学号12345'
field = '人员域'
uie = Info_Extraction(schemas_dict=schemas_dict_education, model='uie-m-base')
extract_result, mg = uie(text, field)

# 对抽取结果进行判断
classify = extraction_classify(field, extract_result)
```

### 文档指纹接口

#### `get_file_finger`

计算指定路径下所有文件的文件指纹

***接口输入参数***

|名称|类型|可选/必选|示例值|描述|
| :-: | :-: | :-: | :-: | :-: |
|file\_dir|String|必选|"D/OCR/OCR\_test"|输入文件的路径|
|out\_dir|String|必选|“D/OCR/OCR\_test"|结果输出路径，可用于复合指纹输出路径输出指纹|

***接口输出格式***

|名称|类型|示例值|描述|
| :-: | :-: | :-: | :-: |
|||||

***接口调用示例***

```python
get_file_finger("./data", "./out")
```

#### `file_file_check`

直接对比两个文件的相似度

***接口输入参数***

|名称|类型|可选/必选|示例值|描述|
| :-: | :-: | :-: | :-: | :-: |
|file\_dir|String|必选|"D/OCR/OCR\_test"|输入文件的路径|

***接口输出格式***

|名称|类型|示例值|描述|
| :-: | :-: | :-: | :-: |
|ro|float|0\.891|两个文件的相似度|

***接口调用示例***

```python
file_file_finger("./data")
```

#### `check_file_finger`

对比文件是否为敏感文件

***接口输入参数***

|名称|类型|可选/必选|示例值|描述|
| :-: | :-: | :-: | :-: | :-: |
|file\_dir|String|必选|"D/OCR/OCR\_test"|输入文件的路径|
|check\_file\_dir|String|必选|“D/OCR/OCR\_test"|用于对比的文件指纹路径|
|out\_dir|String|必选|“D/OCR/OCR\_test"|用于测试的输出路径|

***接口输出格式***

|名称|类型|示例值|描述|
| :-: | :-: | :-: | :-: |
|ro|bool|True|是否为敏感文件|

***接口调用示例***

```python
check_file_finger("./file1", "./out/asd5861pick", out_dir)
```

#### `write_pickle`

保存文件指纹

***接口输入参数***

|名称|类型|可选/必选|示例值|描述|
| :-: | :-: | :-: | :-: | :-: |
|file\_directory|String|必选|"D/OCR/OCR\_test"|将文件指纹保存到何处|
|data|dict|必选|{'hash1': ans[0].hash1, 'hash2': ans[0].hash2, 'hash3': dict1}|经过整理的文件指纹关键信息，仅保留了计算出的hash值，源文件信息被隐藏|

***接口输出格式***

|名称|类型|示例值|描述|
| :-: | :-: | :-: | :-: |
|file\_path|String|"D/OCR/OCR\_test"|保存文件的最终路径|
|file\_name|String|”asdsdsdsds“|为该指纹随机生成的文件名，与原文件名无关|

***接口调用示例***

```python
data2 = {'hash1': ans[1].hash1, 'hash2': ans[1].hash2, 'hash3': dict2}
file_path_2, file_name_2 = write_pickle(fingerprint_file_directory, data2)
```

#### `read_pickle`

读取文件指纹

***接口输入参数***

|名称|类型|可选/必选|示例值|描述|
| :-: | :-: | :-: | :-: | :-: |
|file\_path|String|必选|"D/OCR/OCR\_test"|文件指纹路径|

***接口输出格式***

|名称|类型|示例值|描述|
| :-: | :-: | :-: | :-: |
|data|dict|{'hash1': ans[1].hash1, 'hash2': ans[1].hash2, 'hash3': dict2}|文件指纹内的hash值|

***接口调用示例***

```python
data2 = read_pickle(fingerprint_file_directory)
```

