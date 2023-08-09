# IPD-system

## 信息抽取

### Schema抽取

**`Info_Extraction`类**

类初始化

|参数名     |类型       |示例       |描述    |
|:---:      |:---:      |:---:      |:---:   |
|schemas_dict	|dict	|schemas_dict_education = {'人员域': {'学生-基本信息': ['姓名'] },'财务域': {'一卡通消费信息': ['钱包交易金额','钱包余额'],    },}	|设定要抽取的信息对应schema|
|model	|str	|‘uie-m-base’	|Paddlenlp官方模型名称|
|task_path	|str	|'./checkpoint/model_best'	|使用微调过的模型，使用此项则model参数无效|
|use_fast	|bool	|True	|使用FastTokenizer库加速(需要安装fast-tokenizer-python包)|

类调用(`__call__`方法)

|参数名	|类型	|示例	|描述|
|:---:  |:---:  |:---:  |:---:|
|text_str	|str	|'我叫小明，学号12345'	|文档识别出来的纯本文字符串|
|field	|str	|'人员域'	|文档所属领域|
|print_info	|bool	|False	|是否打印信息|

返回值

|参数名	|类型	|示例	|描述|
|:---:  |:---:  |:---:  |:---:|
|mg	|bool	|False	|是否是敏感文件|
|res	|dict	|{'姓名':[ '小明'], '学号':[ '12345']}	|抽取信息的schema键值对|

使用示例

```python
uie = Info_Extraction(
    schemas_dict= schemas_dict_education,
    model= 'uie-m-base',
    )

field1 = '人员域'
text1 = '我叫小明，学号12345'
extract_result1, mg1 = uie(text1, field1, print_info= True)

field2 = '财务域'
text2 = '发票金额1000元, 共1张'
extract_result2, mg2 = uie(text2, field2)

uie2 = Info_Extraction(
    schemas_dict= schemas_dict_education,
    model= 'uie-base',
    task_path = './checkpoint/model_best',
    use_fast = False,
    )

extract_result3, mg3 = uie2(text1, field1)
extract_result4, mg4 = uie2(text2, field2)
```

### 关键词匹配

此部分代码未作修改,仅封装函数

**`sensitive_word`函数**

调用参数

|参数名	|类型	|示例	|描述|
|:---:  |:---:  |:---:  |:---:|
|file_txt	|str	|"此件不公开"	|文档识别出来的纯本文字符串|
|print_info	|bool	|True	|是否打印信息|

返回值

|参数名	|类型	|示例	|描述|
|:---:  |:---:  |:---:  |:---:|
|have_sensitive_words	|bool	|True	|是否包含关键词|

使用示例

```python
text = '我叫小明，学号12345'
have_sensitive_words = sensitive_word(text, print_info= True)
```

### 分级分类接口

**`extraction_classify`函数**

|参数名	|类型	|示例	|描述|
|:---:  |:---:  |:---:  |:---:|
|field  |str    |'人员域'   |文档所属领域|
|extract_result |dict   |{'姓名':[ '小明'], '学号':[ '12345']}  |抽取信息的schema键值对|
|schema_dict    |dict   |schemas_dict_education_D = {'人员域': {'学生身份信息': ['姓名','学号'],}}  |分类所用字典，需要根据领域词schema设置|
|print_info	|bool	|True	|是否打印信息|

返回值

|参数名	|类型	|示例	|描述|
|:---:  |:---:  |:---:  |:---:|
|classify	|list	|['学生身份信息','学生校内信息']	|识别出来文档分类表中D列对应项（多个表示均有可能）|

使用示例

```python
text = '我叫小明，学号12345'
field = '人员域'
extract_result, mg = uie(text, field, print_info= True)
classify = extraction_classify(field, extract_result, print_info= True)
```


**初始代码版本 - 不再改动**

文件目录：
integration_optimize.py: 文件敏感识别全流程处理

utils.py: 基础函数库

update_keywords_table.py: 更新领域词和交集词表（创建领域词库、旧领域添加新词、添加新领域、删除某领域）

find_new_domain_file.py: 提取文件夹下所有文件的标题、判断内容是否包含机密词等，输出对应csv文件

doc_class_exp.py: 基于关键词的文本分类试验，确定模型效果（准确率等）(早期代码，目前无用)

configuration ：配置文件
* 20230531：当前批次文件的部分域名信息？
* 100000-small-modi.txt：腾讯词向量，10万词（删除了空格的向量）
* base.log：领域词基本信息
* domain_keywords.txt：各领域关键词表
* intersection.txt：领域词交集
* domain_mapping_trade_table.txt：已知域名的领域分类结果


需要关注的部分：
* integration_optimize.py（包含的是文件全流程的处理过程）
* utils.py（包含的是功能细节实现）
* update_keywords_table.py（包含的是领域词表的增删改查）
* 100000-small-modi.txt（可能需要优化包含的词）
* domain_keywords.txt
* intersection.txt


不用关注的部分：
* find_new_domain_file.py（只使用了全流程中部分功能，大致属于integration_optimize子集的应用）
* doc_class_exp.py（早期的基于领域词的领域分类效果测试，现在可能不大能直接用了）
* 20230531（福建方自己加的，与我们自己代码逻辑无关）
* domain_mapping_trade_table.txt（福建方自己的需求，与我们自己代码逻辑无关）
