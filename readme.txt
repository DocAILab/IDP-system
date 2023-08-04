文件目录：
integration_optimize.py: 文件敏感识别全流程处理

utils.py: 基础函数库

update_keywords_table.py: 更新领域词和交集词表（创建领域词库、旧领域添加新词、添加新领域、删除某领域）

find_new_domain_file.py: 提取文件夹下所有文件的标题、判断内容是否包含机密词等，输出对应csv文件

doc_class_exp.py: 基于关键词的文本分类试验，确定模型效果（准确率等）(早期代码，目前无用)

configuration ：配置文件
    20230531：当前批次文件的部分域名信息？
    100000-small-modi.txt：腾讯词向量，10万词（删除了空格的向量）
    base.log：领域词基本信息
    domain_keywords.txt：各领域关键词表
    intersection.txt：领域词交集
    domain_mapping_trade_table.txt：已知域名的领域分类结果


需要关注的部分：
integration_optimize.py（包含的是文件全流程的处理过程）
utils.py（包含的是功能细节实现）
update_keywords_table.py（包含的是领域词表的增删改查）
100000-small-modi.txt（可能需要优化包含的词）
domain_keywords.txt
intersection.txt


不用关注的部分：
find_new_domain_file.py（只使用了全流程中部分功能，大致属于integration_optimize子集的应用）
doc_class_exp.py（早期的基于领域词的领域分类效果测试，现在可能不大能直接用了）
20230531（福建方自己加的，与我们自己代码逻辑无关）
domain_mapping_trade_table.txt（福建方自己的需求，与我们自己代码逻辑无关）



