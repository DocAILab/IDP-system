from paddlenlp import Taskflow
from uie_schemas import schemas_dict_education, schemas_dict_education_D, sensitive_words
#from test_dict import schemas_dict_education_test_short

class Info_Extraction:
    def __init__(self, schemas_dict= schemas_dict_education, model= 'uie-m-base', task_path= None, use_fast= True):
        """
        初始化UIE模型字典
        :param schemas_dict: 领域词字典
        :param model: 使用paddlenlp模型名
        :param task_path: 使用微调或自定义模型路径，使用该项model设置无效
        :param use_fast: 使用FastTokenizer库加速(需要安装fast-tokenizer-python包)
        """

        self.model = model
        self.task_path = task_path
        self.use_fast = use_fast
        
        # 整合schema
        assemble_schema = True # 是否整合schema,为减小显存占用而整合
        if assemble_schema:
            self.schema_assemble_dict = self.assemble_schema(schemas_dict)
        else:
            self.schema_assemble_dict = schemas_dict
        
        # 初始化模型
        self.uie_dict = {}

        load_all_uie = False # 是否一次性加载所有uie模型
        if load_all_uie:
            self.load_all_uie()
    
    @classmethod
    def assemble_schema(cls, schemas_dict):
        """
        整合schema
        :param schemas_dict: 领域词字典
        :return: 返回集成后的schema字典（减小显存占用）
        """
        schema_assemble_dict = {}
        for domain, domian_schemas in schemas_dict.items():
            tmp = set()
            for schema in domian_schemas.values():
                tmp.update(schema)
            schema_assemble_dict[domain] = {'assemble_schemas': list(tmp)}
        return schema_assemble_dict
    
    def load_all_uie(self):
        for field in self.schema_assemble_dict.keys():
            self.update_uie(field)

    def update_uie(self, field):
        """
        增量加载uie模型
        :param field: 文档所属领域
        :return: 返回是否加载成功
        """
        # 增量加载uie模型
        if field not in self.uie_dict.keys():
            if field in self.schema_assemble_dict.keys():
                if self.task_path:
                    self.uie_dict[field] = {}
                    for schema_type in self.schema_assemble_dict[field].keys():
                        self.uie_dict[field][schema_type] = Taskflow("information_extraction", 
                                                                task_path= self.task_path, 
                                                                schema=self.schema_assemble_dict[field][schema_type],
                                                                use_fast= self.use_fast)
                else:
                    self.uie_dict[field] = {}
                    for schema_type in self.schema_assemble_dict[field].keys():
                        self.uie_dict[field][schema_type] = Taskflow("information_extraction", 
                                                                model=self.model, 
                                                                schema=self.schema_assemble_dict[field][schema_type],
                                                                use_fast= self.use_fast)
                return True
            else:
                print("领域不存在")
                return False
    
    def __call__(self, text_str, field, print_info= False):
        """
        输入纯文本和对应领域的schema字典（目前同领域下多个文件类型各有一个schema）
        field_uie为uie_dict下对应领域的uie模型
        :param text_str: 文档识别出来的纯本文字符串
        :param field: 文档所属领域
        :param print_info: 运行时是否打印信息
        :return: 返回是否是敏感文件、匹配到的schema:关键词列表组成的字典
        """
        
        # 增量加载uie模型
        self.update_uie(field)

        extract_num = 0
        extract_result = {}
        if text_str != '':
            if print_info:
                print("关键词和匹配情况")
            for schema_type in self.schema_assemble_dict[field].keys():
                ie_result = self.uie_dict[field][schema_type](text_str)
                #if print_info:
                    #print("文件类型: ", schema_type)
                if(ie_result[0]):
                    schema_num = 0
                    for schema, match_list in ie_result[0].items():
                        extract_result[schema] = []
                        schema_num += len(match_list)
                        if print_info:
                            print("    ", schema, ":", end=" ", sep="")
                            for each_match in match_list:
                                print(each_match['text'], end=" ")
                            print()
                        for each_match in match_list:
                            extract_result[schema].append(each_match['text'])
                    extract_num += schema_num
                    if print_info:
                        print("该类型下匹配到个数: ", schema_num, "\n")
                else:
                    if print_info:
                        print("该类型下未匹配到schema\n")
        if print_info:
            print("总共匹配到的个数: ", extract_num)
        
        # 判断是否敏感
        schema_num = sum([len(schemas) for schemas in self.schema_assemble_dict[field].values()])
        schema_num_avg = schema_num / len(self.schema_assemble_dict[field].keys())
        rate = len(extract_result) / schema_num_avg
        if print_info:
            print("平均匹配率: ", rate)
        mg = True if (extract_num > 10 or rate > 0.5) else False

        return extract_result, mg

def sensitive_word(file_txt, print_info= False):
    # 基于文件内容的信息抽取
    ## 机密词匹配
    have_sensitive_words = any((word in file_txt) for word in sensitive_words)
    if print_info:
        if have_sensitive_words:
            print('是否包含机密词 : 是')
        else:
            print('是否包含机密词 : 否')
    return have_sensitive_words

def extraction_classify(field, extract_result, schema_dict= schemas_dict_education_D, print_info= False):
    """
    根据抽取结果和scheam分类
    :param field: 文档所属领域
    :param extract_result: 抽取结果
    :param schema_dict: 领域词字典
    :return: 返回分类结果
    """

    classify_res = {}
    max_rate = 0
    for schema_type, schemas in schema_dict[field].items():
        match_num = 0
        repeat_num = 0
        for match_schema, match_words in extract_result.items():
            if match_schema in schemas:
                match_num += 1
                repeat_num += len(match_words)/10
        #classify_res[schema_type] = match_num + repeat_num
        classify_res[schema_type] = match_num / len(schemas) + repeat_num
        if classify_res[schema_type] > max_rate:
            max_rate = classify_res[schema_type]
    
    classify = []
    for schema_type, rate in classify_res.items():
        if rate == max_rate:
            classify.append(schema_type)

    if print_info:
        print("分类匹配")
        for schema_type, rate in classify_res.items():
            print("   ", schema_type, "匹配值: ", rate)
        print("分类结果: ", *classify, sep= " ")
    return classify

if __name__ == '__main__':
    # debug测试以及使用示例
    
    # 初始化uie模型,加载字典
    uie = Info_Extraction()

    # 获得领域和文本（由上一步ocr得到）
    field = '财务域'
    text = "有效张数51，作废张数2，实开金额1000，金额限制1000，票本序号123456，姓名小明"

    field2 = '人员域'
    text2 = "姓名小明，性别男，年龄18，身份证号123456789，电话号码123456789，家庭住址北京市海淀区"
    
    # 函数使用示例
    extract_result, mg = uie(text, field, print_info= True)
    have_sensitive_words = sensitive_word(text, print_info= True)
    classify = extraction_classify(field, extract_result, print_info= True)

    extract_result2, mg2 = uie(text2, field2)
    have_sensitive_words2 = sensitive_word(text2)
    classify2 = extraction_classify(field2, extract_result2)

    # 打印结果
    from pprint import pprint
    #pprint(uie.schema_assemble_dict)
    print("是否敏感:", "是" if mg else "否")
    #pprint(extract_result)
    print("是否含有敏感关键词:", "是"if have_sensitive_words else "否")
    print("分类结果:", *classify, sep= " ")

    print()

    print("是否敏感:", "是" if mg2 else "否")
    #pprint(extract_result2)
    print("是否含有敏感关键词:", "是" if have_sensitive_words2 else "否")
    print("分类结果:", *classify2, sep= " ")