"""
测试uie模型的识别效果，根据已有实现方式，主要判断依据是识别出的敏感词的个数
"""

import csv
import os
import paddleocr
import logging
from utils import schemas_dict, read_file
from paddlenlp import Taskflow
import time

def init_uie(schemas_dict, model= 'uie-base', specify_field= None, use_fast= False):
    """
    初始化UIE模型字典
    可单独调用
    """
    uie_dict = {}
    if specify_field:# 测试时为了加速除了特定领域都不初始化
        uie_dict[specify_field] = {}
        for schema_type in schemas_dict[specify_field].keys():
            uie_dict[specify_field][schema_type] = Taskflow("information_extraction", 
                                                    model=model, 
                                                    schema=schemas_dict[specify_field][schema_type],
                                                    use_fast= use_fast)
    else:
        for domain in schemas_dict.keys():
            uie_dict[domain] = {}
            for schema_type in schemas_dict[domain].keys():
                uie_dict[domain][schema_type] = Taskflow("information_extraction", 
                                                        model=model, 
                                                        schema=schemas_dict[domain][schema_type])
    return uie_dict

def test_onefile(file_text:str, 
                 schemas:dict, # 领域下对应的schema字典
                 field_uie:list, # 领域下对应uie的taskflow
                 print_info:bool= True, # 打印即时信息
                 ):
    """
    输入纯文本和对应领域的schema字典（目前同领域下多个文件类型各有一个schema）
    field_uie为uie_dict下对应领域的uie模型
    """
    
    extraction_num = 0
    model_time = 0
    res = {}
    if file_text != '':
        if print_info:
            print("关键词和匹配情况")
        for schema_name in schemas.keys():
            strat_time = time.time()
            ie_result = field_uie[schema_name](file_text)
            end_time = time.time()
            model_time += (end_time - strat_time)
            if print_info:
                print("文件类型: ", schema_name)
                print("模型匹配用时: ", end_time - strat_time, "schema个数", len(schemas[schema_name]))
            if(ie_result[0]):
                schema_num = 0
                for schema, match_list in ie_result[0].items():
                    res[schema] = []
                    schema_num += len(match_list)
                    if print_info:
                        print("    ", schema, ":", end=" ", sep="")
                        for each_match in match_list:
                            print(each_match['text'], end=" ")
                        print()
                    for each_match in match_list:
                        res[schema].append(each_match['text'])
                extraction_num += schema_num
                if print_info:
                    print("该文件类型下匹配到个数: ", schema_num, "\n")
            else:
                if print_info:
                    print("该文件类型下未匹配到schema\n")
    if print_info:
        print("总共匹配到的个数: ", extraction_num)
    return extraction_num, model_time, len(schemas.keys()), res

def uie_unit(file_dir, file_name, field, uie_dict, ocr, table_engine, table_extract, table_res_dir, print_info= False):
    """
    对某个模型的uie在单个文件上测试，返回匹配数目
    """
    field_uie = uie_dict[field]
    schemas = schemas_dict[field]
    file_path = os.path.normpath(file_dir + os.sep + file_name)
    ori_filename, file_name_ocr, file_text = read_file(file_path, ocr, table_engine, table_extract, table_res_dir)
    match_num, model_time, rounds, result = test_onefile(file_text, schemas, field_uie, print_info)
    print(file_name, "\n总共匹配到的个数:", match_num, "模型匹配用时:", model_time, "模型运行次数:", rounds, "\n")
    return match_num, model_time, rounds, result

def prepare_file_list(file_dir, log_type= 'num', update= False):
    """
    检测并在某个目录下创建文件的全部文件名列表，防止os.listdir()函数顺序不稳定
    可单独调用，注意是否已有.csv文件
    """
    file_list_dir = os.path.normpath(file_dir + os.sep + "uie_test")
    file_list_path = os.path.normpath(file_list_dir + os.sep + "filelist.txt")
    if update or (not os.path.exists(file_list_path)):
        os.makedirs(file_list_dir, exist_ok= True)
    # 在文件目录下创建一次且仅一次文件列表，因为listdir顺序可能不稳定，所以用文件存储
        file_lists = os.listdir(file_dir)
        if "filelist.txt" in file_lists:
            file_lists.remove("filelist.txt")
        remove_list = []
        for each in file_lists:
            if os.path.isdir(file_dir + os.sep + each):
                remove_list.append(each)
            elif each.split(".")[-1] != "pdf":
                remove_list.append(each)
        for each in remove_list:
            file_lists.remove(each)
        file_list = open(file_list_path, 'w')
        print(*file_lists, file= file_list, sep="\n", end="")
        file_list.close()
        
    file_list = open(file_list_path, 'r')
    file_lists = file_list.read().splitlines()
    if "filelist.txt" in file_lists:
        file_lists.remove("filelist.txt")
    file_list.close()
    last_dir = os.path.normpath(file_dir).split(os.sep)[-1]
    log_path = os.path.normpath(last_dir + "-" + log_type + ".csv")
    if not os.path.exists(log_path):
        # 创建记录文件
        test_log = open(log_path, 'x', encoding='utf8')
        writer = csv.writer(test_log)
        writer.writerow(['文件路径', file_dir])
        writer.writerow(['模型-文件名'] + file_lists)
        test_log.close()
    return file_lists

def prepare_run(file_dir, domain, model= 'uie-base', log_type= 'num', table_extract= True, table_res_dir= r'table_ocr', use_fast= False):
    """
    读取文档
    加载模型并确定使用某个模型
    记录结果
    """
    file_lists = prepare_file_list(file_dir, log_type)

    # 加载模型
    uie_dict = init_uie(schemas_dict, model, domain, use_fast)
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")
    table_engine = paddleocr.PPStructure(table=False, ocr=False, show_log=True)

    # 使用uie进行匹配
    test_results = []
    total_files_num = len(file_lists)
    for i in range(total_files_num): # file_name = file_lists[i]
        match_num, model_time, rounds, result = uie_unit(file_dir, file_lists[i], domain, uie_dict, ocr, table_engine, table_extract, table_res_dir)
        if log_type == 'num':
            test_results.append(match_num)
        elif log_type == 'time' or log_type == 'fast-time':
            test_results.append(model_time)
        print(model,"模型", i+1, "/", total_files_num, "文件已完成\n")

    # 写入记录
    last_dir = os.path.normpath(file_dir).split(os.sep)[-1]
    log_path = os.path.normpath(last_dir + "-" + log_type + ".csv")
    test_log = open(log_path, 'a', encoding='utf8')
    writer = csv.writer(test_log)
    writer.writerow([model] + test_results)
    test_log.close()

def run_all_models():
    """
    测试所有模型并记录
    """
    models = ['uie-base', 'uie-medium', 'uie-mini', 'uie-micro', 'uie-nano', 'uie-m-base', 'uie-m-large', 'uie-x-base']
    file_dir = r'documents/领域分类测试-模拟文件/education'
    domain = 'sechool'
    for model in models:
        prepare_run(file_dir, domain, model, log_type= 'time')

def main():
    import uie_schemas
    from pprint import pprint

    file_path = r'documents\领域分类测试-模拟文件\education-细分\A人员域\2. 郭怀明任职资格申报表.pdf'
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")
    table_engine = paddleocr.PPStructure(table=False, ocr=False, show_log=True)
    
    ori_filename, file_name_ocr, file_text = read_file(file_path, ocr, table_engine, True, 'table_ocr')
    #file_text = uie_schemas.text
    field = 'sechool'

    uie_all = init_uie(schemas_dict, 'uie-base', field, True)
    match_num, model_time, rounds, result1 = test_onefile(file_text, schemas_dict[field], uie_all[field], print_info= False)
    #print("测试文件", "\n总共匹配到的个数:", match_num, "模型匹配用时:", model_time, "模型运行次数:", rounds, "\n")
    pprint(result1)

    schemas_dict_sum = uie_schemas.schema_assmeble(schemas_dict)
    uie_sum = init_uie(schemas_dict_sum, 'uie-base', field, True)
    match_num, model_time, rounds, result2 = test_onefile(file_text, schemas_dict_sum[field], uie_sum[field], print_info= False)
    #print("测试文件", "\n总共匹配到的个数:", match_num, "模型匹配用时:", model_time, "模型运行次数:", rounds, "\n")
    pprint(result2)

    print( result1 == result2)

def compare_schema():
    """
    对比多个schema分别匹配和将schema组合起来的效果
    """
    import uie_schemas
    models = ['uie-base', 'uie-medium']#, 'uie-mini', 'uie-micro', 'uie-nano', 'uie-m-base', 'uie-m-large']
    file_dir = r'documents\领域分类测试-模拟文件\education-细分\D校园生活'
    field = 'sechool'
    log_type = 'schemas对比'
    
    file_lists = prepare_file_list(file_dir, log_type)
    schemas_dict_assemble = uie_schemas.schema_assmeble(schemas_dict)

    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")
    table_engine = paddleocr.PPStructure(table=False, ocr=False, show_log=True)
    for model in models:
        # log按模型每行写入
        uie_all = init_uie(schemas_dict, model, field, True)
        uie_assemble = init_uie(schemas_dict_assemble, model, field, True)
        test_results = []

        # 模型匹配
        total_files_num = len(file_lists)
        for i in range(total_files_num): # file_name = file_lists[i]
            file_path = os.path.normpath(file_dir + os.sep + file_lists[i])
            ori_filename, file_name_ocr, file_text = read_file(file_path, ocr, table_engine, True, 'table_ocr')
            match_num1, model_time1, rounds1, result1 = test_onefile(file_text, schemas_dict[field], uie_all[field], print_info= False)
            match_num2, model_time2, rounds2, result2 = test_onefile(file_text, schemas_dict_assemble[field], uie_assemble[field], print_info= False)
            if result1 == result2:
                test_results.append(True)
            else:
                test_results.append(False)
            print(model,"模型", i+1, "/", total_files_num, "文件已完成\n")

        # 写入记录
        last_dir = os.path.normpath(file_dir).split(os.sep)[-1]
        log_path = (last_dir + "-" + log_type + ".csv")
        test_log = open(log_path, 'a', encoding='utf8')
        writer = csv.writer(test_log)
        writer.writerow([model] + test_results)
        test_log.close()

def compare_new_schemas(file_dir, schema_dict, models, log_type):
    """
    测试新的分类效果
    每个模型运行（减小内存占用）
    """
    """
    import uie_schemas
    models = ['uie-base', 'uie-medium', 'uie-mini', 'uie-micro', 'uie-nano', 'uie-m-base', 'uie-m-large']
    file_dir = r'documents/领域分类测试-模拟文件/education'
    schema_dict = uie_schemas.schemas_dict_education
    log_type = '新schema逐文件'
    """
    file_lists = prepare_file_list(file_dir, log_type)

    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")
    table_engine = paddleocr.PPStructure(table=False, ocr=False, show_log=True)
    #uie_model = {}
    #for model in models:
        # 初始化每个模型的Taskflow
    #    uie_model[model] = init_uie(schema_dict, model, use_fast= True, specify_field= None)

    # 每个文件 模型匹配
    #test_result = []
    total_files_num = len(file_lists)
    test_result = [[]] * total_files_num
    for model in models:
        uie_model = init_uie(schema_dict, model, use_fast= True, specify_field= None)
        for i in range(total_files_num): # file_name = file_lists[i]
            # 读取文件获得文本信息
            #test_result.append([])
            file_path = os.path.normpath(file_dir + os.sep + file_lists[i])
            ori_filename, file_name_ocr, file_text = read_file(file_path, ocr, table_engine, True, 'table_ocr')

            # 为每个文件单独创建展示结果的log
            file_log_path = os.path.normpath(file_dir + os.sep + log_type + file_lists[i] + '-新抽取效果.txt')
            file_log = open(file_log_path, 'a', encoding='utf8')
            #for model in models:
                #file_log.write(model + "模型\n")
            print("%s模型" % (model), file= file_log)
            match_totoal = 0
            for field in schema_dict.keys():
                match_num, model_time, rounds, result = test_onefile(file_text, schema_dict[field], uie_model[field], print_info= False)
                match_totoal += match_num
                #file_log.write("    " + field + " 下共匹配到" + match_num + "个:\n")
                print("    %s下共匹配到%d个:" % (field, match_num), file= file_log)
                for schema, matches in result.items():
                    #file_log.write("        " + schema + ":\n")
                    print("        %s: " % (schema), end="", file= file_log)
                    print(*matches, file= file_log)
            test_result[i].append(match_totoal)
            file_log.close()
            print(model, "模型", log_type, i+1, "/", total_files_num, "文件已完成\n")

    # 写入记录
    last_dir = os.path.normpath(file_dir).split(os.sep)[-1]
    log_path = (last_dir + "-" + log_type + ".csv")
    test_log = open(log_path, 'a', encoding='utf8')
    writer = csv.writer(test_log)
    for j in range(len(models)):
        model_result = []
        for i in range(total_files_num):
            model_result.append(test_result[i][j])
        writer.writerow([models[j]] + model_result)
    test_log.close()

if __name__ == '__main__':
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    
    compare_schema()