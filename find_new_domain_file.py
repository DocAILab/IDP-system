"""
提取文件夹下所有文件的标题、判断内容是否包含机密词等，输出对应csv文件。
输入：
    file_path: 待处理文件夹路径（支持一级目录：即指定路径下即为待处理文件；支持二级目录：即指定路径下为多个文件夹，每个文件夹下为待处理文件）
    log_path: 结果存储路径
输出：
    一级目录时：每组文件（即自定义的batch_size值）输出一个csv文件
    二级目录时：每个文件夹输出一个csv文件
"""
import os
import logging
import paddleocr
from paddleocr import PPStructure
import multiprocessing
from multiprocessing import Pool
import csv
import time
from utils import sensitive_words, get_filename, get_text

logging.getLogger('ppocr').setLevel(logging.ERROR)
def chinese_filter(text):
    """
        提取文本中中文的内容。
    :param text: 待处理文本，str
    :return: 处理后文本，str
    """
    new_text = ''
    for c in text:
        if '\u4e00' <= c <= '\u9fff':
            new_text += c
    return new_text
def readconfiguration():
    mappering_table = {}
    trade_table = {}
    with open('./configuration/domain_mapping_trade_table.txt', 'r') as f:
        lists = f.readlines()
        for line in lists:
            info = line.replace('\n', '').split(',')
            trade_table[info[0]] = info[1]
    with open('./configuration/20230613/count_domain_url.txt', 'r') as f:
        lists = f.readlines()
        for line in lists:
            info = line.replace('\n', '').split('&!&')
            domain = info[0]
            log_id = info[1]
            fils = info[2]
            file_name = fils[0:fils.rfind('.')]
            request_num = info[3]
            server_ip = info[4]
            client_ip = info[5]
            host = info[6]
            company = info[7]
            if domain not in mappering_table:
                mappering_table[domain] = {}
            if domain not in mappering_table[domain]:
                mappering_table[domain][file_name] = []
            mappering_table[domain][file_name].append(request_num)
            mappering_table[domain][file_name].append(server_ip)
            mappering_table[domain][file_name].append(client_ip)
            if host=='':
                mappering_table[domain][file_name].append('-')
            else:
                mappering_table[domain][file_name].append(host)
            if company == '':
                mappering_table[domain][file_name].append('-')
            else:
                mappering_table[domain][file_name].append(company)
            #mappering_table[domain][file_name].append(host)
    return mappering_table,trade_table


def multiprocess(file_path, log_path, filenames, begin, end, filenames_list_len, domain_name=None,mappering_table=None,trade_table=None,,dir=None,date_str=None):
    count_domestic_path = '/usr/local/etc/ie_flow/parse_pdf/count_file_csv/merger_csv_' + dir + '_' + date_str+ '.csv'
    merger_ps = open(count_domestic_path, 'a+')
    writer_list = csv.writer(merger_ps)

    """
    多进程处理文件
    :param file_path: 文件夹路径
    :param log_path: 结果存储路径
    :param filenames: 文件名列表
    :param begin: 文件个数起始序号
    :param end: 文件个数结束序号
    :param filenames_list_len: 文件夹下总文件个数
    :param domain_name: 域名，可选
    :return: 输入一个csv文件，无返回值
    """
    with open(os.path.join(log_path, 'new_domain_file_{}-{}.csv'.format(begin, min(end, filenames_list_len))), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if domain_name is not None:
            writer.writerow([ '原文件名', '提取文件名', '是否包含机密词','一级域名','行业','请求次数','server_ip','client_ip','host', '公司'])
        else:
            writer.writerow(['原文件名', '提取文件名', '是否包含机密词'])
        for filename in filenames:
            ori_filename, new_filename = get_filename(os.path.join(file_path, filename), ocr=ocr, table_engine=table_engine)

            new_filename_only_chinese = chinese_filter(new_filename)
            text = get_text(os.path.join(file_path, filename), ocr=ocr, max_page=1)
            have_sensitive_words = any((word in text) for word in sensitive_words)
            content = []
            content.append(ori_filename.replace(',','|'))
            if new_filename_only_chinese != '':
                content.append(new_filename_only_chinese)
            else:
                content.append(new_filename.replace(',','|'))
            if have_sensitive_words:
                content.append(1)
            else:
                content.append(0)

            if domain_name is not None:
                content.append(domain_name)
                if domain_name in trade_table:
                    content.append(trade_table[domain_name])
                else:
                    content.append('-')
                if domain_name  in  mappering_table:

                    for nams in mappering_table[domain_name]:

                        if ori_filename in nams:
                            #print(ori_filename,nams)
                            for string in mappering_table[domain_name][nams]:
                                content.append(string)
                            break

            writer.writerow(content)
            writer_list.writerow(content)

    if domain_name is not None:
        print('Done: {}-{}-{}'.format(domain_name, begin, end))
    else:
        print('Done: {}-{}'.format(begin, end))
    merger_ps.close()


if __name__ == '__main__':
    # file_path = r'/usr/local/etc/ie_flow/partfile/urlfilelist/20230531/server_domestic_client_domestic'   # 待处理文件夹路径
    # log_path = r'/usr/local/etc/ie_flow/partfile_results//urlfilelist/20230531/server_domestic_client_domestic'  # 结果存储路径
    disr = ['server_domestic_client_abroad','server_abroad_client_domestic','server_domestic_client_domestic']
    date_str='20230613'
    for dir in disr:
        file_path = r'/usr/local/etc/ie_flow/partfile/urlfilelist/'+date_str+'/' +dir   # 待处理文件夹路径
        log_path = r'/usr/local/etc/ie_flow/partfile_results/urlfilelist/'+date_str+'/' +dir  # 结果存储路径
        count_domestic_path='/usr/local/etc/ie_flow/parse_pdf/count_file_csv/merger_csv_'+dir+'_'+date_str+'.csv'
        merger_ps = open(count_domestic_path,'a+')
        writer_list = csv.writer(merger_ps)
        writer_list.writerow(['原文件名', '提取文件名', '是否包含机密词', '一级域名', '行业', '请求次数', 'server_ip', 'client_ip', 'host', '公司'])
        merger_ps.close()

        mappering_table,trade_table = readconfiguration()
        table_engine = PPStructure(table=False, ocr=False, show_log=True)   # 初始化版面识别模型
        ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")  # 初始化OCR模型
        current_time = time.time()
        os.makedirs(log_path, exist_ok=True)
        filenames_list = os.listdir(file_path)

        print("程序开始！！！")
        if not os.path.isdir(os.path.join(file_path, filenames_list[0])):
            # 若file_path下为文件，则将文件分组，每组使用一个进程处理
            batch_size = 1000  # 每组处理的文件个数

            pool = multiprocessing.Pool(10)  # 使用10个进程
            for i in range(0, len(filenames_list), batch_size):
                filenames = filenames_list[i: i + batch_size]
                pool.apply_async(multiprocess, (file_path, log_path, filenames, i, i+batch_size, len(filenames_list)))
            pool.close()
            pool.join()
        else:
            # 若file_dir下为文件夹，则按文件夹分组，每组使用一个进程处理
            batch_size = -1  # 每个域名下最多处理的文件个数，-1时处理所有文件

            pool = multiprocessing.Pool(10)  # 使用10个进程
            for domain in filenames_list:
                domain_file_path = os.path.join(file_path, domain)
                domain_log_path = os.path.join(log_path, domain)
                os.makedirs(domain_log_path, exist_ok=True)
                filenames = os.listdir(domain_file_path)
                if batch_size > 0:
                    filenames = filenames[:batch_size]
                #multiprocess(domain_file_path, domain_log_path, filenames, 0, len(filenames), len(filenames), domain,mappering_table,trade_table,writer_list)
                pool.apply_async(multiprocess, (domain_file_path, domain_log_path, filenames, 0, len(filenames), len(filenames), domain,mappering_table,trade_table,dir,date_str))
            pool.close()
            pool.join()

        print('时间:{}'.format(time.time() - current_time))
