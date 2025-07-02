"""
@Created Time : 2022/11/22
@Author  : LiYao
@FileName: BehaviorExtractor.py
@Description:nfstream提取行为特征
@Modified:
    :First modified
    :Modified content:040312整理上传
"""

'''
VNAT数据集标签编号
Stream 1 vimeo netflix youtube
Chat 2 skype
C&C 3 ssh rdp
SFTP 4 sftp rsync scp
VoIP 5 zoiper

VNAT数据集的筛选
voip:5
zoiper:RIP的端口是8000，属于udp，混有DNS,所以是'ip and (tcp or udp) and (!port 53)'

文件传输：4
rsync:udp全是DNS
new1 无ssh(删去)
1 有ssh

sftp: udp全是DNS
1 2 3 new1有ssh
new2 无ssh(删去)

scp: udp全是DNS
1 long1 有ssh
new1 无ssh

c2:3
ssh:tcp ssh DNS
rdp: tcp tls rdpudp

chat:2
skype: tcp tls

流媒体：1
vimeo:tcp tls
netflix:tcp tls
youtube:tcp tls

'''


"""
USTC-TFC16
BitTorrent1 facetime2 gmail3 outloo4 Skyp5 SMB6 weibo7 WOW8

"""


"""
考虑全部的数据包，并csv就直接叠加
"""


from nfstream import NFStreamer
import pandas as pd
import os
from NfstreamPlugin import TrafficExtractorPlugin


class BehaviorExtractor(object):
    def __init__(self):
        """"""
        self.pcap_path = 'USTC'
        #self.pcap_path = 'F:\\dataset\\111\\'
        self.bpf_filter = 'ip and (tcp or udp) and (!port 53 and !port 5353 and !port 5355) and !host 239.255.255.250'
        # 过滤非 IP、TCP、UDP 流量
        # 排除 DNS 流量（53, 5353, 5355 端口）
        # 排除 SSDP 广播流量（239.255.255.250）

    #     pass
    #
    def extract(self):
        """将指定目录pcap中的流量组流，并提取需要的特征至csv文件"""

        print('PCAP PROCESSING...\n')
        pcap_list = []
        for src_pcap in os.listdir(self.pcap_path):
            if os.path.splitext(src_pcap)[1] == '.pcapng' or os.path.splitext(src_pcap)[1] == '.pcap':
                pcap_list.append(src_pcap)

        print(pcap_list)

        # #VNAT数据集
        # rdp（远程桌面）：只保留 3389 端口
        # ssh 或 4 开头的：只保留 TCP 流量
        # 5 开头的：保留 TCP 和 UDP，但排除 DNS
        # for index, pcap in enumerate(pcap_list):
        #     if pcap[9] == 'r' and pcap[10] == 'd' and pcap[11] == 'p':
        #         self.bpf_filter = 'ip and (tcp port 3389 or udp port 3389)'
        #     elif pcap[9] == 's' and pcap[10] == 's' and pcap[11] == 'h' or pcap[0] == '4':
        #         self.bpf_filter = 'ip and (tcp)'
        #     elif pcap[0] == '5':
        #         self.bpf_filter = 'ip and (tcp or udp) and (!port 53)'
        #     else:
        #         self.bpf_filter = 'ip and (tcp or udp) and (!port 53 and !port 5353 and !port 5355) and !host 239.255.255.250'

        # #USTC数据集
        for index, pcap in enumerate(pcap_list):
            if pcap[0]=='1':
                self.bpf_filter='ip and (tcp) and !port 80'
            else:
                self.bpf_filter='ip and (tcp)'



        # ZEAC自建数据集
        # for index, pcap in enumerate(pcap_list):
        #     if pcap[0]=='2':
        #         self.bpf_filter='ip and (tcp) and !port 80'
        #     else:
        #         self.bpf_filter='ip and (tcp)'

        # iscx vpn
        # for index, pcap in enumerate(pcap_list):
        #     filename = pcap  # 如果 pcap 是文件名字符串
        #     if filename.split('_')[0]=='7' or filename == '11_9_scp1.pcapng' or filename=='13_11_skype_file1.pcap' or filename=='16_11_skype_video2b.pcapng':
        #         self.bpf_filter = 'ip and tcp and not port 80'
        #     elif filename.split('_')[0]=='10':
        #         self.bpf_filter = self.bpf_filter = 'ip and tcp and not (tcp port 21 or tcp port 6060 or tcp port 6061 or tcp port 6062 or tcp port 6063)'
        #     elif filename.split('_')[0] == '17':
        #         # self.bpf_filter = 'ip and ((tcp.port != 80 and tcp.port != 53) or (udp.port != 53 and udp.port != 5355 and udp.port != 137 and udp.port != 123))'
        #         self.bpf_filter = 'ip and (tcp or udp) and (!port 53)'
        #     else:
        #         self.bpf_filter = 'ip and tcp'

        # iscx tor
        # for index, pcap in enumerate(pcap_list):
        #     filename = pcap  # 如果 pcap 是文件名字符串
        #     if filename.split('_')[
        #         0] == '7' or filename == '11_9_scp1.pcapng' or filename == '13_11_skype_file1.pcap' or filename == '16_11_skype_video2b.pcapng':
        #         self.bpf_filter = 'ip and tcp and not port 80'
        #     elif filename.split('_')[0] == '10':
        #         self.bpf_filter = self.bpf_filter = 'ip and tcp and not (tcp port 21 or tcp port 6060 or tcp port 6061 or tcp port 6062 or tcp port 6063)'
        #     elif filename.split('_')[0] == '17':
        #         # self.bpf_filter = 'ip and ((tcp.port != 80 and tcp.port != 53) or (udp.port != 53 and udp.port != 5355 and udp.port != 137 and udp.port != 123))'
        #         self.bpf_filter = 'ip and (tcp or udp) and (!port 53)'
        #     else:
        #         self.bpf_filter = 'ip and tcp'

            my_streamer = NFStreamer(source=os.path.join(self.pcap_path, pcap),
                                     decode_tunnels=True,
                                     bpf_filter=self.bpf_filter,
                                     promiscuous_mode=True,
                                     snapshot_length=1536,
                                     idle_timeout=1200,  # 无数据流 20 分钟自动超时。
                                     active_timeout=18000,  # 最长 5 小时超时。
                                     accounting_mode=0,
                                     udps=TrafficExtractorPlugin(),
                                     n_dissections=0,
                                     statistical_analysis=True,
                                     splt_analysis=0,
                                     n_meters=0,
                                     performance_report=0)
            print(f"Processing: {pcap}")
            print(f"Using BPF filter: {self.bpf_filter}")
            print(f"Streamer type: {type(my_streamer)}")
            flow_count = sum(1 for _ in my_streamer)
            print(f"Flow count for {pcap}: {flow_count}")

            df = my_streamer.to_pandas()

            label = int(pcap.split('_')[0])
            df['label'] = label



            df.to_csv(self.pcap_path + '/' + pcap + '.csv', encoding='utf-8', index=False)

        '''合并生成的多个csv文件'''
        csv_list = []
        data_list = []
        for csv in os.listdir(self.pcap_path):
            if os.path.splitext(csv)[1] == '.csv':
                csv_list.append(self.pcap_path + '/' + csv)

        '''拼接所有的csv文件'''
        for _ in csv_list:
            df = pd.read_csv(_).sample(frac=1.0)
            df_data = df.iloc[:]
            data_list.append(df_data)
        res = pd.concat(data_list)
        # print(len(res['label']=='4'))
        # 这里如果已经存在了csv文件，可能会直接追加


        res.to_csv('merge_USTC_4_class_hex_data_mit.csv.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    extract_object = BehaviorExtractor()
    extract_object.extract()


