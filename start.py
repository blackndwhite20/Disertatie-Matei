# -*- coding: utf-8 -*-

import analiza_main
#import cicflow

url = input('\nIntroduceti calea catre fisierul de intrare .csv:  ')

#url = "/home/matei/WORK/pcaps/captura_8Iunie_pcap.pcap"
#url = "/home/matei/WORK/csv/captura_8Iunie_pcap.pcap_Flow.csv"
#url = "/home/matei/WORK/csv_cu_label/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
 #/home/matei/WORK/pcaps/2020-03-14-traffic-analysis-exercise.pcap

#cicflow.Apel_CICFLOW(url, "/home/matei/WORK/tt/")

analiza_main.main(url)