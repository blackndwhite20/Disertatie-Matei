# -*- coding: utf-8 -*-

import os
from subprocess import call
#call(["ls", "-l"])
#os.system("ls")
#print("outt")

def Apel_CICFLOW(inFile, outFile):
    #print(inFile)
    #os.system("/home/matei/WORK/CICFlowMeter-4.0/bin/cfm '" + inFile + "' '" + outFile + "'")
    call(["/home/matei/WORK/CICFlowMeter-4.0/bin/cfm", inFile, outFile])   #nu o sa mearga
    
    
    
#Apel_CICFLOW("/home/matei/WORK/pcaps/captura_8Iunie_pcap.pcap", "/home/matei/WORK/cicflowCSV/")   