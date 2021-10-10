import glob
import os
import numpy as np
import cv2
import Utils
import easyocr
import json

from Params_KTP_Template import NIKPosition, detailPosition, fotoPosition, ttdPosition
from Utils_KTP_OCR import convertNIK, convertDetail

resultImagesPath = "ResultImages\\"
resultJSONPath = "ResultJSON\\"

color = (0,0,255)
thickness = 2
reader = easyocr.Reader(['en'])

def main():
    for filename in glob.glob(resultImagesPath+"*.jpg"):
        img = cv2.imread(filename)
        print(filename, img.shape)

        imgRect = img.copy()
        NIKImage = Utils.cropRectangle(imgRect, NIKPosition)
        detailImage = Utils.cropRectangle(imgRect, detailPosition)
        fotoImage = Utils.cropRectangle(imgRect, fotoPosition)
        ttdImage = Utils.cropRectangle(imgRect, ttdPosition)

        NIKResult = reader.readtext(NIKImage)
        detailResult = reader.readtext(detailImage)

        nik = convertNIK(NIKResult)
        nama = convertDetail(detailResult, "nama")
        tempatL, tanggalL, bulanL, tahunL = convertDetail(detailResult, "ttl")
        jenisKelamin = convertDetail(detailResult, "jenisKelamin")
        golDarah = convertDetail(detailResult, "golDarah")
        jalan, rt, rw, kelDesa, kecamatan = convertDetail(detailResult, "alamat")
        agama = convertDetail(detailResult, "agama")
        statusKawin = convertDetail(detailResult, "statusKawin")
        pekerjaan = convertDetail(detailResult, "pekerjaan")
        kewarganegaraan = convertDetail(detailResult, "kewarganegaraan")
        berlakuHingga = convertDetail(detailResult, "berlakuHingga")

        ktpDict = writeJSONFile(nik, 
                                nama, 
                                tempatL, tanggalL, bulanL, tahunL, 
                                jenisKelamin, 
                                golDarah,
                                jalan, rt, rw, kelDesa, kecamatan,
                                agama, statusKawin, pekerjaan, kewarganegaraan, berlakuHingga)

        json_object = json.dumps(ktpDict, indent = 4)
        pathJSONSave = resultJSONPath + nik + "_" + nama.split()[0] + ".json"
        with open(pathJSONSave, "w") as outfile:
            outfile.write(json_object)
        
        pathFotoSave = resultImagesPath + "foto_" + nik + "_" + nama.split()[0] + ".jpg"
        pathTtdSave = resultImagesPath + "ttd_" + nik + "_" + nama.split()[0] + ".jpg"
        cv2.imwrite(pathFotoSave, fotoImage)
        cv2.imwrite(pathTtdSave, ttdImage)

def writeJSONFile(  nik, 
                    nama, 
                    tempatL, tanggalL, bulanL, tahunL, 
                    jenisKelamin, 
                    golDarah, 
                    jalan, rt, rw, kelDesa, kecamatan, 
                    agama, statusKawin, pekerjaan, kewarganegaraan, berlakuHingga):
    
    ktp_dict = {
        "nik" : nik,
        "nama" : nama,
        "ttl" : {
            "tempatL" : tempatL,
            "tanggalL" : tanggalL,
            "bulanL" : bulanL,
            "tahunL" : tahunL
        },
        "jenisKelamin" : jenisKelamin,
        "golDarah" : golDarah,
        "alamat" : {
            "jalan" : jalan,
            "rt" : rt,
            "rw" : rw,
            "kelDesa" : kelDesa,
            "kecamatan" : kecamatan
        },
        "agama" : agama,
        "statusPerkawinan" : statusKawin,
        "pekerjaan" : pekerjaan,
        "kewarganegaraan" : kewarganegaraan,
        "berlakuHingga" : berlakuHingga
    }
    return ktp_dict

if __name__ == '__main__':       
    # Calling main() function 
    main()